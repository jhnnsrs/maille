"""Writing a collection into a store, in the one order that is safe.

**The manifest lands last.** A prefix has no atomic "upload finished" flag: a ``PutObject``
either happened or it did not, but a tree is a sequence of writes that can stop anywhere. So
every file the manifest refers to is written before the manifest is, and a prefix without a
manifest is an interrupted write rather than a collection. That ordering is the entire
completion protocol, and it is why this module exists rather than a loop at the call site --
a caller who writes the manifest first has built something that registers cleanly and fails
later, on a reader, with no way to tell an interrupted write from a corrupt one.

The same tree lands on a local directory and in an S3 prefix; see :mod:`maille.stores`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from maille.build import MeshCollection, build_collection
from maille.frames import (
    DEFAULT_ROW_GROUP_BYTES,
    blob_sizes,
    plan_byte_chunks,
    table_to_chunked_parquet,
    table_to_parquet,
    validate_columns,
)
from maille.manifest import (
    CELL_CATALOG_PATH,
    MANIFEST_NAME,
    OBJECT_CATALOG_PATH,
    Decimation,
    FileEntry,
    Manifest,
    level_part_path,
)
from maille.simplifiers import Simplifier
from maille.sources import MeshSource
from maille.stores import MailleStore, join, put_bytes

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

#: How large a level's Parquet part is allowed to get before the level is split across parts.
#: A level is a directory precisely so this can happen without the layout changing shape.
DEFAULT_MAX_PART_BYTES = 512 * 1024 * 1024


def _plan_parts(shard: pa.Table, max_part_bytes: int) -> list[pa.Table]:
    """Split one level's rows into parts, budgeting on the blob bytes each row carries.

    The same byte-budgeted grouping that cuts a part into row groups, one scale up -- see
    :func:`maille.frames.plan_byte_chunks`, which both go through.
    """
    if shard.num_rows == 0:
        return [shard]
    sizes = blob_sizes(shard)
    if sum(sizes) <= max_part_bytes:
        return [shard]
    return [shard.slice(start, count) for start, count in plan_byte_chunks(sizes, max_part_bytes)]


def _locate(shard: pa.Table, part: int, chunks: Sequence[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Map each cell in a written part to the part and row group a reader must fetch for it."""
    levels = shard.column("level").to_pylist()
    cells = shard.column("cell").to_pylist()
    sizes = blob_sizes(shard)
    found: dict[tuple[int, int], tuple[int, int, int]] = {}
    for group, (start, count) in enumerate(chunks):
        for row in range(start, start + count):
            found[(int(levels[row]), int(cells[row]))] = (part, group, sizes[row])
    return found


def _with_locators(catalog: pa.Table, locators: Mapping[tuple[int, int], tuple[int, int, int]]) -> pa.Table:
    """Fill the cell catalog's ``part`` / ``row_group`` / ``blob_bytes`` from what was written.

    A cell the geometry does not hold keeps its nulls rather than being dropped or defaulted:
    a catalog row without a locator is a real inconsistency, and :func:`maille.verify` should
    be the thing that says so -- not a silent zero here that points a reader at row group 0 of
    part 0 and hands it the wrong cell.
    """
    import pyarrow as pa

    keys = list(zip(catalog.column("level").to_pylist(), catalog.column("cell").to_pylist()))
    resolved = [locators.get((int(level), int(cell))) for level, cell in keys]
    columns = {
        "part": pa.array([None if item is None else item[0] for item in resolved], type=pa.int32()),
        "row_group": pa.array([None if item is None else item[1] for item in resolved], type=pa.int32()),
        "blob_bytes": pa.array([None if item is None else item[2] for item in resolved], type=pa.int64()),
    }
    filled = catalog
    for name, values in columns.items():
        filled = filled.set_column(filled.schema.get_field_index(name), catalog.schema.field(name), values)
    return filled


def write_collection(
    collection: MeshCollection,
    store: MailleStore,
    prefix: str = "",
    *,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
) -> Manifest:
    """Write a built collection into ``store`` under ``prefix``, manifest last.

    Returns the manifest as written -- which is not the one the collection was built with:
    ``files`` is rewritten to name the parts that actually landed *and how long each one is*,
    so a reader that can neither list nor stat a prefix (an HTTP store, say) can still find
    every level and range-read inside it.

    **The geometry goes first, then the catalog that points into it, then the manifest.** The
    cell catalog cannot be written before the geometry any more: it carries the part and row
    group holding each cell, and those are facts about bytes that do not exist until they have
    been serialized. That ordering is also strictly safer than the reverse -- an interrupted
    write can now leave a catalog pointing at nothing only if it also leaves no manifest, which
    is already the signal for an unfinished collection.
    """
    validate_columns(collection.cell_catalog, "cell_catalog")
    validate_columns(collection.object_catalog, "object_catalog")
    for _, shard in collection.shards:
        validate_columns(shard, "geometry")

    levels: dict[str, list[dict[str, Any]]] = {}
    locators: dict[tuple[int, int], tuple[int, int, int]] = {}

    for level, shard in sorted(collection.shards, key=lambda item: item[0]):
        entries: list[FileEntry] = []
        for number, part in enumerate(_plan_parts(shard, max_part_bytes)):
            body, chunks = table_to_chunked_parquet(part, row_group_bytes=row_group_bytes)
            path = level_part_path(level, number)
            put_bytes(store, join(prefix, path), body)
            entries.append(FileEntry(path=path, size=len(body), row_groups=len(chunks)))
            locators.update(_locate(part, number, chunks))
        levels[str(level)] = [entry.to_dict() for entry in entries]

    catalog = table_to_parquet(_with_locators(collection.cell_catalog, locators))
    put_bytes(store, join(prefix, CELL_CATALOG_PATH), catalog)
    objects = table_to_parquet(collection.object_catalog)
    put_bytes(store, join(prefix, OBJECT_CATALOG_PATH), objects)

    written: dict[str, Any] = {
        "cells": FileEntry(path=CELL_CATALOG_PATH, size=len(catalog)).to_dict(),
        "objects": FileEntry(path=OBJECT_CATALOG_PATH, size=len(objects)).to_dict(),
        "levels": levels,
    }

    # Everything the manifest names now exists. Only now is the collection a collection.
    manifest = replace(collection.manifest, files=written)
    put_bytes(store, join(prefix, MANIFEST_NAME), manifest.to_json())
    return manifest


async def awrite_collection(
    collection: MeshCollection,
    store: MailleStore,
    prefix: str = "",
    *,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
) -> Manifest:
    """Write a collection without blocking the event loop.

    Parquet serialization is CPU-bound and obstore's ``put`` releases the GIL, so the whole
    write is handed to a worker thread rather than interleaved -- which also keeps the
    manifest-last ordering trivially true instead of a scheduling question.
    """
    return await asyncio.to_thread(
        write_collection,
        collection,
        store,
        prefix,
        max_part_bytes=max_part_bytes,
        row_group_bytes=row_group_bytes,
    )


def write_meshes(
    objects: Mapping[int, MeshSource],
    store: MailleStore,
    prefix: str = "",
    *,
    axes: Sequence[str] | None = None,
    cell_size: Sequence[int] | None = None,
    levels: int = 3,
    codec: str | None = None,
    compression: str | None = None,
    simplifier: Simplifier | str | None = None,
    decimation: Decimation | None = None,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
) -> Manifest:
    """Build a collection from objects and write it, in one call.

    The convenience form of :func:`maille.build_collection` followed by
    :func:`write_collection`. Build them separately when you want to inspect or check the
    frames before spending the writes. ``simplifier`` and ``decimation`` are how the coarse
    levels are made; see :func:`maille.build_collection`.
    """
    from maille.manifest import CODEC_NONE, COMPRESSION_NONE

    collection = build_collection(
        objects,
        axes=axes,
        cell_size=cell_size,
        levels=levels,
        codec=codec or CODEC_NONE,
        compression=compression or COMPRESSION_NONE,
        simplifier=simplifier,
        decimation=decimation,
    )
    return write_collection(
        collection, store, prefix, max_part_bytes=max_part_bytes, row_group_bytes=row_group_bytes
    )


__all__ = ["DEFAULT_MAX_PART_BYTES", "awrite_collection", "write_collection", "write_meshes"]
