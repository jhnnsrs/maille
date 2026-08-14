"""Writing a collection into a store, in the one order that is safe.

**The manifest lands last.** A prefix has no atomic "upload finished" flag: a ``PutObject``
either happened or it did not, but a tree is a sequence of writes that can stop anywhere. So
every file the manifest refers to is written before the manifest is, and a prefix without a
manifest is an interrupted write rather than a collection. That ordering is the entire
completion protocol, and it is why this module exists rather than a loop at the call site --
a caller who writes the manifest first has built something that registers cleanly and fails
later, on a reader, with no way to tell an interrupted write from a corrupt one.

The same tree lands on a local directory and in an S3 prefix; see :mod:`maille.store`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from maille.build import MeshCollection, build_collection
from maille.frames import table_to_parquet, validate_columns
from maille.manifest import (
    CELL_CATALOG_PATH,
    MANIFEST_NAME,
    OBJECT_CATALOG_PATH,
    Manifest,
    level_part_path,
)
from maille.sources import MeshSource
from maille.store import MailleStore, join, put_bytes

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

#: How large a level's Parquet part is allowed to get before the level is split across parts.
#: A level is a directory precisely so this can happen without the layout changing shape.
DEFAULT_MAX_PART_BYTES = 512 * 1024 * 1024


def _plan_parts(shard: pa.Table, max_part_bytes: int) -> list[pa.Table]:
    """Split one level's rows into parts, budgeting on the blob bytes each row carries.

    Budgeting on the blobs rather than on a row count is what keeps parts even: a cell holding
    one small object and a cell holding two hundred differ by orders of magnitude in bytes and
    not at all in rows.
    """
    if shard.num_rows == 0:
        return [shard]

    positions = shard.column("positions").to_pylist()
    indices = shard.column("indices").to_pylist()
    sizes = [len(p or b"") + len(i or b"") for p, i in zip(positions, indices)]
    if sum(sizes) <= max_part_bytes:
        return [shard]

    parts: list[pa.Table] = []
    start = 0
    running = 0
    for row, size in enumerate(sizes):
        # A single row larger than the budget still goes in a part of its own rather than
        # being split -- a cell is the smallest thing a reader fetches.
        if running and running + size > max_part_bytes:
            parts.append(shard.slice(start, row - start))
            start, running = row, 0
        running += size
    parts.append(shard.slice(start, len(sizes) - start))
    return parts


def write_collection(
    collection: MeshCollection,
    store: MailleStore,
    prefix: str = "",
    *,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
) -> Manifest:
    """Write a built collection into ``store`` under ``prefix``, manifest last.

    Returns the manifest as written -- which is not always the one the collection was built
    with: ``files`` is rewritten to name the parts that actually landed, so a reader that
    cannot list a prefix (an HTTP store, say) can still find every level.
    """
    validate_columns(collection.cell_catalog, "cell_catalog")
    validate_columns(collection.object_catalog, "object_catalog")
    for _, shard in collection.shards:
        validate_columns(shard, "geometry")

    written: dict[str, Any] = {"cells": CELL_CATALOG_PATH, "objects": OBJECT_CATALOG_PATH, "levels": {}}

    put_bytes(store, join(prefix, CELL_CATALOG_PATH), table_to_parquet(collection.cell_catalog))
    put_bytes(store, join(prefix, OBJECT_CATALOG_PATH), table_to_parquet(collection.object_catalog))

    for level, shard in sorted(collection.shards, key=lambda item: item[0]):
        paths: list[str] = []
        for number, part in enumerate(_plan_parts(shard, max_part_bytes)):
            path = level_part_path(level, number)
            put_bytes(store, join(prefix, path), table_to_parquet(part))
            paths.append(path)
        written["levels"][str(level)] = paths

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
) -> Manifest:
    """Write a collection without blocking the event loop.

    Parquet serialization is CPU-bound and obstore's ``put`` releases the GIL, so the whole
    write is handed to a worker thread rather than interleaved -- which also keeps the
    manifest-last ordering trivially true instead of a scheduling question.
    """
    return await asyncio.to_thread(
        write_collection, collection, store, prefix, max_part_bytes=max_part_bytes
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
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
) -> Manifest:
    """Build a collection from objects and write it, in one call.

    The convenience form of :func:`maille.build_collection` followed by
    :func:`write_collection`. Build them separately when you want to inspect or check the
    frames before spending the writes.
    """
    from maille.manifest import CODEC_MESHOPT

    collection = build_collection(
        objects,
        axes=axes,
        cell_size=cell_size,
        levels=levels,
        codec=codec or CODEC_MESHOPT,
    )
    return write_collection(collection, store, prefix, max_part_bytes=max_part_bytes)


__all__ = ["DEFAULT_MAX_PART_BYTES", "awrite_collection", "write_collection", "write_meshes"]
