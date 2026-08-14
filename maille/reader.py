"""Opening a collection back up: the manifest, the two catalogs, and a decoded cell.

A store that describes itself is only half a format if nothing can open it, so this is the
other half. It is also what makes the round-trip check real: the encoder is proven by decoding
its own output, not by asserting on the bytes it happened to produce.

Reading is lazy and in that order on purpose. The manifest is one small object and tells you
whether the collection is readable at all; the cell catalog is the next smallest and answers
*which cells, at which level* without opening a single geometry file; only then does anything
fetch a blob.

And a blob is fetched **by the row group holding it**, not by the level containing it. Every
cell-catalog row names the part and row group its geometry sits in, and a part is opened
through :class:`maille.stores.StoreFile` -- a seekable view over the store -- so pyarrow reads
the footer once and then only the column chunks of the one row group asked for. That is what
makes the format's claim true in the reader as well as on paper: drawing forty cells costs
forty row groups, not the levels they came from.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from maille.codecs import decode_indices, decode_positions
from maille.errors import FormatError, UnfinishedCollectionError
from maille.frames import parquet_to_table
from maille.manifest import (
    CELL_CATALOG_PATH,
    MANIFEST_NAME,
    OBJECT_CATALOG_PATH,
    Encoding,
    FileEntry,
    Grid,
    Manifest,
    level_prefix,
)
from maille.octree import cell_box, morton_decode
from maille.sources import Mesh
from maille.stores import (
    MailleStore,
    StoreFile,
    aget_bytes,
    aget_range_bytes,
    get_bytes,
    join,
    list_paths,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa


def _optional_int(column: Sequence[Any], row: int) -> int | None:
    """One nullable integer out of a column that a manifest may not have written at all."""
    if row >= len(column):
        return None
    value = column[row]
    return None if value is None else int(value)


def _present(columns: Mapping[str, Sequence[Any]], name: str, row: int) -> Any:  # noqa: ANN401
    """One value out of a column the format declares non-null, or a refusal naming it.

    Every catalog column but ``part``/``row_group``/``blob_bytes`` is non-null in the schema a
    server checks, so a null here is a catalog maille did not write. Saying which column and
    which row beats the ``TypeError: int() argument must be...`` that reading it blindly gives.
    """
    value = columns[name][row]
    if value is None:
        raise FormatError(
            f"Row {row} of this catalog holds null in `{name}`, which the format declares non-null. "
            f"A catalog with a null there was not written by maille, and reading it would silently "
            f"invent a value for a cell a planner is about to fetch."
        )
    return value


def _int(columns: Mapping[str, Sequence[Any]], name: str, row: int) -> int:
    """One non-null integer out of a catalog column."""
    return int(_present(columns, name, row))


def _float(columns: Mapping[str, Sequence[Any]], name: str, row: int) -> float:
    """One non-null float out of a catalog column."""
    return float(_present(columns, name, row))


def _buffer(body: bytes) -> Any:  # noqa: ANN401
    """An in-memory random-access file, for a part whose length nothing recorded."""
    import pyarrow as pa

    return pa.BufferReader(body)


def _row_group_span(metadata: Any, row_group: int) -> tuple[int, int] | None:  # noqa: ANN401
    """The ``(start, length)`` of every byte one row group occupies in its part.

    A row group's column chunks are written consecutively, so the span from the first chunk's
    offset to the end of the last is exactly what a reader has to have -- and one window is
    what an object store would rather serve than a dozen adjacent ones.

    ``None`` when the metadata does not say, which leaves the caller to fetch lazily instead of
    guessing a range and reading the wrong bytes.
    """
    try:
        group = metadata.row_group(int(row_group))
    except Exception:  # noqa: BLE001 - any malformed metadata means "cannot prefetch"
        return None
    starts: list[int] = []
    ends: list[int] = []
    for index in range(group.num_columns):
        column = group.column(index)
        start = column.dictionary_page_offset or column.data_page_offset
        if start is None:
            return None
        starts.append(int(start))
        ends.append(int(start) + int(column.total_compressed_size))
    if not starts:
        return None
    return min(starts), max(ends) - min(starts)


@dataclass(frozen=True)
class CellEntry:
    """One row of the cell catalog: everything a planner needs, with no geometry fetched."""

    level: int
    cell: int
    vertex_count: int
    index_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    lod_error: float
    object_count: int
    child_mask: int
    #: Where this cell's geometry sits: the part of its level, and the row group inside it. A
    #: reader fetches exactly that row group. ``None`` on a collection whose manifest named no
    #: locator, in which case the reader falls back to reading the part whole.
    part: int | None = None
    row_group: int | None = None
    #: How many bytes of encoded geometry this cell carries -- what lets a planner budget a
    #: *fetch* rather than a face count.
    blob_bytes: int | None = None

    @property
    def triple(self) -> tuple[int, int, int]:
        """The ``(i, j, k)`` cell index this cell's Morton code encodes."""
        return morton_decode(self.cell)

    @property
    def face_count(self) -> int:
        """How many triangles this cell holds."""
        return self.index_count // 3

    def children(self) -> list[tuple[int, int]]:
        """The ``(level, cell)`` keys of the children that hold geometry.

        Read straight off ``child_mask``, so descending the octree costs no listing and no
        catalog scan.
        """
        from maille.octree import morton_encode_one

        if self.level == 0 or not self.child_mask:
            return []
        i, j, k = self.triple
        found: list[tuple[int, int]] = []
        for octant in range(8):
            if not (self.child_mask >> octant) & 1:
                continue
            dx, dy, dz = octant & 1, (octant >> 1) & 1, (octant >> 2) & 1
            found.append((self.level - 1, morton_encode_one((2 * i + dx, 2 * j + dy, 2 * k + dz))))
        return found


@dataclass(frozen=True)
class ObjectEntry:
    """One row of the object catalog: where an object is, without scanning any geometry."""

    object_id: int
    ordinal: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    vertex_count: int
    index_count: int
    cells: tuple[tuple[int, int], ...]

    def cells_at(self, level: int) -> tuple[int, ...]:
        """The cells holding this object at one level."""
        return tuple(cell for cell_level, cell in self.cells if cell_level == level)


@dataclass(frozen=True)
class DecodedCell:
    """A cell's geometry, decoded: the concatenated vertices and faces, and who owns what.

    ``object_vertex_offsets`` and ``object_index_offsets`` are **start offsets** into those
    concatenated arrays, ascending, one per entry of ``object_ids``. An object's extent is
    implied by the next start, or by the totals for the last one.
    """

    level: int
    cell: int
    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]
    object_ids: tuple[int, ...]
    object_ordinals: tuple[int, ...]
    object_vertex_offsets: tuple[int, ...]
    object_index_offsets: tuple[int, ...]

    def __len__(self) -> int:
        """How many objects this cell holds."""
        return len(self.object_ids)

    def mesh(self) -> Mesh:
        """The whole cell as one mesh."""
        return Mesh(vertices=self.vertices, faces=self.faces)

    def object_mesh(self, object_id: int) -> Mesh:
        """One object's slice of this cell, with its faces re-based to its own vertices.

        The stored indices are **cell-global** -- offset-corrected by the object's own vertex
        start rather than local to it -- so isolating an object means subtracting that start
        back off, which is what this does.
        """
        try:
            position = self.object_ids.index(int(object_id))
        except ValueError as error:
            raise KeyError(
                f"Object {object_id} is not in cell {self.cell} at level {self.level}; it holds {self.object_ids}."
            ) from error

        vertex_start = self.object_vertex_offsets[position]
        index_start = self.object_index_offsets[position]
        last = position + 1 == len(self.object_ids)
        vertex_stop = len(self.vertices) if last else self.object_vertex_offsets[position + 1]
        index_stop = len(self.faces) * 3 if last else self.object_index_offsets[position + 1]

        faces = self.faces[index_start // 3 : index_stop // 3] - vertex_start
        return Mesh(vertices=self.vertices[vertex_start:vertex_stop], faces=faces)


class Collection:
    """A collection opened from a store: its manifest, its catalogs and its geometry."""

    def __init__(self, store: MailleStore, prefix: str = "") -> None:
        """Open a collection, reading its manifest -- and refusing a prefix without one."""
        self.store = store
        self.prefix = prefix
        self.manifest = self._read_manifest()

    def __repr__(self) -> str:
        """Show the octree and how much is in it."""
        counts = self.manifest.counts or {}
        return (
            f"Collection(levels={self.manifest.grid.levels}, cell_size={self.manifest.grid.cell_size}, "
            f"objects={counts.get('objects', '?')}, codec={self.manifest.encoding.codec})"
        )

    # -- declarations ------------------------------------------------------ #

    @property
    def grid(self) -> Grid:
        """The octree this collection was cut with."""
        return self.manifest.grid

    @property
    def encoding(self) -> Encoding:
        """How this collection's blobs are packed."""
        return self.manifest.encoding

    def _read_manifest(self) -> Manifest:
        """Fetch and parse ``maille.json``, naming a missing one as an unfinished write."""
        path = join(self.prefix, MANIFEST_NAME)
        try:
            body = get_bytes(self.store, path)
        except FileNotFoundError as error:
            raise UnfinishedCollectionError(
                f"No `{MANIFEST_NAME}` at {path!r}, so this prefix is not a readable collection. A writer lands the "
                f"manifest last, so an interrupted run leaves exactly this."
            ) from error
        except Exception as error:  # a store may raise anything for a missing key
            raise UnfinishedCollectionError(
                f"Could not read `{MANIFEST_NAME}` at {path!r} ({error}). A writer lands the manifest last, so an "
                f"interrupted run leaves exactly this."
            ) from error
        return Manifest.from_json(body)

    # -- the catalogs ------------------------------------------------------ #

    @cached_property
    def cell_catalog(self) -> pa.Table:
        """The spatial index, read once."""
        return parquet_to_table(get_bytes(self.store, join(self.prefix, self._path("cells", CELL_CATALOG_PATH))))

    @cached_property
    def object_catalog(self) -> pa.Table:
        """The identity index, read once."""
        return parquet_to_table(get_bytes(self.store, join(self.prefix, self._path("objects", OBJECT_CATALOG_PATH))))

    def _path(self, role: str, fallback: str) -> str:
        """Where the manifest says a catalog is, or where the format puts it."""
        return self.manifest.catalog_file(role, fallback).path

    @cached_property
    def cells(self) -> dict[tuple[int, int], CellEntry]:
        """Every cell, keyed by ``(level, cell)``.

        This is the whole spatial index in memory -- one small row per cell, no blobs -- which
        is what the split into two catalogs exists to make affordable.
        """
        table = self.cell_catalog
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        entries: dict[tuple[int, int], CellEntry] = {}
        for row in range(table.num_rows):
            entry = CellEntry(
                level=_int(columns, "level", row),
                cell=_int(columns, "cell", row),
                vertex_count=_int(columns, "vertex_count", row),
                index_count=_int(columns, "index_count", row),
                bbox_min=(
                    _float(columns, "bbox_min_x", row),
                    _float(columns, "bbox_min_y", row),
                    _float(columns, "bbox_min_z", row),
                ),
                bbox_max=(
                    _float(columns, "bbox_max_x", row),
                    _float(columns, "bbox_max_y", row),
                    _float(columns, "bbox_max_z", row),
                ),
                lod_error=_float(columns, "lod_error", row),
                object_count=_int(columns, "object_count", row),
                child_mask=_int(columns, "child_mask", row),
                part=_optional_int(columns.get("part", []), row),
                row_group=_optional_int(columns.get("row_group", []), row),
                blob_bytes=_optional_int(columns.get("blob_bytes", []), row),
            )
            entries[(entry.level, entry.cell)] = entry
        return entries

    @cached_property
    def objects(self) -> dict[int, ObjectEntry]:
        """Every object, keyed by the id it was written with."""
        table = self.object_catalog
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        entries: dict[int, ObjectEntry] = {}
        for row in range(table.num_rows):
            entry = ObjectEntry(
                object_id=_int(columns, "object_id", row),
                ordinal=_int(columns, "ordinal", row),
                bbox_min=(
                    _float(columns, "bbox_min_x", row),
                    _float(columns, "bbox_min_y", row),
                    _float(columns, "bbox_min_z", row),
                ),
                bbox_max=(
                    _float(columns, "bbox_max_x", row),
                    _float(columns, "bbox_max_y", row),
                    _float(columns, "bbox_max_z", row),
                ),
                vertex_count=_int(columns, "vertex_count", row),
                index_count=_int(columns, "index_count", row),
                cells=tuple(
                    (int(item["level"]), int(item["cell"])) for item in _present(columns, "cells", row)
                ),
            )
            entries[entry.object_id] = entry
        return entries

    def cells_at(self, level: int) -> list[CellEntry]:
        """Every cell at one level, in Morton order."""
        return sorted(
            (entry for entry in self.cells.values() if entry.level == level),
            key=lambda entry: entry.cell,
        )

    def roots(self) -> list[CellEntry]:
        """The cells at the coarsest level -- where a descent starts."""
        return self.cells_at(self.grid.levels - 1)

    def cells_for_object(self, object_id: int, level: int | None = None) -> list[CellEntry]:
        """Which cells hold an object -- the lookup the object catalog exists for."""
        try:
            entry = self.objects[int(object_id)]
        except KeyError as error:
            raise KeyError(f"This collection has no object {object_id}.") from error
        return [
            self.cells[key]
            for key in entry.cells
            if key in self.cells and (level is None or key[0] == level)
        ]

    # -- the geometry ------------------------------------------------------ #

    @cached_property
    def _level_tables(self) -> dict[int, pa.Table]:
        """Whole levels held by :meth:`geometry`, which is a bulk-export path and says so."""
        return {}

    @cached_property
    def _parquet_files(self) -> dict[tuple[int, int], Any]:
        """Open ``ParquetFile`` handles per ``(level, part)``.

        The footer is parsed once per part and reused for every cell read out of it, which is
        why a row-group fetch is cheap after the first: the per-session cost is the footer, the
        per-cell cost is the row group.
        """
        return {}

    @cached_property
    def _opening(self) -> dict[tuple[int, int], asyncio.Lock]:
        """One lock per part, so concurrent readers open it once between them."""
        return {}

    @cached_property
    def _part_files(self) -> dict[tuple[int, int], StoreFile]:
        """The :class:`~maille.stores.StoreFile` under each open part, for the async path.

        Held separately because it is what a prefetch primes, and because a part opened from a
        whole-object read has none -- there is nothing left to fetch for it.
        """
        return {}

    def _entry(self, level: int, cell: int) -> CellEntry:
        """One cell's catalog row, or a message naming what was asked for."""
        entry = self.cells.get((int(level), int(cell)))
        if entry is None:
            raise KeyError(f"Level {level} holds no cell {cell}.")
        return entry

    def _part_entry(self, level: int, part: int) -> FileEntry:
        """The manifest's entry for one part of one level."""
        files = self.level_files(level)
        try:
            return files[part]
        except IndexError as error:
            raise FormatError(
                f"The cell catalog points at part {part} of level {level}, but the collection names "
                f"{len(files)} part(s) there."
            ) from error

    def level_files(self, level: int) -> list[FileEntry]:
        """The parts holding one level, from the manifest if it says, by listing if not."""
        declared = self.manifest.level_files(level)
        if declared is not None:
            return declared
        head = level_prefix(level)
        found = [path for path in list_paths(self.store, join(self.prefix, head)) if path.endswith(".parquet")]
        if not found:
            raise FormatError(f"This collection declares {self.grid.levels} levels but nothing is stored under {head!r}.")
        # A listing is absolute within the store; make it relative to the collection again.
        head_prefix = join(self.prefix, "")
        return [
            FileEntry(path=path[len(head_prefix) + 1 :] if head_prefix and path.startswith(head_prefix) else path)
            for path in found
        ]

    def level_paths(self, level: int) -> list[str]:
        """The paths of the parts holding one level."""
        return [entry.path for entry in self.level_files(level)]

    def _parquet_file(self, level: int, part: int) -> Any:  # noqa: ANN401
        """A ``ParquetFile`` over one part, reading through the store rather than into memory."""
        import pyarrow.parquet as pq

        key = (int(level), int(part))
        if key not in self._parquet_files:
            entry = self._part_entry(level, part)
            path = join(self.prefix, entry.path)
            if entry.size is None:
                # No recorded length means nothing can seek to the footer, so the part is read
                # whole. Correct, and the reason the writer bothers to record the length.
                self._parquet_files[key] = pq.ParquetFile(_buffer(get_bytes(self.store, path)))
            else:
                handle = StoreFile(self.store, path, entry.size)
                # A read-only seekable file is all `ParquetFile` uses; the stub asks for `IO`.
                self._parquet_files[key] = pq.ParquetFile(cast("IO[bytes]", handle))
                self._part_files[key] = handle
        return self._parquet_files[key]

    def geometry(self, level: int) -> pa.Table:
        """One level's geometry table, read and cached **whole**.

        The bulk path -- an export, a verification pass, a level small enough not to care. It
        is deliberately not what :meth:`read_cell` uses: fetching a level to draw a cell is the
        thing this format exists to avoid. Its cache is separate for the same reason, so a
        viewer that never calls this never accumulates one.
        """
        import pyarrow as pa

        if level not in self._level_tables:
            tables = [
                parquet_to_table(get_bytes(self.store, join(self.prefix, path)))
                for path in self.level_paths(level)
            ]
            self._level_tables[level] = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        return self._level_tables[level]

    def release(self) -> None:
        """Drop every cached table and open part.

        The catalogs stay -- they are small and a planner needs them every frame. What goes is
        the geometry, which is unbounded: a long-lived viewer that has panned across a large
        collection would otherwise hold every level it ever touched.
        """
        self._level_tables.clear()
        self._parquet_files.clear()
        self._part_files.clear()
        self._opening.clear()

    def _row_group(self, level: int, part: int, row_group: int) -> pa.Table:
        """One row group of one part, fetched as itself."""
        return self._parquet_file(level, part).read_row_group(int(row_group))

    def read_cell(self, level: int, cell: int) -> DecodedCell:
        """Fetch one cell and decode it into vertices and faces.

        Costs the row group holding the cell, plus the part's footer the first time that part
        is touched -- never the level.
        """
        entry = self._entry(level, cell)
        return self._decode(entry, self._table_for(entry))

    def _table_for(self, entry: CellEntry) -> pa.Table:
        """The smallest table that holds a cell: its row group, or its level if unlocated."""
        if entry.part is None or entry.row_group is None:
            return self.geometry(entry.level)
        return self._row_group(entry.level, entry.part, entry.row_group)

    def _decode(self, entry: CellEntry, table: pa.Table) -> DecodedCell:
        """Decode one cell out of a table that contains it."""
        cells = table.column("cell").to_pylist()
        try:
            row = cells.index(int(entry.cell))
        except ValueError as error:
            raise FormatError(
                f"Cell {entry.cell} of level {entry.level} is not in the row group the cell catalog names "
                f"(part {entry.part}, row group {entry.row_group}). The catalog and the geometry disagree; "
                f"`maille.verify` reports which."
            ) from error

        record = {name: table.column(name)[row].as_py() for name in table.column_names}
        # Both taken from the manifest rather than assumed: they are the two declarations a
        # decoder cannot re-derive from the bytes, which is why the format refuses to default
        # them. See `maille.manifest.Encoding`.
        codec = self.encoding.codec
        compression = self.encoding.compression
        vertices = decode_positions(
            record["positions"],
            cell=entry.cell,
            level=entry.level,
            cell_size=self.grid.cell_size,
            codec=codec,
            compression=compression,
            vertex_count=int(record["vertex_count"]),
        )
        faces = decode_indices(
            record["indices"],
            codec=codec,
            compression=compression,
            index_count=int(record["index_count"]),
        )
        return DecodedCell(
            level=entry.level,
            cell=entry.cell,
            vertices=vertices,
            faces=faces,
            object_ids=tuple(int(value) for value in record["object_ids"]),
            object_ordinals=tuple(int(value) for value in record["object_ordinals"]),
            object_vertex_offsets=tuple(int(value) for value in record["object_vertex_offsets"]),
            object_index_offsets=tuple(int(value) for value in record["object_index_offsets"]),
        )

    def read_cells(self, keys: Sequence[tuple[int, int]]) -> Iterator[DecodedCell]:
        """Decode a planner's selection, reading each row group once however many cells want it.

        A plan comes out in Morton order and the geometry is written in Morton order, so cells
        that are neighbours in a plan are usually neighbours in a row group. Grouping by row
        group before reading is what turns that adjacency into fewer fetches; yielding in the
        caller's order is what keeps it invisible.
        """
        entries = [self._entry(level, cell) for level, cell in keys]

        groups: dict[tuple[int, int | None, int | None], list[CellEntry]] = {}
        for entry in entries:
            groups.setdefault((entry.level, entry.part, entry.row_group), []).append(entry)

        decoded: dict[tuple[int, int], DecodedCell] = {}
        for members in groups.values():
            table = self._table_for(members[0])
            for entry in members:
                decoded[(entry.level, entry.cell)] = self._decode(entry, table)
        for entry in entries:
            yield decoded[(entry.level, entry.cell)]

    def object_mesh(self, object_id: int, level: int = 0) -> Mesh:
        """Reassemble one object at one level, across every cell that holds a piece of it.

        The end-to-end form of what the object catalog is for: a lookup, then one fetch per
        cell it names, never a scan.
        """
        from maille.geometry import concatenate_and_weld

        pieces = []
        for entry in sorted(self.cells_for_object(object_id, level=level), key=lambda item: item.cell):
            piece = self.read_cell(entry.level, entry.cell).object_mesh(object_id)
            if len(piece.faces):
                pieces.append((piece.vertices, piece.faces))
        if not pieces:
            raise KeyError(f"Object {object_id} has no geometry at level {level}.")
        vertices, faces = concatenate_and_weld(pieces)
        return Mesh(vertices=vertices, faces=faces)

    def cell_box(self, level: int, cell: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The grid box a cell addresses: its origin and its extent, in voxels."""
        return cell_box(int(cell), int(level), self.grid.cell_size)

    # -- reading without blocking the loop --------------------------------- #

    async def aread_cells(
        self, keys: Sequence[tuple[int, int]], *, concurrency: int = 16
    ) -> list[DecodedCell]:
        """Read a planner's selection with every fetch it needs in flight at once.

        A frame is forty cells, and forty sequential round trips to an object store is not a
        frame. The work is arranged in two waves, because the second cannot be known without
        the first: opening a part means reading its footer, and only the footer says where a
        row group's bytes are.

        1. Fetch the footer of every part the selection touches, concurrently.
        2. Fetch the byte span of every row group it needs, concurrently.
        3. Parse and decode -- against memory, in a worker thread.

        pyarrow's reader is synchronous and there is no honest way around that, so nothing here
        pretends otherwise. What is asynchronous is the part that is actually I/O.
        """
        entries = [self._entry(level, cell) for level, cell in keys]
        groups: dict[tuple[int, int | None, int | None], list[CellEntry]] = {}
        for entry in entries:
            groups.setdefault((entry.level, entry.part, entry.row_group), []).append(entry)

        gate = asyncio.Semaphore(max(1, int(concurrency)))

        async def guarded(coroutine: Any) -> Any:  # noqa: ANN401
            async with gate:
                return await coroutine

        # Only located row groups can be prefetched; an unlocated cell falls through to the
        # synchronous whole-level path inside `_table_for`, which is its own answer.
        located: list[tuple[int, int, int]] = [
            (level, part, group)
            for level, part, group in groups
            if part is not None and group is not None
        ]
        parts = {(level, part) for level, part, _ in located}
        await asyncio.gather(*(guarded(self._aopen_part(level, part)) for level, part in parts))
        await asyncio.gather(*(guarded(self._aprefetch(level, part, group)) for level, part, group in located))

        # The row groups are parsed here, on the loop's own thread, and only then is the decode
        # handed out. **That split is not a preference.** A `ParquetFile` and the `StoreFile`
        # under it hold a seek position, and neither is thread-safe: two threads reading row
        # groups out of one part do not race to a wrong answer, they segfault. Parsing is cheap
        # now anyway -- every byte it needs was prefetched above, so it touches memory and no
        # store at all -- while decoding is the meshopt and numpy work that a thread is for.
        tables = [self._table_for(members[0]) for members in groups.values()]

        def decode_group(members: list[CellEntry], table: pa.Table) -> list[DecodedCell]:
            return [self._decode(entry, table) for entry in members]

        results = await asyncio.gather(
            *(
                asyncio.to_thread(decode_group, members, table)
                for members, table in zip(groups.values(), tables)
            )
        )

        decoded: dict[tuple[int, int], DecodedCell] = {}
        for members, cells in zip(groups.values(), results):
            for entry, cell in zip(members, cells):
                decoded[(entry.level, entry.cell)] = cell
        return [decoded[(entry.level, entry.cell)] for entry in entries]

    async def aread_cell(self, level: int, cell: int) -> DecodedCell:
        """Read one cell without blocking the event loop."""
        return (await self.aread_cells([(level, cell)]))[0]

    async def _aopen_part(self, level: int, part: int) -> None:
        """Make sure a part's footer is in hand, fetching it without blocking.

        Guarded by a per-part lock because two overlapping ``aread_cells`` calls would
        otherwise both see the part as unopened, both fetch its footer, and both store a
        handle -- leaving ``_parquet_files`` reading through one file while ``_part_files``
        primes the other, so every prefetch after that lands in the wrong place and is
        silently wasted. Correct output, quietly no faster, which is the worst kind of bug to
        have in the thing whose whole purpose is speed.
        """
        import pyarrow.parquet as pq

        key = (int(level), int(part))
        if key in self._parquet_files:
            return
        async with self._opening.setdefault(key, asyncio.Lock()):
            if key in self._parquet_files:
                return
            entry = self._part_entry(level, part)
            path = join(self.prefix, entry.path)
            if entry.size is None:
                self._parquet_files[key] = pq.ParquetFile(_buffer(await aget_bytes(self.store, path)))
                return
            handle = StoreFile(self.store, path, entry.size)
            start, length = handle.tail_window()
            handle.prime_tail(await aget_range_bytes(self.store, path, start, length))
            self._parquet_files[key] = pq.ParquetFile(cast("IO[bytes]", handle))
            self._part_files[key] = handle

    async def _aprefetch(self, level: int, part: int, row_group: int) -> None:
        """Fetch one row group's byte span and hand it to the file the parse will read."""
        handle = self._part_files.get((int(level), int(part)))
        if handle is None:  # the part was opened from a whole-object read; nothing to prefetch
            return
        span = _row_group_span(self._parquet_files[(int(level), int(part))].metadata, int(row_group))
        if span is None:
            return
        start, length = span
        if handle.holds(start, length):
            # Already covered -- usually because the part is smaller than the footer window and
            # was pulled whole when it was opened. Re-fetching it here would undo that.
            return
        handle.prime(start, await aget_range_bytes(self.store, handle.path, start, length))

    # -- planning ---------------------------------------------------------- #

    def plan(self, **kwargs: Any) -> list[CellEntry]:  # noqa: ANN401
        """Choose which cells to draw at which level. See :func:`maille.plan_cells`."""
        from maille.planner import plan_cells

        return plan_cells(self, **kwargs)


def open_collection(store: MailleStore, prefix: str = "") -> Collection:
    """Open a collection written under ``prefix`` in ``store``.

    Reads one small object -- the manifest -- and nothing else until asked.
    """
    return Collection(store, prefix)


async def aopen_collection(store: MailleStore, prefix: str = "") -> Collection:
    """Open a collection without blocking the event loop.

    Returns the same :class:`Collection`: only the one small read that opening performs is
    moved off the loop, because everything after it is already lazy and the async read path
    hangs off the object itself.
    """
    collection = Collection.__new__(Collection)
    collection.store = store
    collection.prefix = prefix
    collection.manifest = Manifest.from_json(await _amanifest_bytes(store, prefix))
    return collection


async def _amanifest_bytes(store: MailleStore, prefix: str) -> bytes:
    """Fetch ``maille.json``, naming a missing one as the unfinished write it is."""
    path = join(prefix, MANIFEST_NAME)
    try:
        return await aget_bytes(store, path)
    except Exception as error:  # a store may raise anything for a missing key
        raise UnfinishedCollectionError(
            f"Could not read `{MANIFEST_NAME}` at {path!r} ({error}). A writer lands the manifest last, so an "
            f"interrupted run leaves exactly this."
        ) from error


__all__ = [
    "CellEntry",
    "Collection",
    "DecodedCell",
    "ObjectEntry",
    "aopen_collection",
    "open_collection",
]
