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
through :class:`maille.store.StoreFile` -- a seekable view over the store -- so pyarrow reads
the footer once and then only the column chunks of the one row group asked for. That is what
makes the format's claim true in the reader as well as on paper: drawing forty cells costs
forty row groups, not the levels they came from.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from maille.codec import cell_box, decode_indices, decode_positions, morton_decode
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
from maille.sources import Mesh
from maille.store import MailleStore, StoreFile, get_bytes, join, list_paths

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa


def _optional_int(column: Sequence[Any], row: int) -> int | None:
    """One nullable integer out of a column that a manifest may not have written at all."""
    if row >= len(column):
        return None
    value = column[row]
    return None if value is None else int(value)


def _buffer(body: bytes) -> Any:  # noqa: ANN401
    """An in-memory random-access file, for a part whose length nothing recorded."""
    import pyarrow as pa

    return pa.BufferReader(body)


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
        from maille.codec import morton_encode_one

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
    vertices: np.ndarray
    faces: np.ndarray
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

    @property
    def axes(self) -> tuple[str, ...] | None:
        """The axis order the writer declared, or ``None`` if it declared none.

        Optional by design: nothing in the format decodes through it, so a collection without
        it is complete. See :func:`maille.manifest.validate_axes`.
        """
        return self.manifest.axes

    def _read_manifest(self) -> Manifest:
        """Fetch and parse ``meshed.json``, naming a missing one as an unfinished write."""
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
                level=int(columns["level"][row]),
                cell=int(columns["cell"][row]),
                vertex_count=int(columns["vertex_count"][row]),
                index_count=int(columns["index_count"][row]),
                bbox_min=(
                    float(columns["bbox_min_x"][row]),
                    float(columns["bbox_min_y"][row]),
                    float(columns["bbox_min_z"][row]),
                ),
                bbox_max=(
                    float(columns["bbox_max_x"][row]),
                    float(columns["bbox_max_y"][row]),
                    float(columns["bbox_max_z"][row]),
                ),
                lod_error=float(columns["lod_error"][row]),
                object_count=int(columns["object_count"][row]),
                child_mask=int(columns["child_mask"][row]),
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
                object_id=int(columns["object_id"][row]),
                ordinal=int(columns["ordinal"][row]),
                bbox_min=(
                    float(columns["bbox_min_x"][row]),
                    float(columns["bbox_min_y"][row]),
                    float(columns["bbox_min_z"][row]),
                ),
                bbox_max=(
                    float(columns["bbox_max_x"][row]),
                    float(columns["bbox_max_y"][row]),
                    float(columns["bbox_max_z"][row]),
                ),
                vertex_count=int(columns["vertex_count"][row]),
                index_count=int(columns["index_count"][row]),
                cells=tuple((int(item["level"]), int(item["cell"])) for item in columns["cells"][row]),
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
            files = self.level_files(level)
            try:
                entry = files[part]
            except IndexError as error:
                raise FormatError(
                    f"The cell catalog points at part {part} of level {level}, but the collection names "
                    f"{len(files)} part(s) there."
                ) from error
            path = join(self.prefix, entry.path)
            if entry.size is None:
                # No recorded length means nothing can seek to the footer, so the part is read
                # whole. Correct, and the reason the writer bothers to record the length.
                self._parquet_files[key] = pq.ParquetFile(_buffer(get_bytes(self.store, path)))
            else:
                self._parquet_files[key] = pq.ParquetFile(StoreFile(self.store, path, entry.size))
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

    def _row_group(self, level: int, part: int, row_group: int) -> pa.Table:
        """One row group of one part, fetched as itself."""
        return self._parquet_file(level, part).read_row_group(int(row_group))

    def read_cell(self, level: int, cell: int) -> DecodedCell:
        """Fetch one cell and decode it into vertices and faces.

        Costs the row group holding the cell, plus the part's footer the first time that part
        is touched -- never the level.
        """
        entry = self.cells.get((int(level), int(cell)))
        if entry is None:
            raise KeyError(f"Level {level} holds no cell {cell}.")
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
        codec = self.encoding.codec
        vertices = decode_positions(
            record["positions"],
            cell=entry.cell,
            level=entry.level,
            cell_size=self.grid.cell_size,
            codec=codec,
            vertex_count=int(record["vertex_count"]),
        )
        faces = decode_indices(record["indices"], codec=codec, index_count=int(record["index_count"]))
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
        entries: list[CellEntry] = []
        for level, cell in keys:
            entry = self.cells.get((int(level), int(cell)))
            if entry is None:
                raise KeyError(f"Level {level} holds no cell {cell}.")
            entries.append(entry)

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

    def cell_box(self, level: int, cell: int) -> tuple[np.ndarray, np.ndarray]:
        """The grid box a cell addresses: its origin and its extent, in voxels."""
        return cell_box(int(cell), int(level), self.grid.cell_size)

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


__all__ = ["CellEntry", "Collection", "DecodedCell", "ObjectEntry", "open_collection"]
