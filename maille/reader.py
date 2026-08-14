"""Opening a collection back up: the manifest, the two catalogs, and a decoded cell.

A store that describes itself is only half a format if nothing can open it, so this is the
other half. It is also what makes the round-trip check real: the encoder is proven by decoding
its own output, not by asserting on the bytes it happened to produce.

Reading is lazy and in that order on purpose. The manifest is one small object and tells you
whether the collection is readable at all; the cell catalog is the next smallest and answers
*which cells, at which level* without opening a single geometry file; only then does anything
fetch a blob.
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
    Grid,
    Manifest,
    level_prefix,
)
from maille.sources import Mesh
from maille.store import MailleStore, get_bytes, join, list_paths

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa


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
        declared = (self.manifest.files or {}).get(role)
        return str(declared) if isinstance(declared, str) else fallback

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
        """Lazily-filled cache of the geometry table per level."""
        return {}

    def level_paths(self, level: int) -> list[str]:
        """The parts holding one level, from the manifest if it says, by listing if not."""
        declared = (self.manifest.files or {}).get("levels")
        if isinstance(declared, dict):
            paths = declared.get(str(level))
            if isinstance(paths, list) and paths:
                return [str(path) for path in paths]
        head = level_prefix(level)
        found = [path for path in list_paths(self.store, join(self.prefix, head)) if path.endswith(".parquet")]
        if not found:
            raise FormatError(f"This collection declares {self.grid.levels} levels but nothing is stored under {head!r}.")
        # A listing is absolute within the store; make it relative to the collection again.
        head_prefix = join(self.prefix, "")
        return [path[len(head_prefix) + 1 :] if head_prefix and path.startswith(head_prefix) else path for path in found]

    def geometry(self, level: int) -> pa.Table:
        """One level's geometry table, read and cached whole.

        Reading a level whole is right for a level that fits in memory and wrong for one that
        does not; a collection sized past that wants a Parquet reader that pushes the cell
        predicate down, which is a reader concern rather than a format one.
        """
        import pyarrow as pa

        if level not in self._level_tables:
            tables = [
                parquet_to_table(get_bytes(self.store, join(self.prefix, path)))
                for path in self.level_paths(level)
            ]
            self._level_tables[level] = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        return self._level_tables[level]

    def read_cell(self, level: int, cell: int) -> DecodedCell:
        """Fetch one cell and decode it into vertices and faces."""
        table = self.geometry(level)
        cells = table.column("cell").to_pylist()
        try:
            row = cells.index(int(cell))
        except ValueError as error:
            raise KeyError(f"Level {level} holds no cell {cell}.") from error

        record = {name: table.column(name)[row].as_py() for name in table.column_names}
        vertex_count = int(record["vertex_count"])
        index_count = int(record["index_count"])
        codec = self.encoding.codec

        vertices = decode_positions(
            record["positions"],
            cell=int(cell),
            level=int(level),
            cell_size=self.grid.cell_size,
            codec=codec,
            vertex_count=vertex_count,
        )
        faces = decode_indices(record["indices"], codec=codec, index_count=index_count)
        return DecodedCell(
            level=int(level),
            cell=int(cell),
            vertices=vertices,
            faces=faces,
            object_ids=tuple(int(value) for value in record["object_ids"]),
            object_ordinals=tuple(int(value) for value in record["object_ordinals"]),
            object_vertex_offsets=tuple(int(value) for value in record["object_vertex_offsets"]),
            object_index_offsets=tuple(int(value) for value in record["object_index_offsets"]),
        )

    def read_cells(self, keys: Sequence[tuple[int, int]]) -> Iterator[DecodedCell]:
        """Decode a planner's selection, one cell at a time."""
        for level, cell in keys:
            yield self.read_cell(level, cell)

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
