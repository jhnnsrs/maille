"""The layout of a collection's tree, and the manifest that describes it.

A meshed store is **self-describing**, which is the whole reason it is one store rather than a
handful. Everything a reader needs to turn a Morton code into a box and a blob into geometry
travels next to the geometry, in ``meshed.json`` at the root of the prefix::

    <prefix>/
      meshed.json                    <- the manifest, written LAST
      catalog/cells.parquet          <- the spatial index, one row per (level, cell)
      catalog/objects.parquet        <- the identity index, one row per object
      level=0/part-00000.parquet     <- the geometry, finest level
      level=1/part-00000.parquet
      level=2/part-00000.parquet

**The manifest is the completion marker.** A prefix has no atomic "upload finished" flag -- a
``PutObject`` either happened or it did not, but a tree is a sequence of writes that can stop
anywhere. So the manifest lands after every file it refers to, and a prefix without one is an
interrupted write rather than a collection. :func:`maille.write_collection` enforces the order
and the server refuses a prefix that lacks it, which is what turns "this collection is half
written" from something a renderer discovers into something registration rejects.

Two catalogs, because they answer opposite questions
----------------------------------------------------
``catalog/cells.parquet`` is the **spatial index**: one row per ``(level, cell)`` with the
exact bounds, the LOD error, the object count and the child mask. A renderer reads it once at
mount and from it alone decides which cells to fetch at which level, without opening a single
geometry file.

``catalog/objects.parquet`` is the **identity index, inverted**: one row per object with its
bounds, its dense ordinal and the cells that hold it. It answers *"where is segment 4711?"*
with a set of cell keys, which is what makes isolating or picking one object a lookup rather
than a scan. The forward direction already lives inside the geometry row as ``object_ids``.

They cannot be one file: their rows count different things, and a Parquet file has one schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from maille.errors import FormatError

#: The format version this writer emits and this reader accepts. It selects how every byte in
#: the prefix is read, so it is never defaulted on a reader's behalf and never guessed.
#:
#: 4 adds the two things a reader needs to fetch *one cell* rather than its whole level: the
#: ``part`` / ``row_group`` locator on every cell-catalog row, and a byte length beside every
#: file named in ``files``. The length is what lets a reader range-read a Parquet part without
#: being able to stat it -- a store is only asked for ``get``/``put``/``list``, and an HTTP one
#: can do neither ``head`` nor ``list``.
SPEC_VERSION = "4"

#: The manifest's name, at the root of the collection's prefix.
MANIFEST_NAME = "meshed.json"

#: The two catalogs, at paths the format fixes.
CELL_CATALOG_PATH = "catalog/cells.parquet"
OBJECT_CATALOG_PATH = "catalog/objects.parquet"


def level_prefix(level: int) -> str:
    """The directory holding one octree level's geometry."""
    return f"level={int(level)}"


def level_part_path(level: int, part: int = 0) -> str:
    """The path of one geometry part inside a level.

    A level is a *directory* rather than a file so a large level can be split across parts
    without the layout changing shape -- a reader globs the level and reads what it finds.
    """
    return f"{level_prefix(level)}/part-{int(part):05d}.parquet"


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #

#: Vertices are three ``uint16`` quantized against the cell's own grid box.
POSITIONS_UINT16_QUANTIZED_PER_CELL = "UINT16_QUANTIZED_PER_CELL"

#: Triangle indices, three per triangle.
INDICES_UINT32 = "UINT32"
INDICES_UINT16 = "UINT16"

#: The blob codec. ``MESHOPT`` is what glTF's ``EXT_meshopt_compression`` uses, so a web
#: renderer already ships the decoder.
CODEC_MESHOPT = "MESHOPT"
CODEC_NONE = "NONE"

COMPRESSION_NONE = "NONE"
COMPRESSION_ZSTD = "ZSTD"

#: Vertices on a cell face plane did not move during decimation, so two neighbouring cells
#: drawn at different levels meet without a crack.
BOUNDARY_LOCKED = "LOCKED"
BOUNDARY_OPEN = "OPEN"

#: Level ``L`` targets ``ratio**L`` of the level-0 face count. ``QUARTER`` is the default and
#: the only value the original spec defined; the rest are maille naming what it actually did,
#: because declaring ``QUARTER`` over a different ratio would be a claim no check could catch.
DECIMATION_QUARTER = "QUARTER"
DECIMATION_HALF = "HALF"
DECIMATION_EIGHTH = "EIGHTH"
DECIMATION_CUSTOM = "CUSTOM"

SORT_KEY_MORTON = "MORTON"

_ENCODING_VOCABULARY: dict[str, frozenset[str]] = {
    "positions": frozenset({POSITIONS_UINT16_QUANTIZED_PER_CELL}),
    "indices": frozenset({INDICES_UINT32, INDICES_UINT16}),
    "codec": frozenset({CODEC_MESHOPT, CODEC_NONE}),
    "compression": frozenset({COMPRESSION_NONE, COMPRESSION_ZSTD}),
    "boundary": frozenset({BOUNDARY_LOCKED, BOUNDARY_OPEN}),
    "decimation": frozenset({DECIMATION_QUARTER, DECIMATION_HALF, DECIMATION_EIGHTH, DECIMATION_CUSTOM}),
}

#: The keys a reader cannot work without, and which are therefore never defaulted. ``codec``
#: and ``compression`` are the load-bearing pair: guessing them does not produce an error, it
#: produces geometry that decodes to garbage.
_REQUIRED_ENCODING_KEYS = ("positions", "indices", "codec", "compression", "boundary", "decimation")


@dataclass(frozen=True)
class FileEntry:
    """One file the manifest names, with what a reader needs to range-read it.

    ``size`` is here because **nothing else in the tree can tell a reader how long a file is**.
    maille asks a store for ``put``/``get``/``list`` and nothing more, so there is no ``head``
    to call, and an HTTP-backed store usually cannot list either. Yet the length is the first
    thing a Parquet reader needs: the footer lives at the end, so a reader that cannot seek to
    the end cannot parse the file at all without downloading it whole -- which is the exact
    thing the locator exists to avoid.

    The writer knows it for free (it had just serialized the bytes), so it records it. When it
    is absent -- a hand-written manifest -- a reader falls back to fetching the part whole,
    which is correct and merely slow.
    """

    path: str
    size: int | None = None
    row_groups: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The entry as it is written, omitting what was never measured."""
        written: dict[str, Any] = {"path": self.path}
        if self.size is not None:
            written["bytes"] = int(self.size)
        if self.row_groups is not None:
            written["rowGroups"] = int(self.row_groups)
        return written

    @classmethod
    def from_any(cls, raw: Any) -> FileEntry:  # noqa: ANN401
        """Read an entry, accepting a bare path string for a hand-written manifest."""
        if isinstance(raw, str):
            return cls(path=raw)
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            size = raw.get("bytes")
            groups = raw.get("rowGroups")
            return cls(
                path=str(raw["path"]),
                size=None if size is None else int(size),
                row_groups=None if groups is None else int(groups),
            )
        raise FormatError(
            f"A file entry in a manifest is a path, or an object carrying one, got {raw!r}."
        )


@dataclass(frozen=True)
class Grid:
    """The octree: how big a level-0 cell is, how many levels there are, how cells are keyed.

    ``cell_size`` is **one size per component, in the same order as the vertices** -- the order
    the ``bbox_*_x/y/z`` columns also use, where ``x``/``y``/``z`` are labels for slots 0, 1
    and 2 rather than claims about physical axes. Nothing here reads it as a physical axis, so
    a collection built from ``(z, y, x)`` data states its cell size ``(z, y, x)`` too and is
    entirely consistent. What must not differ is the order *between* the two: ``(128, 128, 64)``
    against vertices whose third component is the tall one describes a different octree, which
    is legal and simply partitions the data less well.

    Units are voxels of the collection's own coordinate system, which is what lets the octree
    align to the grid the meshes were extracted from.
    """

    cell_size: tuple[int, int, int]
    levels: int
    sort_key: str = SORT_KEY_MORTON

    def __post_init__(self) -> None:
        """Refuse a grid that cannot address geometry."""
        if len(self.cell_size) != 3 or any(int(component) < 1 for component in self.cell_size):
            raise FormatError(
                f"`cell_size` is three whole numbers of at least 1 voxel, one per component in the same order as the vertices, got {self.cell_size!r}."
            )
        if self.levels < 1:
            raise FormatError(f"An octree has at least one level, got {self.levels}.")
        if self.sort_key != SORT_KEY_MORTON:
            raise FormatError(f"`sort_key` is {self.sort_key!r}; the format defines {SORT_KEY_MORTON}.")

    def to_dict(self) -> dict[str, Any]:
        """The manifest's ``grid`` object."""
        return {
            "cellSize": [int(component) for component in self.cell_size],
            "levels": int(self.levels),
            "sortKey": self.sort_key,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Grid:
        """Read a manifest's ``grid`` object."""
        try:
            cell_size = tuple(int(component) for component in raw["cellSize"])
        except (KeyError, TypeError, ValueError) as error:
            raise FormatError(f"A manifest's `grid` needs a three-component `cellSize`, got {raw!r}.") from error
        if len(cell_size) != 3:
            raise FormatError(f"A mesh grid is three-dimensional, so `cellSize` takes 3 values, got {raw['cellSize']!r}.")
        return cls(
            cell_size=cell_size,  # type: ignore[arg-type]
            levels=int(raw.get("levels", 0)),
            sort_key=str(raw.get("sortKey", SORT_KEY_MORTON)),
        )

    def cell_extent(self, level: int) -> tuple[float, float, float]:
        """How many voxels a cell spans per axis at ``level``."""
        scale = 2**int(level)
        return tuple(float(component) * scale for component in self.cell_size)  # type: ignore[return-value]


#: The ratios that have a name of their own. Anything else is ``CUSTOM``.
_NAMED_RATIOS: dict[float, str] = {
    0.25: DECIMATION_QUARTER,
    0.5: DECIMATION_HALF,
    0.125: DECIMATION_EIGHTH,
}


@dataclass(frozen=True)
class Decimation:
    """How much of a level survives into the next one, and what to call that in the manifest.

    Level ``L`` targets ``ratio**L`` of the level-0 face count, with ``floor_faces`` as the
    smallest budget any one object's piece is given -- below a few triangles there is nothing
    left to take, and taking it anyway removes the object from the level.

    ``declaration`` is what lands in ``encoding.decimation``, and it is **checked against the
    ratio rather than trusted**: a writer that declared ``QUARTER`` while asking for half would
    be making a claim nothing downstream could test, so the two are required to agree. Use the
    constructors and this never comes up.

    **It names the target, not the outcome.** A surface can refuse a budget -- a simplifier that
    preserves topology will stop short rather than destroy it -- so a collection declaring
    ``QUARTER`` is one that *asked* each level for a quarter, not one that always got it. That
    is the reading the format's own wording takes ("level ``L`` targets ``(1/4)**L``"), and it
    is why what actually happened is reported by a warning at build time and readable per cell
    in the catalog, rather than being folded into a single word here.
    """

    ratio: float = 0.25
    floor_faces: int = 4
    declaration: str = DECIMATION_QUARTER

    def __post_init__(self) -> None:
        """Refuse a schedule that cannot coarsen, or a name that misdescribes one."""
        if not 0.0 < self.ratio < 1.0:
            raise FormatError(
                f"`ratio` is the fraction of faces a level keeps of the one below, so it lies between 0 and 1, "
                f"got {self.ratio}."
            )
        if self.floor_faces < 4:
            raise FormatError(f"`floor_faces` is at least one tetrahedron's worth, 4, got {self.floor_faces}.")
        if self.declaration not in _ENCODING_VOCABULARY["decimation"]:
            raise FormatError(
                f"`declaration` is {self.declaration!r}; the format defines "
                f"{', '.join(sorted(_ENCODING_VOCABULARY['decimation']))}."
            )
        expected = _NAMED_RATIOS.get(round(self.ratio, 10), DECIMATION_CUSTOM)
        if self.declaration != expected:
            raise FormatError(
                f"A ratio of {self.ratio} is {expected}, not {self.declaration}. The declaration travels with the "
                f"geometry and nothing downstream can re-derive it, so it is required to describe what was actually "
                f"done -- use Decimation.custom({self.ratio}) if that is the reduction you want."
            )

    @classmethod
    def quarter(cls, floor_faces: int = 4) -> Decimation:
        """A quarter of the faces per level: the format's default, and the spec's only name."""
        return cls(ratio=0.25, floor_faces=floor_faces, declaration=DECIMATION_QUARTER)

    @classmethod
    def half(cls, floor_faces: int = 4) -> Decimation:
        """Half the faces per level: gentler, so more levels are needed to reach a given budget."""
        return cls(ratio=0.5, floor_faces=floor_faces, declaration=DECIMATION_HALF)

    @classmethod
    def eighth(cls, floor_faces: int = 4) -> Decimation:
        """An eighth per level: aggressive, and matches an octree's cell count per level."""
        return cls(ratio=0.125, floor_faces=floor_faces, declaration=DECIMATION_EIGHTH)

    @classmethod
    def custom(cls, ratio: float, floor_faces: int = 4) -> Decimation:
        """Any other ratio, declared as ``CUSTOM`` so the manifest does not misdescribe it."""
        named = _NAMED_RATIOS.get(round(ratio, 10), DECIMATION_CUSTOM)
        return cls(ratio=ratio, floor_faces=floor_faces, declaration=named)

    def target_faces(self, faces: int, level: int) -> int:
        """The face budget for a piece of ``faces`` triangles at ``level``."""
        return max(self.floor_faces, round(faces * (self.ratio**level)))


@dataclass(frozen=True)
class Encoding:
    """How the blobs are packed. Every value is a claim a decoder acts on.

    Nothing here is defaulted from maille's side beyond the two keys that have exactly one
    legal value under this format. ``codec`` and ``compression`` in particular are always
    stated by the writer: a wrong one is not an error anywhere, it is geometry that decodes to
    garbage.
    """

    positions: str = POSITIONS_UINT16_QUANTIZED_PER_CELL
    indices: str = INDICES_UINT32
    codec: str = CODEC_MESHOPT
    compression: str = COMPRESSION_NONE
    boundary: str = BOUNDARY_LOCKED
    decimation: str = DECIMATION_QUARTER

    def __post_init__(self) -> None:
        """Refuse a value outside the format's vocabulary."""
        for key in _REQUIRED_ENCODING_KEYS:
            value = getattr(self, key)
            allowed = _ENCODING_VOCABULARY[key]
            if value not in allowed:
                raise FormatError(
                    f"`encoding.{key}` is {value!r}; the format defines {', '.join(sorted(allowed))}."
                )

    def to_dict(self) -> dict[str, str]:
        """The manifest's ``encoding`` object, always complete.

        Never sparse: a renderer configures its decoder from what it reads back, so a manifest
        that omits a key it resolved internally hands every reader an encoding that says
        nothing.
        """
        return {key: getattr(self, key) for key in _REQUIRED_ENCODING_KEYS}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Encoding:
        """Read a manifest's ``encoding`` object, refusing one that leaves a decoder guessing."""
        missing = [key for key in _REQUIRED_ENCODING_KEYS if key not in raw]
        if missing:
            raise FormatError(
                f"This manifest's `encoding` omits {', '.join(missing)}. A decoder cannot infer them -- a wrong "
                f"guess is not an error, it is geometry that decodes to garbage -- so the collection is refused."
            )
        return cls(**{key: str(raw[key]) for key in _REQUIRED_ENCODING_KEYS})


@dataclass(frozen=True)
class Manifest:
    """``meshed.json``: what a reader learns before opening a single Parquet file."""

    grid: Grid
    encoding: Encoding
    #: Optional, and carried rather than used -- see :func:`validate_axes`. Omitted from the
    #: written manifest when unset, because a key that is absent says "this layer did not
    #: claim an axis order" where a guessed one says something false.
    axes: tuple[str, ...] | None = None
    spec_version: str = SPEC_VERSION
    counts: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The manifest as it is written, with the resolved declarations rather than the input."""
        written: dict[str, Any] = {
            "specVersion": self.spec_version,
            "grid": self.grid.to_dict(),
            "encoding": self.encoding.to_dict(),
        }
        if self.axes is not None:
            written["axes"] = list(self.axes)
        written["counts"] = dict(self.counts)
        written["files"] = dict(self.files)
        return written

    def to_json(self) -> bytes:
        """The manifest's bytes, as they land at the root of the prefix."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False).encode("utf-8")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Manifest:
        """Read a manifest, refusing one this reader cannot act on."""
        version = str(raw.get("specVersion", "")).strip()
        if version != SPEC_VERSION:
            raise FormatError(
                f"This manifest declares specVersion {version!r}, which maille cannot read. Supported: {SPEC_VERSION}. "
                f"The version selects how every byte in the prefix is read, so an unknown one is refused rather than "
                f"read as though it were familiar."
            )
        grid, encoding = raw.get("grid"), raw.get("encoding")
        if not isinstance(grid, Mapping) or not isinstance(encoding, Mapping):
            raise FormatError(
                "A manifest must carry a `grid` and an `encoding` object: they are how a reader turns a Morton code "
                "into a box and the blobs into geometry, and nothing else in the store states them."
            )
        # Absent is a legitimate manifest: nothing in the format is decoded through `axes`, so a
        # collection that never claimed an axis order is readable in full. Present but malformed
        # is not -- that is a layer above stating something it got wrong.
        axes = raw.get("axes")
        if axes is not None:
            axes = validate_axes(axes)
        return cls(
            grid=Grid.from_dict(grid),
            encoding=Encoding.from_dict(encoding),
            axes=axes,
            spec_version=version,
            counts=dict(raw.get("counts") or {}),
            files=dict(raw.get("files") or {}),
        )

    def catalog_file(self, role: str, fallback: str) -> FileEntry:
        """Where the manifest says a catalog is, or where the format puts it."""
        declared = (self.files or {}).get(role)
        return FileEntry(path=fallback) if declared is None else FileEntry.from_any(declared)

    def level_files(self, level: int) -> list[FileEntry] | None:
        """The parts holding one level, or ``None`` if the manifest does not say.

        ``None`` rather than an empty list, because "this manifest names no parts" and "this
        level has no parts" are different answers: the first sends a reader to the store's
        listing, the second would send it nowhere.
        """
        declared = (self.files or {}).get("levels")
        if not isinstance(declared, Mapping):
            return None
        entries = declared.get(str(int(level)))
        if not isinstance(entries, Sequence) or isinstance(entries, str) or not entries:
            return None
        return [FileEntry.from_any(entry) for entry in entries]

    @classmethod
    def from_json(cls, body: bytes) -> Manifest:
        """Parse a manifest's bytes, naming a truncated one for what it is."""
        try:
            raw = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormatError(
                f"This manifest is not valid JSON ({error}). It is {len(body)} bytes and starts {body[:80]!r} -- a "
                f"truncated body is the other shape an interrupted write leaves behind."
            ) from error
        if not isinstance(raw, Mapping):
            raise FormatError(f"A manifest is a JSON object, got {type(raw).__name__}.")
        return cls.from_dict(raw)


def validate_axes(axes: Sequence[str]) -> tuple[str, ...]:
    """Check an axis order a caller chose to declare. maille never invents one.

    **``axes`` is optional, and nothing here reads it.** Everything maille computes is
    positional ``(x, y, z)`` and self-consistent without any names at all: vertex components,
    ``cell_size``, the ``bbox_*_x/y/z`` columns and the Morton interleave. A collection with no
    declared axes decodes identically to one that declares them.

    It exists because *naming* those axes is a question about the collection's relationship to
    something else -- the image it was extracted from, the coordinate graph it is placed in --
    and that relationship lives a layer up, in whatever owns the coordinate system. Only that
    layer can say whether the order matches its source, and where it does not the honest place
    to record it is the derivation edge between the two spaces (a ``MAP_AXIS`` naming each axis
    on both sides), never a reordered ``cell_size``.

    So a caller who knows the answer passes it and it is carried through to the manifest
    unchanged; a caller who does not, omits it. Requiring it would be worse than either: a
    parameter that cannot be reasoned about from the geometry in hand gets filled with a
    plausible guess, which is exactly the wrong declaration this would be trying to prevent.
    """
    if isinstance(axes, str) or len(axes) != 3:
        raise FormatError(f"A mesh collection is three-dimensional, so `axes` names 3 axes, got {axes!r}.")
    names = tuple(str(axis) for axis in axes)
    if len(set(names)) != 3:
        raise FormatError(f"`axes` names each axis once, got {names!r}.")
    return names


__all__ = [
    "BOUNDARY_LOCKED",
    "BOUNDARY_OPEN",
    "CELL_CATALOG_PATH",
    "CODEC_MESHOPT",
    "CODEC_NONE",
    "COMPRESSION_NONE",
    "COMPRESSION_ZSTD",
    "DECIMATION_CUSTOM",
    "DECIMATION_EIGHTH",
    "DECIMATION_HALF",
    "DECIMATION_QUARTER",
    "INDICES_UINT16",
    "INDICES_UINT32",
    "MANIFEST_NAME",
    "OBJECT_CATALOG_PATH",
    "POSITIONS_UINT16_QUANTIZED_PER_CELL",
    "SORT_KEY_MORTON",
    "SPEC_VERSION",
    "Encoding",
    "FileEntry",
    "Grid",
    "Manifest",
    "level_part_path",
    "level_prefix",
    "validate_axes",
]
