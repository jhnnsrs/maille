"""Turning objects into the three frames of a collection.

What this builder does not do
-----------------------------
It holds every object in memory and builds one shard per level, so it is sized for thousands
of objects rather than millions. It computes ``lod_error`` as an upper bound -- the
quantization step plus the largest distance any vertex moved during decimation -- rather than
a measured Hausdorff distance.

What it does do, and what depends on it
---------------------------------------
Rows within a level are emitted in ascending Morton order, which is what ``grid.sortKey``
declares. That is not a cosmetic ordering: the writer groups consecutive rows into row groups,
so Morton order is what makes a row group a *spatially compact* set of cells rather than an
arbitrary one, and therefore what makes fetching one row group a sensible unit of work for a
reader. Reordering these rows would not corrupt anything -- it would quietly make every range
read fetch scattered geometry.

The one thing left unfilled here is the cell catalog's ``part`` / ``row_group`` / ``blob_bytes``
locator. Those describe the serialized bytes, which do not exist yet at build time; a
:class:`MeshCollection` you inspect without writing carries nulls there, and
:func:`maille.write_collection` fills them in as it lands each part.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from maille.codecs import QUANT_MAX, encode_indices, encode_positions
from maille.errors import FormatError
from maille.frames import arrow_schemas, build_table
from maille.geometry import (
    clip_to_cells,
    concatenate_and_weld,
    drop_degenerate,
    on_planes,
    snap_boundary,
)
from maille.manifest import (
    CELL_CATALOG_PATH,
    CODEC_MESHOPT,
    CODEC_NONE,
    COMPRESSION_NONE,
    MAX_ORDINAL,
    OBJECT_CATALOG_PATH,
    Decimation,
    Encoding,
    Grid,
    Manifest,
    level_part_path,
)
from maille.octree import cell_box, morton_decode, morton_encode_one
from maille.simplifiers import Simplifier, resolve_simplifier, simplify_to_target
from maille.sources import MeshSource, coerce_objects
from maille.stores import MailleStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

#: Enough cells for a frustum query to narrow a fetch, few enough that the catalog stays small.
_TARGET_LEVEL0_CELLS = 4096


@dataclass
class MeshCollection:
    """A collection of meshes"""

    cell_catalog: pa.Table
    """ The spatial index, one row per cell, with the columns the format defines. """
    object_catalog: pa.Table
    """ The identity index, one row per object, with the columns the format defines. """
    shards: list[tuple[int, pa.Table]]
    """ The geometry, one table per level, with the columns the format defines. """
    manifest: Manifest
    """ The manifest, which describes the collection and its files. """

    @property
    def grid(self) -> Grid:
        """The octree this collection was cut with."""
        return self.manifest.grid

    @property
    def encoding(self) -> Encoding:
        """How this collection's blobs are packed."""
        return self.manifest.encoding

    def write(self, store: MailleStore, prefix: str = "") -> Manifest:
        """Write this collection into a store. See :func:`maille.write_collection`."""
        from maille.writer import write_collection

        return write_collection(self, store, prefix)


def choose_cell_size(objects: Mapping[int, Any], *, levels: int = 3) -> tuple[int, int, int]:
    """Pick a level-0 cell, in voxels, that most objects fit inside whole.

    **Prefer passing the source array's chunk shape when you know it.** ``cell_size`` is in
    voxels precisely so the octree can align to the grid the meshes were extracted from, and a
    cell that matches the array's chunking means a viewer fetching image chunks and mesh cells
    pulls the same regions. Nothing about the meshes themselves can tell you that, so this
    function cannot find it -- it only avoids the two choices that go wrong silently.

    Too small is the harmful direction: an object wider than a cell is cut, its cut vertices
    are pinned to the cell faces by ``boundary: LOCKED``, and a decimator that may not move them
    cannot reach the ``QUARTER`` budget -- a coarse level then costs a file and saves nothing.
    So the size is driven off the *objects*, at twice the 90th-percentile object extent per
    axis, rounded up to a power of two.

    Too large wastes the structure rather than corrupting it: one cell holding everything is a
    correct collection whose octree never narrows a fetch. So the result is halved while level 0
    would hold fewer cells than 4096, down to the point where objects would start being cut.

    Powers of two because a coarser cell is exactly two finer ones, so the level-0 planes stay a
    superset of every coarser level's -- which is what the boundary argument rests on.
    """
    del levels  # part of the signature so a caller can pass it; the heuristic is level-free
    if not objects:
        raise FormatError(
            "Cannot choose a cell size for an empty collection; pass `cell_size` explicitly."
        )

    bounds = [mesh.bounds for mesh in objects.values()]
    spans = np.array([np.asarray(high) - np.asarray(low) for low, high in bounds])
    lower = np.min([np.asarray(low) for low, _ in bounds], axis=0)
    upper = np.max([np.asarray(high) for _, high in bounds], axis=0)

    # Twice the 90th percentile: a cell that merely equals the typical object still cuts about
    # half of them, since an object has to be *placed* inside a cell, not just fit in one.
    wanted = 2.0 * np.percentile(spans, 90, axis=0)
    floor = np.maximum(wanted, 8.0)  # a cell below a few voxels addresses noise
    chosen = np.maximum(np.exp2(np.ceil(np.log2(floor))), 1.0)

    # Do not exceed the data: a cell larger than everything is the degenerate one-cell octree.
    extent = np.maximum(upper, 1.0)
    chosen = np.minimum(chosen, np.exp2(np.ceil(np.log2(np.maximum(extent, 1.0)))))

    def level0_cells(size: npt.NDArray[np.float64]) -> float:
        return float(np.prod(np.maximum(np.ceil((upper - np.minimum(lower, 0.0)) / size), 1.0)))

    # Halve while the octree is doing too little work, but never below the object-fit size --
    # that is the constraint whose violation is silent.
    while (
        level0_cells(chosen) < _TARGET_LEVEL0_CELLS
        and (chosen / 2 >= wanted).all()
        and (chosen > 8).all()
    ):
        chosen = chosen / 2

    return tuple(int(component) for component in chosen)  # type: ignore[return-value]


def build_collection(
    objects: Mapping[int, MeshSource],
    *,
    cell_size: Sequence[int] | None = None,
    levels: int = 3,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    simplifier: Simplifier | str | None = None,
    decimation: Decimation | None = None,
) -> MeshCollection:
    """Turn ``{object_id: mesh}`` into a collection's three frames and its manifest.

    ``objects`` is keyed by the id the object carries in whatever it was extracted from -- a
    label volume's instance id, say -- and those ids are written to ``object_ids`` as given.
    Each value is a ``trimesh.Trimesh``, a :class:`maille.Mesh`, or a ``(vertices, faces)``
    pair. Vertices are in voxels, ordered ``(x, y, z)``.

    **Components are positional throughout, and never named.** Vertex components, ``cell_size``
    and the ``bbox_*`` columns are slots 0, 1 and 2, and nothing here asks which physical axis a
    slot holds -- meshes off a ``(z, y, x)`` volume stay ``(z, y, x)``. What those slots *mean*
    is a statement about how this collection relates to whatever it came from, which belongs to
    the layer that owns the coordinate system and is recorded there, not here.

    ``cell_size`` is the level-0 cell in voxels, ``(x, y, z)``. **Left unset it is chosen from
    the objects** by :func:`choose_cell_size` -- pass it when you know the source array's chunk
    shape, which is the value worth matching and the one no amount of looking at meshes can
    reveal. ``levels`` is how deep the octree goes; every level from 0 to ``levels - 1`` gets a
    file, because a gap is geometry a planner never asks for.

    ``simplifier`` is how a coarse level is made, and like ``codec`` it is a name out of a small
    vocabulary: ``"QUADRIC"`` (the default -- quadric-optimal shapes with the boundary pinned) or
    ``"GREEDY"``. Pass an instance instead -- ``maille.GreedyEdgeCollapse(placement="onto_fixed")``
    -- to adjust a backend's own settings, or your own object providing ``simplify``. See
    :mod:`maille.simplifiers`. ``decimation`` is how much survives each level,
    defaulting to :meth:`maille.Decimation.quarter` -- and whatever it is, the manifest declares
    what was actually done rather than the format's default name.
    """
    schemas = arrow_schemas()  # fail on a missing pyarrow before the expensive clipping
    backend = resolve_simplifier(simplifier)
    schedule = decimation or Decimation.quarter()

    # Validated here rather than at the first blob, so a bad pair costs nothing instead of
    # costing the whole clipping pass -- and the same for a missing optional codec.
    Encoding(codec=codec, compression=compression)
    if codec == CODEC_MESHOPT:
        from maille.codecs import require_meshoptimizer

        require_meshoptimizer()
    if levels < 1:
        raise FormatError(f"An octree has at least one level, got {levels}.")
    if len(objects) > MAX_ORDINAL:
        raise FormatError(
            f"The format's dense ordinal is 24 bits, so a collection holds at most {MAX_ORDINAL} objects."
        )

    meshes = coerce_objects(objects)
    if not meshes:
        raise FormatError("A collection holds at least one object.")

    if cell_size is None:
        cell_size = choose_cell_size(meshes, levels=levels)
    grid = Grid(cell_size=tuple(int(c) for c in cell_size), levels=levels)  # type: ignore[arg-type]
    cell_size_array = np.asarray(grid.cell_size, dtype=np.int64)

    object_ids = sorted(meshes)  # ascending, as the format requires
    ordinals = {object_id: ordinal for ordinal, object_id in enumerate(object_ids)}

    # --- level 0: cut every object once, at the level-0 planes -------------------------
    #
    # This is the only cut that ever happens. Coarser levels group these fragments rather than
    # re-cutting, which is what keeps a boundary vertex in exactly the same place at every
    # level and makes `boundary: LOCKED` true rather than merely declared.
    #
    # Snapping happens here, once, on the level-0 vertices every level is later built from --
    # so a boundary vertex is at a position all levels can name exactly. See `snap_boundary`
    # for why an odd 65535 makes that a separate step.
    coarse_extent = cell_size_array.astype(np.float64) * (2 ** (levels - 1))
    fragments_by_object: dict[
        int, dict[tuple[int, int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]]
    ] = {}
    for object_id in object_ids:
        fragments: dict[tuple[int, int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]] = {}
        for triple, fragment in clip_to_cells(meshes[object_id], cell_size_array).items():
            vertices, boundary = snap_boundary(
                np.asarray(fragment.vertices, dtype=np.float64), cell_size_array, coarse_extent
            )
            faces = drop_degenerate(vertices, np.asarray(fragment.faces, dtype=np.int64))
            if len(faces) == 0:
                continue
            fragments[triple] = (vertices, faces, boundary)
        fragments_by_object[object_id] = fragments

    # How much of the finest level is pinned to a cell face. Reported only if the decimation
    # actually falls short below -- a high share is the *explanation* for a missed budget, not
    # a problem by itself, and warning on the proxy rather than the outcome cries wolf on
    # collections whose coarse levels came out fine.
    locked = sum(
        int(boundary.sum())
        for fragments in fragments_by_object.values()
        for _, _, boundary in fragments.values()
    )
    pinnable = sum(
        len(vertices)
        for fragments in fragments_by_object.values()
        for vertices, _, _ in fragments.values()
    )

    # --- the per-level geometry ------------------------------------------------------
    #
    # A level's representation is standalone: the whole object is decimated to that level's
    # budget and then split by the fragments' cells, never expressed as a delta on a finer one.
    per_level: dict[int, dict[int, dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]]]] = {}
    #: The simplifier's own estimate of how far it strayed, per ``(level, cell)`` -- the
    #: decimation half of that cell's ``lod_error``.
    displacement: dict[tuple[int, int], float] = {}
    #: Per level, how many object-in-cell pieces the schedule could take nothing more from --
    #: either because they were already at the face floor, or because the target had to be
    #: relaxed to keep the piece from being decimated out of existence.
    at_the_floor: dict[int, int] = {level: 0 for level in range(levels)}

    for level in range(levels):
        level_cells: dict[int, dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]]] = {}
        level_extent = cell_size_array * (2**level)
        for object_id in object_ids:
            fragments = fragments_by_object[object_id]
            if not fragments:
                continue

            # Group this object's level-0 fragments by the cell that holds them at *this*
            # level. The eight children tile their parent exactly, so a coarse cell is
            # assembled by merging -- never by cutting again, which is what would move a
            # boundary and break LOCKED.
            groups: dict[tuple[int, int, int], list[tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]]] = {}
            for triple, (vertices, faces, _) in fragments.items():
                coarse = tuple(int(component) // (2**level) for component in triple)
                groups.setdefault(coarse, []).append((vertices, faces))

            for coarse_triple, pieces in groups.items():
                vertices, faces = concatenate_and_weld(pieces)
                cell = morton_encode_one(coarse_triple)

                if level > 0:
                    # Welding the children dissolves the seams *interior* to this cell: those
                    # planes are not faces here, so their vertices are free and the decimator
                    # can spend them. Only vertices on this level's own faces stay locked --
                    # which is what keeps `QUARTER` reachable instead of stalling against a
                    # boundary inherited from the finest level.
                    own_faces = on_planes(vertices, level_extent)
                    wanted_faces = round(len(faces) * (schedule.ratio**level))
                    target = schedule.target_faces(len(faces), level)
                    # The schedule has nothing left to take when the ratio it asks for is finer
                    # than the face floor, or when the piece is already under the target. Both
                    # are a different explanation for a missed budget than a boundary the
                    # simplifier may not spend, and the warning must not confuse them.
                    at_floor = wanted_faces <= schedule.floor_faces or len(faces) <= target
                    result, relaxed = simplify_to_target(
                        backend, vertices, faces, fixed=own_faces, target_faces=target
                    )
                    vertices, faces = result.vertices, result.faces
                    # Per *cell*, not per level. A level-wide maximum would let one badly
                    # simplified cell set the error for every cell at that level, and
                    # `lod_error` is exactly what a planner spends its budget against -- so a
                    # single bad cell would drag the whole level down to full detail and the
                    # octree would never be used.
                    displacement[(level, cell)] = max(
                        displacement.get((level, cell), 0.0), result.error
                    )
                    if at_floor or relaxed:
                        at_the_floor[level] += 1

                level_cells.setdefault(cell, {})[object_id] = (vertices, faces)
        per_level[level] = level_cells

    # --- assemble the shards and the cell catalog -------------------------------------
    shards: list[tuple[int, pa.Table]] = []
    cell_rows: list[dict[str, Any]] = []
    #: Each cell's resolved `lod_error`, filled level by level so a parent can dominate its
    #: children -- levels are assembled finest-first, so a child is always already in here.
    cell_errors: dict[tuple[int, int], float] = {}
    object_cells: dict[int, list[dict[str, int]]] = {object_id: [] for object_id in object_ids}
    object_totals: dict[int, tuple[int, int]] = {object_id: (0, 0) for object_id in object_ids}
    #: The (low, high) corner of each object's bounds, in voxels.
    object_bounds: dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

    for level in range(levels):
        rows: list[dict[str, Any]] = []
        for cell in sorted(per_level[level]):
            bucket = per_level[level][cell]
            ids = sorted(bucket)  # ascending object ids, as the format requires

            all_vertices: list[npt.NDArray[np.float64]] = []
            all_faces: list[npt.NDArray[np.int64]] = []
            vertex_offsets: list[int] = []
            index_offsets: list[int] = []
            vertex_cursor = 0
            index_cursor = 0
            for object_id in ids:
                vertices, faces = bucket[object_id]
                vertex_offsets.append(vertex_cursor)
                index_offsets.append(index_cursor)
                all_vertices.append(vertices)
                all_faces.append(faces + vertex_cursor)  # cell-global, not object-local
                vertex_cursor += len(vertices)
                index_cursor += len(faces) * 3

                if level == 0:
                    object_totals[object_id] = (
                        object_totals[object_id][0] + len(vertices),
                        object_totals[object_id][1] + len(faces) * 3,
                    )
                    low, high = vertices.min(axis=0), vertices.max(axis=0)
                    if object_id in object_bounds:
                        prev_low, prev_high = object_bounds[object_id]
                        low, high = np.minimum(low, prev_low), np.maximum(high, prev_high)
                    object_bounds[object_id] = (low, high)
                object_cells[object_id].append({"level": level, "cell": cell})

            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            low, high = vertices.min(axis=0), vertices.max(axis=0)

            _, extent = cell_box(cell, level, grid.cell_size)
            lod_error = float(extent.max() / QUANT_MAX + displacement.get((level, cell), 0.0))

            # A parent's error must dominate its children's, and this is what makes a descent
            # well-founded: a planner keeps a cell when its error fits the budget and descends
            # when it does not, so a child that reported *more* error than its parent would be
            # refined into by a tighter budget and then look worse -- a budget that buys less
            # detail the more of it you spend. Greedy decimation does not guarantee the
            # ordering on its own (a coarse cell can happen to collapse a shorter edge than the
            # fine one below it), so it is imposed here, where every child is already known.
            for child in _children_of(cell, level, per_level):
                lod_error = max(lod_error, cell_errors.get((level - 1, child), 0.0))
            cell_errors[(level, cell)] = lod_error

            rows.append(
                {
                    "level": level,
                    "cell": cell,
                    "positions": encode_positions(
                        vertices,
                        cell=cell,
                        level=level,
                        cell_size=grid.cell_size,
                        codec=codec,
                        compression=compression,
                    ),
                    "indices": encode_indices(faces, codec=codec, compression=compression),
                    "vertex_count": len(vertices),
                    "index_count": len(faces) * 3,
                    "object_ids": ids,
                    "object_ordinals": [ordinals[object_id] for object_id in ids],
                    "object_vertex_offsets": vertex_offsets,
                    "object_index_offsets": index_offsets,
                }
            )
            cell_rows.append(
                {
                    "level": level,
                    "cell": cell,
                    "vertex_count": len(vertices),
                    "index_count": len(faces) * 3,
                    "bbox_min_x": low[0],
                    "bbox_min_y": low[1],
                    "bbox_min_z": low[2],
                    "bbox_max_x": high[0],
                    "bbox_max_y": high[1],
                    "bbox_max_z": high[2],
                    "lod_error": lod_error,
                    "object_count": len(ids),
                    "child_mask": _child_mask(cell, level, per_level),
                    # The locator, left unfilled here on purpose: which part holds a cell and
                    # which row group of it are facts about the *serialized* geometry, and
                    # nothing knows them until `write_collection` has actually written it.
                    "part": None,
                    "row_group": None,
                    "blob_bytes": None,
                }
            )
        shards.append((level, build_table(rows, schemas["geometry"])))

    _warn_if_decimation_missed(
        cell_rows, levels, locked, pinnable, grid.cell_size, at_the_floor, schedule, backend
    )

    object_rows = []
    for object_id in object_ids:
        low, high = object_bounds.get(object_id, (np.zeros(3), np.zeros(3)))
        vertex_count, index_count = object_totals[object_id]
        object_rows.append(
            {
                "object_id": object_id,
                "ordinal": ordinals[object_id],
                "bbox_min_x": low[0],
                "bbox_min_y": low[1],
                "bbox_min_z": low[2],
                "bbox_max_x": high[0],
                "bbox_max_y": high[1],
                "bbox_max_z": high[2],
                "vertex_count": vertex_count,
                "index_count": index_count,
                "cells": object_cells[object_id],
            }
        )

    manifest = Manifest(
        grid=grid,
        encoding=Encoding(codec=codec, compression=compression, decimation=schedule.declaration),
        counts={
            "objects": len(object_ids),
            "cellsPerLevel": [len(per_level[level]) for level in range(levels)],
        },
        files={
            "cells": CELL_CATALOG_PATH,
            "objects": OBJECT_CATALOG_PATH,
            "levels": {str(level): [level_part_path(level)] for level in range(levels)},
        },
    )

    return MeshCollection(
        cell_catalog=build_table(cell_rows, schemas["cell_catalog"]),
        object_catalog=build_table(object_rows, schemas["object_catalog"]),
        shards=shards,
        manifest=manifest,
    )


def _warn_if_decimation_missed(
    cell_rows: Sequence[Mapping[str, Any]],
    levels: int,
    locked: int,
    pinnable: int,
    cell_size: Sequence[int],
    at_the_floor: Mapping[int, int],
    schedule: Decimation,
    backend: Simplifier,
) -> None:
    """Say so when the declared reduction did not actually happen.

    It is a declaration nothing downstream can check and no reader will notice, so a coarse
    level that saved nothing costs a file, an upload and a fetch while looking exactly like one
    that worked. The usual cause is a cell size small relative to the objects: every cut vertex
    is pinned by ``boundary: LOCKED`` and the decimator may not spend it.
    """
    faces_at = {
        level: sum(row["index_count"] for row in cell_rows if row["level"] == level) // 3
        for level in range(levels)
    }

    # A coarse level *larger* than the one it summarises is not a missed budget, it is an
    # inversion: a viewer that zooms out downloads more and draws worse. It comes from the last
    # resort in `_decimate_to_the_tightest_target_that_survives`, where a surface this decimator
    # cannot coarsen at any target is kept whole -- rare, and stated rather than hidden, because
    # nothing downstream would report it.
    inverted = [level for level in range(1, levels) if faces_at[level] > faces_at[level - 1]]
    if inverted:
        worst_inversion = ", ".join(
            f"level {level} holds {faces_at[level]} faces against level {level - 1}'s {faces_at[level - 1]}"
            for level in inverted
        )
        warnings.warn(
            f"A coarse level came out larger than the level it summarises ({worst_inversion}). This decimator "
            f"could not coarsen some surface at any target and kept it whole rather than dropping it from the "
            f"level, so a zoomed-out view of this collection fetches more than a close one. Consider a larger "
            f"`cell_size` or fewer `levels`.",
            stacklevel=3,
        )

    missed = [
        (level, faces_at[level] / faces_at[level - 1])
        for level in range(1, levels)
        if faces_at[level - 1] and faces_at[level] / faces_at[level - 1] > 2 * schedule.ratio
    ]
    if not missed:
        return

    worst = ", ".join(
        f"level {level} kept {ratio:.0%} of level {level - 1}" for level, ratio in missed
    )

    # Two quite different causes look identical in the ratio, so the message names whichever
    # one the numbers actually support rather than always blaming the cell size.
    floored = sum(at_the_floor.get(level, 0) for level, _ in missed)
    if floored:
        cause = (
            f"{floored} object piece(s) had nothing left to take: they were already at the "
            f"{schedule.floor_faces}-face floor, or decimating them to the target would have removed them from the level "
            f"altogether -- and an object that vanishes when a viewer zooms out is a worse artifact than a "
            f"coarse level that saves little. This is expected for a collection of small objects."
        )
    elif getattr(backend, "uses_fixed_mask", True):
        cause = (
            f"{locked}/{pinnable} level-0 vertices ({locked / max(pinnable, 1):.0%}) lie on a cell face and are "
            f"pinned by `boundary: LOCKED`, so the {getattr(backend, 'name', 'simplifier')} backend cannot spend "
            f"them. The cell size "
            f"{tuple(int(component) for component in cell_size)} is small relative to the objects -- pass a "
            f"larger `cell_size`, leave it unset so `choose_cell_size` picks one, or reduce `levels`, since a "
            f"coarse level that saves nothing still costs a file and a fetch."
        )
    else:
        # This backend is held by the topological boundary rather than by the `fixed` mask, so
        # the share of vertices in that mask would explain the miss with a figure that does not
        # describe the constraint. Name the cause without a number that is not about it.
        cause = (
            f"The {getattr(backend, 'name', 'simplifier')} backend locks the cut boundary and preserves topology, "
            f"so it stops short of a budget rather than destroying a surface to reach one. The cell size "
            f"{tuple(int(component) for component in cell_size)} being small relative to the objects leaves most of "
            f"each fragment on a boundary -- pass a larger `cell_size`, leave it unset so `choose_cell_size` picks "
            f"one, reduce `levels`, or pass `simplifier=GreedyEdgeCollapse()`, which will collapse onto a locked "
            f"vertex and reach the budget at the cost of a much looser error bound."
        )

    warnings.warn(
        f"`decimation: {schedule.declaration}` targets {schedule.ratio:.0%} of the faces per level, "
        f"but {worst}. {cause}",
        stacklevel=3,
    )


def _children_of(cell: int, level: int, per_level: Mapping[int, Mapping[int, Any]]) -> list[int]:
    """The cells at level ``L-1`` that hold geometry inside this cell."""
    if level == 0:
        return []
    finer = per_level.get(level - 1, {})
    i, j, k = morton_decode(cell)
    found = []
    for octant in range(8):
        dx, dy, dz = octant & 1, (octant >> 1) & 1, (octant >> 2) & 1
        child = morton_encode_one((2 * i + dx, 2 * j + dy, 2 * k + dz))
        if child in finer:
            found.append(child)
    return found


def _child_mask(cell: int, level: int, per_level: Mapping[int, Mapping[int, Any]]) -> int:
    """Which of the eight children at level ``L-1`` hold geometry. Zero at the finest level.

    Bit ``dx | dy<<1 | dz<<2`` is set when the child at triple ``(2i+dx, 2j+dy, 2k+dz)`` holds
    geometry -- which is what lets a planner descend without listing the level below.
    """
    if level == 0:
        return 0
    finer = per_level.get(level - 1, {})
    i, j, k = morton_decode(cell)
    mask = 0
    for octant in range(8):
        dx, dy, dz = octant & 1, (octant >> 1) & 1, (octant >> 2) & 1
        child = morton_encode_one((2 * i + dx, 2 * j + dy, 2 * k + dz))
        if child in finer:
            mask |= 1 << octant
    return mask


__all__ = ["MeshCollection", "build_collection", "choose_cell_size"]
