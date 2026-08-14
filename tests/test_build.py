"""Building a collection: the frames, the invariants, and the axis order.

The first test in this file is the most valuable one in the package. A cubic ``cell_size`` and
a spherical object make a transposed writer pass every check there is -- the counts are right,
the round trip closes, the columns are the right type, and the geometry draws sideways. So the
fixtures are anisotropic and so is this test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh

import maille
from tests.conftest import CELL_SIZE, LEVELS


def test_geometry_round_trips_through_an_anisotropic_grid(collection: maille.MeshCollection, objects: dict):
    """Decoded vertices land where the source vertices were, with a non-cubic cell.

    If ``cell_size`` were used in the wrong component order anywhere -- when quantizing, when
    finding a cell box, when cutting -- a ``(128, 128, 64)`` grid would put the geometry
    somewhere else entirely. With a cubic grid this test would pass regardless.
    """
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    for object_id, source in objects.items():
        decoded = opened.object_mesh(object_id, level=0)
        source_low, source_high = np.asarray(source.bounds[0]), np.asarray(source.bounds[1])

        # A quantum of the finest cell is the whole error budget for a level-0 vertex.
        quantum = np.asarray(CELL_SIZE, dtype=np.float64) / maille.QUANT_MAX
        assert np.asarray(decoded.bounds[0]) == pytest.approx(source_low, abs=quantum.max() * 2)
        assert np.asarray(decoded.bounds[1]) == pytest.approx(source_high, abs=quantum.max() * 2)


def test_a_cell_addresses_the_box_its_geometry_actually_occupies(collection: maille.MeshCollection):
    """Every vertex lies inside the box its cell's Morton code names, at every level.

    This is the property the server's one-sided ``cellSize`` cross-check leans on: geometry
    outside its address box is not representable, so an observed extent larger than
    ``cell_size * 2**level`` proves the declared size is not the one the octree was cut with.
    """
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    for entry in opened.cells.values():
        origin, extent = opened.cell_box(entry.level, entry.cell)
        cell = opened.read_cell(entry.level, entry.cell)
        assert (cell.vertices >= origin - 1e-6).all(), f"cell {entry.cell} at level {entry.level} spills low"
        assert (cell.vertices <= origin + extent + 1e-6).all(), f"cell {entry.cell} at level {entry.level} spills high"


def test_the_manifest_makes_no_claim_about_what_the_components_mean():
    """maille computes in positional (x, y, z) and states nothing about which axis a slot is.

    Naming those axes is a statement about how the collection relates to whatever it came from,
    which belongs to the layer that owns the coordinate system -- not to a mesh serializer. A
    key here would be a claim nothing in the format could check, use or contradict.
    """
    objects = {1: trimesh.creation.icosphere(radius=20.0).apply_translation([64.0, 64.0, 32.0])}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = maille.build_collection(objects, cell_size=(128, 128, 64), levels=2)

    written = built.manifest.to_dict()
    assert set(written) == {"specVersion", "grid", "encoding", "counts", "files"}
    assert "axes" not in written


def test_the_encoding_always_states_the_keys_a_decoder_cannot_infer(collection: maille.MeshCollection):
    """``codec``, ``compression`` and ``indices`` are never omitted or defaulted downstream.

    A wrong value here is not an error at any layer -- it is geometry that decodes to garbage --
    so the manifest states all of them, always, even where there is currently one legal value.
    """
    encoding = collection.manifest.to_dict()["encoding"]

    for key in ("positions", "indices", "codec", "compression", "boundary", "decimation"):
        assert encoding.get(key), f"the manifest left `{key}` for a reader to guess"
    assert encoding["boundary"] == maille.BOUNDARY_LOCKED
    assert encoding["compression"] == maille.COMPRESSION_NONE, "declared only where it is applied"


def test_the_declared_codec_is_the_one_that_was_applied():
    """A declaration that does not match the bytes is a lie no check downstream could catch."""
    objects = {1: trimesh.creation.icosphere(radius=20.0).apply_translation([64.0, 64.0, 32.0])}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = maille.build_collection(objects, cell_size=(128, 128, 64), levels=2, codec=maille.CODEC_NONE)

    assert raw.manifest.encoding.codec == maille.CODEC_NONE
    row = raw.shards[0][1]
    positions = row.column("positions")[0].as_py()
    assert len(positions) == 6 * row.column("vertex_count")[0].as_py(), "NONE must write the blob raw"


def test_object_ids_and_ordinals_follow_the_format(collection: maille.MeshCollection):
    """Ids ascend, ordinals are dense from zero, and the two agree with the object catalog."""
    catalog = collection.object_catalog
    ids = catalog.column("object_id").to_pylist()
    ordinals = catalog.column("ordinal").to_pylist()

    assert ids == sorted(ids), "the object catalog is ordered by ascending id"
    assert ordinals == list(range(len(ids))), "ordinals are dense and match the catalog's row order"

    for _, shard in collection.shards:
        for row in range(shard.num_rows):
            row_ids = shard.column("object_ids")[row].as_py()
            assert row_ids == sorted(row_ids), "a cell lists its objects in ascending id order"
            assert shard.column("object_ordinals")[row].as_py() == [ids.index(i) for i in row_ids]


def test_object_offsets_are_ascending_starts_into_the_concatenated_arrays(collection: maille.MeshCollection):
    """The offsets are what make a cell decodable into per-object meshes."""
    for _, shard in collection.shards:
        for row in range(shard.num_rows):
            vertex_offsets = shard.column("object_vertex_offsets")[row].as_py()
            index_offsets = shard.column("object_index_offsets")[row].as_py()

            assert vertex_offsets[0] == 0 and index_offsets[0] == 0
            assert vertex_offsets == sorted(vertex_offsets)
            assert index_offsets == sorted(index_offsets)
            assert vertex_offsets[-1] < shard.column("vertex_count")[row].as_py()
            assert all(offset % 3 == 0 for offset in index_offsets), "an index offset lands on a triangle"


def test_the_child_mask_names_the_children_that_exist(collection: maille.MeshCollection):
    """A planner descends on the mask alone, so a wrong bit is geometry never fetched."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    for entry in opened.cells.values():
        if entry.level == 0:
            assert entry.child_mask == 0, "the finest level has no children"
            continue
        for key in entry.children():
            assert key in opened.cells, f"cell {entry.cell} claims a child {key} that does not exist"

        actual = {key for key in opened.cells if key[0] == entry.level - 1}
        claimed = set(entry.children())
        parents = {(entry.level - 1, cell) for _, cell in actual if _is_child(cell, entry)}
        assert claimed == parents, "the mask disagrees with the level below"


def _is_child(cell: int, parent: maille.CellEntry) -> bool:
    """Whether a finer cell is one of a parent's eight children."""
    i, j, k = maille.morton_decode(cell)
    return (i // 2, j // 2, k // 2) == parent.triple


def test_every_level_holds_every_object(collection: maille.MeshCollection):
    """A level is a standalone representation, not a delta on a finer one."""
    per_level = {
        level: {object_id for row in range(shard.num_rows) for object_id in shard.column("object_ids")[row].as_py()}
        for level, shard in collection.shards
    }
    expected = set(collection.object_catalog.column("object_id").to_pylist())

    for level in range(LEVELS):
        assert per_level[level] == expected, f"level {level} is missing objects the collection declares"


def test_every_coarse_level_saves_something_real(collection: maille.MeshCollection):
    """A coarse level that saved nothing costs a file, an upload and a fetch for no benefit.

    Deliberately *not* asserting the ``QUARTER`` ratio: the declaration names the target, and a
    topology-preserving simplifier stops short of it rather than destroying a surface to reach
    it. What must always hold is that each level is meaningfully smaller than the one below --
    the ratio it actually achieved is reported by the build warning and readable per cell.
    """
    faces = {
        level: sum(shard.column("index_count").to_pylist()) // 3
        for level, shard in collection.shards
    }
    for level in range(1, LEVELS):
        assert faces[level] < faces[level - 1] * 0.8, (
            f"level {level} kept {faces[level] / faces[level - 1]:.0%} of level {level - 1}, which is not a "
            f"coarser level in any useful sense"
        )


def test_a_coarse_level_never_holds_more_geometry_than_the_level_below(collection: maille.MeshCollection):
    """A coarser level summarises a finer one, so it cannot cost more than what it summarises.

    Worth stating as its own invariant because the way it breaks is not an error: a decimator
    that destroys a surface at an aggressive target, plus a writer that responds by keeping the
    surface whole, produces a perfectly valid collection whose level 2 is *larger* than its
    level 1 -- a coarse fetch that downloads more and draws worse.
    """
    faces = {level: sum(shard.column("index_count").to_pylist()) // 3 for level, shard in collection.shards}

    for level in range(1, LEVELS):
        assert faces[level] <= faces[level - 1], f"level {level} holds more geometry than level {level - 1}"


def test_a_missed_quarter_budget_is_warned_about():
    """A coarse level that saved nothing costs a file, an upload and a fetch, and looks fine.

    The cell here is far smaller than the object, which is the cause worth warning about: every
    cut vertex is pinned by ``boundary: LOCKED`` and the decimator may not spend it, so
    ``QUARTER`` stalls at about half however many levels are asked for.
    """
    objects = {1: trimesh.creation.icosphere(radius=40.0, subdivisions=3).apply_translation([80.0, 80.0, 80.0])}

    with pytest.warns(UserWarning, match="decimation: QUARTER"):
        maille.build_collection(objects, cell_size=(4, 4, 4), levels=2)


def test_choosing_a_cell_size_beats_a_cell_smaller_than_the_objects(objects: dict):
    """The heuristic's job is avoiding the choice that goes wrong silently."""
    chosen = maille.choose_cell_size({key: maille.coerce_mesh(value) for key, value in objects.items()})

    assert len(chosen) == 3
    assert all(size >= 8 for size in chosen)
    assert all(size & (size - 1) == 0 for size in chosen), "powers of two keep coarse planes a subset of fine ones"


def test_raw_vertex_and_face_arrays_are_accepted_without_trimesh_in_the_caller(objects: dict):
    """A caller who already has arrays should not need trimesh in their own code."""
    source = objects[3]
    pairs = {3: (np.asarray(source.vertices), np.asarray(source.faces))}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_pair = maille.build_collection(pairs, cell_size=CELL_SIZE, levels=2)
        from_mesh = maille.build_collection({3: source}, cell_size=CELL_SIZE, levels=2)

    assert from_pair.cell_catalog.to_pylist() == from_mesh.cell_catalog.to_pylist()


def test_faces_that_do_not_belong_to_their_vertices_are_refused():
    """Caught at coercion, where the message can still name the object."""
    with pytest.raises(ValueError, match="Object 4"):
        maille.build_collection(
            {4: (np.zeros((3, 3)), np.array([[0, 1, 9]]))}, cell_size=CELL_SIZE
        )


def test_lod_error_is_per_cell_rather_than_per_level(collection: maille.MeshCollection):
    """A level-wide maximum would let one bad cell set the error for the whole level.

    ``lod_error`` is what a planner spends its budget against, so a single badly decimated cell
    reporting for all of them drags the entire level down to full detail and the octree stops
    doing its job. Nothing errors -- the collection is valid and simply never gets used.
    """
    catalog = collection.cell_catalog
    per_level: dict[int, list[float]] = {}
    for level, error in zip(catalog.column("level").to_pylist(), catalog.column("lod_error").to_pylist()):
        per_level.setdefault(int(level), []).append(round(float(error), 9))

    # Level 0 is uniform by construction -- nothing is decimated there, so its error is the
    # quantization step and no more. It is the coarse levels that must vary.
    checked = 0
    for level, errors in sorted(per_level.items()):
        if level == 0 or len(errors) < 2:
            continue
        checked += 1
        assert len(set(errors)) > 1, (
            f"every one of the {len(errors)} cells at level {level} reports lod_error {errors[0]}, "
            f"which is what a per-level maximum looks like"
        )

    assert checked, "this fixture has no coarse level holding more than one cell, so it proves nothing"


def test_a_parents_error_dominates_its_childrens(collection: maille.MeshCollection):
    """What makes the descent well-founded.

    A planner keeps a cell when its error fits the budget and descends when it does not. A
    child reporting *more* error than its parent would therefore be refined into by a tighter
    budget and then look worse -- a budget that buys less detail the more of it you spend.
    """
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    for entry in opened.cells.values():
        for key in entry.children():
            child = opened.cells[key]
            assert entry.lod_error >= child.lod_error - 1e-12, (
                f"cell {entry.cell} at level {entry.level} reports {entry.lod_error}, less than its child "
                f"{child.cell} at level {child.level} reporting {child.lod_error}"
            )


def test_level_zero_carries_only_the_quantization_error(collection: maille.MeshCollection):
    """Nothing is decimated at level 0, so its error is the quantization step and nothing else."""
    catalog = collection.cell_catalog
    expected = max(collection.grid.cell_extent(0)) / maille.QUANT_MAX

    for row in range(catalog.num_rows):
        if catalog.column("level")[row].as_py() == 0:
            assert catalog.column("lod_error")[row].as_py() == pytest.approx(expected, rel=1e-9)


def test_tightening_the_budget_never_coarsens_a_region(collection: maille.MeshCollection):
    """The property the parent-dominates-children rule exists to establish.

    Not "the plan keeps covering everything" -- that holds for other reasons and is checked
    elsewhere. This is the sharper one: for any given patch of the collection, the level chosen
    to draw it is monotone as the budget tightens. Without a parent whose error dominates its
    children's, a tighter budget could refine into a child that reports *more* error and be
    handed back a worse-looking cell for the same region.
    """
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    regions = [(entry.level, entry.cell) for entry in opened.cells_at(0)]
    previous: dict[tuple[int, int], int] = {}

    for budget in (1e9, 100.0, 50.0, 20.0, 5.0, 0.0):
        chosen: dict[tuple[int, int], int] = {}
        for entry in opened.plan(error_budget=budget):
            for region in regions:
                if _covers((entry.level, entry.cell), region):
                    chosen[region] = entry.level

        for region, level in chosen.items():
            if region in previous:
                assert level <= previous[region], (
                    f"tightening the budget to {budget} drew region {region} at level {level}, coarser than the "
                    f"level {previous[region]} a looser budget gave it"
                )
        previous = chosen


def _covers(candidate: tuple[int, int], target: tuple[int, int]) -> bool:
    """Whether ``candidate`` is ``target`` or one of its descendants in the octree."""
    candidate_level, candidate_cell = candidate
    target_level, target_cell = target
    if candidate_level > target_level:
        return False
    steps = target_level - candidate_level
    i, j, k = maille.morton_decode(candidate_cell)
    return (i >> steps, j >> steps, k >> steps) == maille.morton_decode(target_cell)


def test_the_component_order_is_the_callers_and_maille_never_interprets_it():
    """maille addresses components by position, never by name -- so any order works.

    Feed vertices and ``cell_size`` in the same order and the octree is identical whatever that
    order is: the same cells, the same partition, geometry decoding back to exactly what went
    in. This matters because meshes usually come out of marching cubes over a ``(z, y, x)``
    array, and having to transpose them into some house convention first is both work and a
    chance to get it wrong.

    The ``x``/``y``/``z`` in the ``bbox_*`` column names are labels for components 0, 1 and 2 --
    fixed by the Parquet schema a server checks -- not a claim about which physical axis each
    one is. The format makes no such claim anywhere.
    """
    source = trimesh.creation.box(extents=[300.0, 170.0, 90.0]).apply_translation([260.0, 210.0, 95.0])
    vertices = np.asarray(source.vertices)
    faces = np.asarray(source.faces)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forward = maille.build_collection({7: (vertices, faces)}, cell_size=(128, 128, 64), levels=3)
        # The same scene in the reverse order: components reversed, cell size reversed with them.
        reversed_ = maille.build_collection(
            {7: (vertices[:, ::-1], faces)}, cell_size=(64, 128, 128), levels=3
        )

    assert [shard.num_rows for _, shard in forward.shards] == [shard.num_rows for _, shard in reversed_.shards], (
        "the octree partition differs between component orders, so something reads position as meaning"
    )

    store = maille.MemoryStore()
    maille.write_collection(reversed_, store, "c")
    decoded = maille.open_collection(store, "c").object_mesh(7)

    # Reversed back, it is the original scene -- to within a quantum of the finest cell.
    quantum = max(64, 128, 128) / maille.QUANT_MAX
    assert np.asarray(decoded.bounds[0])[::-1] == pytest.approx(np.asarray(source.bounds[0]), abs=quantum * 4)
    assert np.asarray(decoded.bounds[1])[::-1] == pytest.approx(np.asarray(source.bounds[1]), abs=quantum * 4)


def test_the_component_order_changes_the_partition_and_never_the_geometry():
    """The reassuring half: at this layer, an axis-order mistake cannot misplace geometry.

    Clipping and quantization both read the same ``cell_size``, so a writer is self-consistent
    with whatever it is handed. Give it a ``cell_size`` in the wrong order and you get a
    differently *shaped* octree -- cells that do not match the data's anisotropy, so the tree
    narrows a fetch less well -- but every vertex still decodes to exactly where it started.

    That is worth pinning down, because it says the failure mode here is efficiency rather than
    correctness. The order question that *can* draw something sideways lives a layer up, in how
    a renderer interprets the collection's relationship to its source -- which is why maille
    states nothing about it rather than inventing a claim.
    """
    # Tall in the third component, narrow in the first: an order mistake cannot go unnoticed.
    source = trimesh.creation.box(extents=[40.0, 40.0, 400.0]).apply_translation([60.0, 60.0, 260.0])
    objects = {1: (np.asarray(source.vertices), np.asarray(source.faces))}

    decoded = {}
    for label, cell_size in (("matched", (128, 128, 512)), ("reversed", (512, 128, 128))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collection = maille.build_collection(objects, cell_size=cell_size, levels=1)
        store = maille.MemoryStore()
        maille.write_collection(collection, store, "c")
        decoded[label] = (collection.shards[0][1].num_rows, maille.open_collection(store, "c").object_mesh(1))

    assert decoded["matched"][0] != decoded["reversed"][0], (
        "an anisotropic cell size read in the wrong order should partition differently"
    )

    quantum = 512 / maille.QUANT_MAX
    for corner in (0, 1):
        assert np.asarray(decoded["reversed"][1].bounds[corner]) == pytest.approx(
            np.asarray(decoded["matched"][1].bounds[corner]), abs=quantum * 4
        ), "the component order moved the geometry, which it must never do"
