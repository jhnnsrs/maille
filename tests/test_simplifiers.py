"""The simplification backends: what each promises, and what both must not break.

The invariants here are the ones no reader can check and no server can verify, so a backend
that quietly violated one would produce a collection that looks perfect and draws wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh

import maille
from maille.geometry import border_vertices
from maille.simplifiers import quadric as quadric_module
from maille.simplifiers import (
    boundary_held,
    measure_deviation,
    resolve_simplifier,
    simplifier_for,
    simplify_to_target,
)

BACKENDS = (maille.SIMPLIFICATION_QUADRIC, maille.SIMPLIFICATION_GREEDY)


def backend(name: str) -> maille.Simplifier:
    """One of the two backends, by the name that selects it."""
    return simplifier_for(name)


def a_sphere(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A closed surface with no boundary at all."""
    mesh = trimesh.creation.icosphere(radius=20.0, subdivisions=subdivisions).apply_translation([50.0, 50.0, 50.0])
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def a_cut_sheet() -> tuple[np.ndarray, np.ndarray]:
    """A surface with an open boundary -- the shape a clipped fragment actually has."""
    mesh = trimesh.creation.icosphere(radius=20.0, subdivisions=3).apply_translation([50.0, 50.0, 50.0])
    cut = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, 50.0], cap=False
    )
    return np.asarray(cut.vertices, dtype=np.float64), np.asarray(cut.faces, dtype=np.int64)


# --------------------------------------------------------------------------- #
# What both backends must do
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", BACKENDS)
def test_a_backend_reduces_toward_the_target(name: str):
    """The one thing a face budget asks for."""
    vertices, faces = a_sphere()
    result = backend(name).simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
    )

    assert len(result.faces) < len(faces)
    assert result.backend == backend(name).name
    assert result.error >= 0.0


@pytest.mark.parametrize("name", BACKENDS)
def test_a_backend_leaves_a_mesh_already_under_target_alone(name: str):
    """No work to do, and no error to report for not doing it."""
    vertices, faces = a_sphere(subdivisions=1)
    result = backend(name).simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) * 2
    )

    assert len(result.faces) == len(faces)
    assert result.error == 0.0
    assert result.reached_target


@pytest.mark.parametrize("name", BACKENDS)
def test_a_backend_produces_faces_that_index_its_own_vertices(name: str):
    """A remap mistake here is geometry that decodes to nonsense rather than an exception."""
    vertices, faces = a_sphere()
    result = backend(name).simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 8
    )

    assert result.faces.min() >= 0
    assert result.faces.max() < len(result.vertices)
    degenerate = (
        (result.faces[:, 0] == result.faces[:, 1])
        | (result.faces[:, 1] == result.faces[:, 2])
        | (result.faces[:, 0] == result.faces[:, 2])
    )
    assert not degenerate.any(), "a degenerate triangle survived"


@pytest.mark.parametrize("name", BACKENDS)
def test_a_backend_keeps_the_boundary_where_it_was(name: str):
    """``LOCKED`` is the promise nothing downstream can check, so it is checked here.

    The cut curve is what the neighbouring cell continues across. Every vertex on it must still
    be present *and* at exactly its old position, or two cells drawn at different levels meet
    with a hole between them.
    """
    vertices, faces = a_cut_sheet()
    boundary = border_vertices(faces)
    assert len(boundary), "this fixture is supposed to have an open boundary"

    fixed = np.zeros(len(vertices), dtype=bool)
    fixed[boundary] = True

    result = backend(name).simplify(vertices, faces, fixed=fixed, target_faces=len(faces) // 4)

    for position in vertices[boundary]:
        assert np.isclose(result.vertices, position).all(axis=1).any(), (
            "a boundary vertex was moved or collapsed away, which is a crack between levels"
        )


# --------------------------------------------------------------------------- #
# What meshopt does that the fallback cannot
# --------------------------------------------------------------------------- #


def test_the_quadric_backend_pins_the_boundary_and_frees_the_interior():
    """The split that makes this the right algorithm for the format.

    A neighbouring cell shares exactly one thing with this one: the cut curve. So that curve
    must not move by any amount at all -- and everything else *should* move, to wherever the
    quadric says the shape is best preserved. A simplifier that held the interior still as well
    would be paying shape quality for a guarantee nobody needs.
    """
    vertices, faces = a_cut_sheet()
    boundary = border_vertices(faces)

    result = maille.QuadricSimplifier().simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
    )

    for position in vertices[boundary]:
        assert (result.vertices == position).all(axis=1).any(), (
            "a boundary vertex moved, which is a crack between levels"
        )

    interior = np.setdiff1d(np.arange(len(vertices)), boundary)
    unmoved = sum(bool((result.vertices == position).all(axis=1).any()) for position in vertices[interior])
    assert unmoved < len(interior), "no interior vertex moved, so this is not a quadric collapse"


def test_the_quadric_backend_reports_a_tighter_error_than_the_fallback():
    """The reason `error_budget` becomes usable at the scale its name implies.

    Held to the same target on the same mesh -- the only comparison that controls for how much
    geometry each one kept. The fallback can only report how far it moved a vertex, which on a
    collapsing object is about that object's radius.
    """
    vertices, faces = a_sphere()
    fixed = np.zeros(len(vertices), dtype=bool)
    target = len(faces) // 4

    quadric = maille.QuadricSimplifier().simplify(vertices, faces, fixed=fixed, target_faces=target)
    greedy = maille.GreedyEdgeCollapse().simplify(vertices, faces, fixed=fixed, target_faces=target)

    assert quadric.error < greedy.error
    assert quadric.error < 0.4 * 20.0, "a radius-20 sphere at a quarter of its faces should not stray far"


def test_the_deviation_measure_is_an_upper_bound_and_grows_with_reduction():
    """It decides whether a viewer fetches more detail, so it must over-report, never under."""
    vertices, faces = a_sphere(subdivisions=4)
    fixed = np.zeros(len(vertices), dtype=bool)

    errors = [
        maille.QuadricSimplifier().simplify(vertices, faces, fixed=fixed, target_faces=len(faces) // ratio).error
        for ratio in (2, 4, 8, 16)
    ]

    assert errors == sorted(errors), f"taking more geometry away reported less error: {errors}"
    assert all(error > 0.0 for error in errors)


def test_measuring_deviation_against_an_unchanged_mesh_is_zero():
    """The identity case, since the bound is what a budget is spent against."""
    vertices, _ = a_sphere(subdivisions=1)

    assert measure_deviation(vertices, vertices) == 0.0
    assert measure_deviation(vertices, vertices[:0]) == 0.0


def test_the_quadric_backend_falls_back_when_the_boundary_did_not_hold():
    """The verification is what turns a library flag into a guarantee.

    Forced here by refusing every result: a backend that cannot satisfy the boundary check must
    hand the piece to the fallback rather than ship a crack.
    """
    vertices, faces = a_sphere()

    class NeverHolds(maille.QuadricSimplifier):
        pass

    monkey = NeverHolds()
    original = quadric_module.boundary_held
    quadric_module.boundary_held = lambda *args, **kwargs: False
    try:
        result = monkey.simplify(
            vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
        )
    finally:
        quadric_module.boundary_held = original

    assert result.backend == maille.GreedyEdgeCollapse().name, "the fallback did not take over"


def test_the_boundary_check_notices_a_boundary_that_did_not_hold():
    """Directly, rather than only through a backend that is supposed to avoid it.

    Note what the check actually asks: *is some surviving vertex at exactly this position*. A
    clipped fragment carries duplicate vertices, so moving one of a pair leaves the position
    occupied and the boundary intact -- which is the right answer, and why the check is written
    against positions rather than against indices.
    """
    vertices, faces = a_cut_sheet()
    boundary = border_vertices(faces)

    assert boundary_held(vertices, faces, vertices)

    # Drop every vertex sharing the first boundary position, so the position is truly gone.
    missing = ~(vertices == vertices[boundary[0]]).all(axis=1)
    assert not boundary_held(vertices, faces, vertices[missing]), "a vanished boundary vertex is a gap"

    # And a wholesale shift of the boundary, which is the crack this exists to catch.
    shifted = vertices.copy()
    shifted[boundary] += 1e-6
    assert not boundary_held(vertices, faces, shifted), "a micrometre off is still a gap"


# --------------------------------------------------------------------------- #
# Choosing and relaxing
# --------------------------------------------------------------------------- #


def test_a_name_selects_a_backend_and_the_default_is_the_quadric_one():
    """`simplification` is a value out of a vocabulary, the way `codec` is."""
    assert isinstance(simplifier_for(maille.SIMPLIFICATION_QUADRIC), maille.QuadricSimplifier)
    assert isinstance(simplifier_for(maille.SIMPLIFICATION_GREEDY), maille.GreedyEdgeCollapse)
    assert isinstance(resolve_simplifier(None), maille.QuadricSimplifier), "the default is the quadric collapse"

    # A fresh instance each time, so one caller's adjustment is not everyone's.
    assert simplifier_for(maille.SIMPLIFICATION_GREEDY) is not simplifier_for(maille.SIMPLIFICATION_GREEDY)


def test_a_backend_this_package_does_not_ship_is_refused_by_name():
    """Naming what exists beats an AttributeError inside the build."""
    with pytest.raises(maille.FormatError, match="`simplification` is"):
        simplifier_for("MESHLAB")

    with pytest.raises(TypeError, match="QUADRIC"):
        resolve_simplifier(object())

    chosen = maille.GreedyEdgeCollapse()
    assert resolve_simplifier(chosen) is chosen
    assert isinstance(resolve_simplifier(maille.SIMPLIFICATION_GREEDY), maille.GreedyEdgeCollapse)


@pytest.mark.parametrize("name", BACKENDS)
def test_a_target_that_destroys_the_surface_is_relaxed_until_something_survives(name: str):
    """A level is a standalone representation, so a piece may never be simplified to nothing."""
    box = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    vertices = np.asarray(box.vertices, dtype=np.float64)
    faces = np.asarray(box.faces, dtype=np.int64)

    result, relaxed = simplify_to_target(
        backend(name), vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=1
    )

    assert len(result.faces) > 0, "the object was simplified out of existence"
    del relaxed  # whether it had to relax is backend-specific; that it survived is not


# --------------------------------------------------------------------------- #
# Through a whole build
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", BACKENDS)
def test_a_collection_built_with_either_backend_holds_the_same_invariants(name: str, objects: dict):
    """Both backends have to satisfy everything the format promises, not just the default one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        collection = maille.build_collection(
            objects, cell_size=(128, 128, 64), levels=3, simplifier=backend(name)
        )

    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    opened = maille.open_collection(store, "c")

    faces = {level: sum(shard.column("index_count").to_pylist()) // 3 for level, shard in collection.shards}
    for level in (1, 2):
        assert faces[level] <= faces[level - 1], "a coarse level holds more geometry than the level below"

    for object_id in opened.objects:
        for level in range(3):
            assert opened.cells_for_object(object_id, level=level), (
                f"object {object_id} vanished at level {level} under the {name} backend"
            )

    for entry in opened.cells.values():
        for key in entry.children():
            assert entry.lod_error >= opened.cells[key].lod_error - 1e-12


def test_the_backend_choice_does_not_change_the_partition(objects: dict):
    """Which simplifier runs is a quality decision, never a structural one.

    The cells, and which objects live in them, come out of the clipping and the octree -- so
    swapping the backend must change how many triangles a cell holds and nothing else about it.
    """
    built = {}
    for name in BACKENDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            built[name] = maille.build_collection(
                objects, cell_size=(128, 128, 64), levels=3, simplifier=backend(name)
            )

    for level in range(3):
        cells = {
            name: sorted(collection.shards[level][1].column("cell").to_pylist())
            for name, collection in built.items()
        }
        assert cells[maille.SIMPLIFICATION_QUADRIC] == cells[maille.SIMPLIFICATION_GREEDY], (
            f"level {level} was partitioned differently"
        )


def test_the_manifest_declares_the_reduction_that_was_actually_applied(objects: dict):
    """`decimation` is a claim about the geometry that nothing downstream can re-derive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quarter = maille.build_collection(objects, cell_size=(128, 128, 64), levels=2)
        half = maille.build_collection(
            objects, cell_size=(128, 128, 64), levels=2, decimation=maille.Decimation.half()
        )

    assert quarter.manifest.encoding.decimation == maille.DECIMATION_QUARTER
    assert half.manifest.encoding.decimation == maille.DECIMATION_HALF

    half_faces = sum(half.shards[1][1].column("index_count").to_pylist())
    quarter_faces = sum(quarter.shards[1][1].column("index_count").to_pylist())
    assert half_faces >= quarter_faces, "halving should keep at least as much as quartering"


def test_a_ratio_and_its_declaration_are_required_to_agree():
    """A writer cannot declare QUARTER while reducing by half; nothing downstream could catch it."""
    with pytest.raises(maille.FormatError, match="is HALF, not QUARTER"):
        maille.Decimation(ratio=0.5)

    assert maille.Decimation.custom(0.3).declaration == maille.DECIMATION_CUSTOM
    assert maille.Decimation.custom(0.25).declaration == maille.DECIMATION_QUARTER
    with pytest.raises(maille.FormatError, match="between 0 and 1"):
        maille.Decimation.custom(1.5)
    with pytest.raises(maille.FormatError, match="at least one tetrahedron"):
        maille.Decimation(ratio=0.25, floor_faces=2)


def test_the_face_floor_is_configurable():
    """A collection of small objects can ask for more of them to survive the coarse levels."""
    schedule = maille.Decimation.quarter(floor_faces=32)

    assert schedule.target_faces(1000, 0) == 1000
    assert schedule.target_faces(1000, 3) == 32, "the floor, not 15"
    assert maille.Decimation.quarter().target_faces(1000, 3) == 16
