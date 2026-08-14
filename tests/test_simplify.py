"""The simplification backends: what each promises, and what both must not break.

The invariants here are the ones no reader can check and no server can verify, so a backend
that quietly violated one would produce a collection that looks perfect and draws wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import maille
from maille.geometry import border_vertices
from maille.simplify import auto_simplifier, resolve_simplifier, simplify_to_target

BACKENDS = ("meshopt", "greedy")


def backend(name: str) -> maille.Simplifier:
    """One of the two backends, skipping when its dependency is absent."""
    if name == "meshopt":
        pytest.importorskip("meshoptimizer")
        return maille.MeshoptSimplifier()
    return maille.GreedyEdgeCollapse()


def a_sphere(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A closed surface with no boundary at all."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(radius=20.0, subdivisions=subdivisions).apply_translation([50.0, 50.0, 50.0])
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def a_cut_sheet() -> tuple[np.ndarray, np.ndarray]:
    """A surface with an open boundary -- the shape a clipped fragment actually has."""
    trimesh = pytest.importorskip("trimesh")
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


def test_meshopt_never_invents_a_vertex_position():
    """The property that makes ``LOCKED`` provable rather than intended.

    meshopt returns an index buffer into the original vertex array, so every surviving vertex
    is bit-identical to its input. Nothing can drift, including the exact placements
    :func:`maille.snap_boundary` computed.
    """
    pytest.importorskip("meshoptimizer")
    vertices, faces = a_sphere()

    result = maille.MeshoptSimplifier().simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
    )

    for position in result.vertices:
        assert (vertices == position).all(axis=1).any(), "a vertex position was invented"


def test_meshopt_reports_a_far_tighter_error_than_the_fallback():
    """The reason `error_budget` becomes usable at the scale its name implies.

    The fallback can only report how far it moved a vertex, which on a collapsing object is
    about that object's own radius. meshopt reports an actual geometric deviation.
    """
    pytest.importorskip("meshoptimizer")
    vertices, faces = a_sphere()
    fixed = np.zeros(len(vertices), dtype=bool)
    target = len(faces) // 4

    meshopt = maille.MeshoptSimplifier().simplify(vertices, faces, fixed=fixed, target_faces=target)
    greedy = maille.GreedyEdgeCollapse().simplify(vertices, faces, fixed=fixed, target_faces=target)

    assert meshopt.error < greedy.error
    assert meshopt.error < 2.0, "a radius-20 sphere at a quarter of its faces should not stray far"


def test_meshopt_falls_back_when_it_cannot_keep_the_boundary():
    """The verification is what makes ``LOCK_BORDER`` safe rather than assumed.

    Forced here by lying about the boundary: a verifier that considers every vertex part of the
    cut curve cannot be satisfied by any reduction, so the fallback must take over rather than
    a crack being shipped.
    """
    pytest.importorskip("meshoptimizer")
    vertices, faces = a_sphere()

    class EverythingIsBoundary(maille.MeshoptSimplifier):
        @staticmethod
        def _boundary_survived(source_faces: np.ndarray, kept: np.ndarray) -> bool:
            return False

    result = EverythingIsBoundary().simplify(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
    )

    assert result.backend == maille.GreedyEdgeCollapse().name, "the fallback did not take over"


# --------------------------------------------------------------------------- #
# Choosing and relaxing
# --------------------------------------------------------------------------- #


def test_the_default_backend_follows_what_is_installed(monkeypatch: pytest.MonkeyPatch):
    """meshopt where it is available, pure numpy where it is not."""
    pytest.importorskip("meshoptimizer")
    assert isinstance(auto_simplifier(), maille.MeshoptSimplifier)

    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "meshoptimizer":
            raise ImportError("pretending it is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert isinstance(auto_simplifier(), maille.GreedyEdgeCollapse)


def test_something_that_is_not_a_simplifier_is_refused():
    """Naming the two built-ins beats an AttributeError inside the build."""
    with pytest.raises(TypeError, match="does not"):
        resolve_simplifier(object())

    assert resolve_simplifier(None) is not None
    chosen = maille.GreedyEdgeCollapse()
    assert resolve_simplifier(chosen) is chosen


@pytest.mark.parametrize("name", BACKENDS)
def test_a_target_that_destroys_the_surface_is_relaxed_until_something_survives(name: str):
    """A level is a standalone representation, so a piece may never be simplified to nothing."""
    trimesh = pytest.importorskip("trimesh")
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
    pytest.importorskip("meshoptimizer")
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
        assert cells["meshopt"] == cells["greedy"], f"level {level} was partitioned differently"


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
