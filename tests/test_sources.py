"""What may be handed in as an object, and what is refused before any work is spent on it."""

from __future__ import annotations

import numpy as np
import pytest

import maille
from maille.sources import Mesh, coerce_mesh, coerce_objects


def test_a_vertices_faces_pair_is_accepted():
    """The shape a caller who already has arrays should be able to pass."""
    mesh = coerce_mesh((np.zeros((4, 3)), np.array([[0, 1, 2], [1, 2, 3]])))

    assert isinstance(mesh, Mesh)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.dtype == np.int64


def test_a_trimesh_is_accepted():
    """The shape a mesh extractor usually hands you."""
    trimesh = pytest.importorskip("trimesh")

    mesh = coerce_mesh(trimesh.creation.box(extents=[2.0, 3.0, 4.0]))

    assert mesh.vertices.shape[1] == 3
    assert len(mesh.faces) == 12


def test_a_maille_mesh_passes_through_unchanged():
    """Coercing twice must not copy or convert twice."""
    original = Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))

    assert coerce_mesh(original) is original


def test_bounds_match_trimeshs_attribute():
    """`bounds` is read the same way off either input shape, so it has to mean the same thing."""
    mesh = Mesh(vertices=np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]), faces=np.zeros((0, 3), dtype=np.int64))
    low, high = mesh.bounds

    assert low.tolist() == [1.0, 2.0, 3.0]
    assert high.tolist() == [4.0, 6.0, 8.0]


def test_an_empty_mesh_has_bounds_rather_than_raising():
    """A degenerate object should not take out the whole build with an axis-0 reduction."""
    low, high = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)).bounds

    assert low.tolist() == [0.0, 0.0, 0.0]
    assert high.tolist() == [0.0, 0.0, 0.0]


def test_something_that_is_not_a_mesh_at_all_is_refused():
    """Naming the accepted shapes beats an AttributeError from inside the clipper."""
    with pytest.raises(TypeError, match="trimesh.Trimesh, a maille.Mesh, or a"):
        coerce_mesh(42)


@pytest.mark.parametrize(
    ("vertices", "faces", "message"),
    [
        (np.zeros((4, 2)), np.array([[0, 1, 2]]), "an \\(n, 3\\) array"),
        (np.zeros((4, 3)), np.array([[0, 1]]), "an \\(m, 3\\) array"),
        (np.zeros((4, 3)), np.array([[0, 1, 9]]), "do not belong to"),
        (np.zeros((4, 3)), np.array([[0, -1, 2]]), "do not belong to"),
    ],
)
def test_malformed_arrays_are_refused_with_the_reason(vertices, faces, message):
    """Each of these produces geometry that is wrong rather than absent, so it fails early."""
    with pytest.raises(ValueError, match=message):
        coerce_mesh((vertices, faces))


def test_the_failing_object_is_named():
    """In a mapping of thousands, which one failed is the whole message."""
    with pytest.raises(ValueError, match="Object 4711"):
        coerce_objects({4711: (np.zeros((2, 3)), np.array([[0, 1, 5]]))})


def test_object_ids_are_normalised_to_ints():
    """Ids come from a label volume and may arrive as numpy integers."""
    coerced = coerce_objects({np.int64(3): (np.zeros((3, 3)), np.array([[0, 1, 2]]))})

    assert list(coerced) == [3]
    assert isinstance(next(iter(coerced)), int)


def test_an_empty_collection_is_refused():
    """There is nothing to choose a cell size from and nothing to write."""
    with pytest.raises(maille.FormatError, match="at least one object"):
        maille.build_collection({}, axes=("z", "y", "x"), cell_size=(64, 64, 64))


def test_choosing_a_cell_size_for_nothing_is_refused():
    """The heuristic reads the objects, so it cannot answer without any."""
    with pytest.raises(maille.FormatError, match="empty collection"):
        maille.choose_cell_size({})


def test_an_unknown_codec_is_refused_before_the_expensive_work():
    """Cutting a collection and then failing on a typo is a waste of minutes."""
    with pytest.raises(maille.FormatError, match="the format defines"):
        maille.build_collection(
            {1: (np.zeros((3, 3)), np.array([[0, 1, 2]]))},
            axes=("z", "y", "x"),
            cell_size=(64, 64, 64),
            codec="DRACO",
        )


def test_a_level_count_below_one_is_refused():
    """An octree has at least one level."""
    with pytest.raises(maille.FormatError, match="at least one level"):
        maille.build_collection(
            {1: (np.zeros((3, 3)), np.array([[0, 1, 2]]))},
            axes=("z", "y", "x"),
            cell_size=(64, 64, 64),
            levels=0,
        )
