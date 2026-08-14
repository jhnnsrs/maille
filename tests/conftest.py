"""Fixtures shared across the suite.

**Every fixture here is anisotropic, and that is deliberate.** A cubic cell size and a
spherical object make a transposed writer indistinguishable from a correct one: swap two axes
and a symmetric round trip still closes. So the cell is ``(128, 128, 64)`` and the objects are
boxes with three different extents at off-centre positions -- a fixture that disagrees with
itself under any permutation of the axes.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import maille

#: Deliberately asymmetric: a symmetric one passes a reversed implementation.
CELL_SIZE = (128, 128, 64)

#: The axis order the fixtures declare. maille never defaults this, so every test states it.
AXES = ("z", "y", "x")

LEVELS = 3


def require_trimesh() -> Any:  # noqa: ANN401
    """Skip a test when the mesh extra is not installed."""
    return pytest.importorskip("trimesh")


@pytest.fixture(scope="session")
def objects() -> dict[int, Any]:
    """Two objects that disagree under any permutation of the axes.

    One spans several level-0 cells so there is real cutting to do -- an object entirely
    inside one cell never exercises the boundary at all -- and one is small enough that
    ``QUARTER`` has nothing left to take from it, which is the case that used to make an object
    disappear from the coarse levels.
    """
    trimesh = require_trimesh()
    return {
        # Wider than a cell in x and y, so it is cut and its fragments share faces.
        7: trimesh.creation.box(extents=[300.0, 170.0, 90.0]).apply_translation([260.0, 210.0, 95.0]),
        # Comfortably inside one cell, and small.
        3: trimesh.creation.box(extents=[40.0, 24.0, 16.0]).apply_translation([90.0, 70.0, 40.0]),
        # A sphere, for geometry that is not axis-aligned.
        11: trimesh.creation.icosphere(radius=26.0, subdivisions=2).apply_translation([420.0, 300.0, 150.0]),
    }


@pytest.fixture(scope="session")
def collection(objects: dict[int, Any]) -> maille.MeshCollection:
    """A built collection, shared across the suite because building it is the slow part."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the QUARTER warning is asserted on where it matters
        return maille.build_collection(objects, axes=AXES, cell_size=CELL_SIZE, levels=LEVELS)


@pytest.fixture(scope="session")
def written(collection: maille.MeshCollection) -> maille.MemoryStore:
    """The collection, written into a store."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "collection")
    return store


@pytest.fixture()
def opened(written: maille.MemoryStore) -> maille.Collection:
    """The collection, read back. Fresh per test so the lazy caches are not shared."""
    return maille.open_collection(written, "collection")


def vertices_of(mesh: Any) -> np.ndarray:  # noqa: ANN401
    """The vertices of a trimesh or a maille Mesh, as a float array."""
    return np.asarray(mesh.vertices, dtype=np.float64)
