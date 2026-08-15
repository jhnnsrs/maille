"""How far a coarse surface strayed, and whether the boundary really did stay put.

Neither backend is handed an error metric by its library, so the number a planner spends its
budget against is computed here -- and the promise the format makes about cell faces is checked
here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from maille.geometry import border_vertices


def measure_deviation(before: npt.NDArray[np.float64], after: npt.NDArray[np.float64]) -> float:
    """How far the simplified surface strays from the one it came from, as an upper bound.

    The furthest any original vertex sits from the *nearest surviving vertex*. Points on the
    simplified surface lie between its vertices, so the true distance to that surface is never
    larger -- this over-reports rather than under-reports, which is the safe direction for a
    number that decides whether a viewer fetches more detail.

    A tighter figure would measure point-to-triangle rather than point-to-vertex, at several
    times the cost; this one is a KD-tree query over a mesh that is usually a few thousand
    vertices, and it runs once per cell per level.
    """
    if not len(before) or not len(after):
        return 0.0
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    return float(cKDTree(np.asarray(after, dtype=np.float64)).query(np.asarray(before, dtype=np.float64))[0].max())


def boundary_held(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64], kept: npt.NDArray[np.float64]
) -> bool:
    """Whether every vertex on the open boundary survived at exactly its old position.

    Position rather than identity, because a backend that moves vertices hands back a fresh
    array with no correspondence to the old one. Exact rather than approximate, because the
    neighbouring cell quantizes the same coordinate against a different box: anything short of
    equality is a gap the format promises is not there.
    """
    boundary = border_vertices(faces)
    if not len(boundary):
        return True
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    distances = cKDTree(np.asarray(kept, dtype=np.float64)).query(np.asarray(vertices[boundary], dtype=np.float64))[0]
    return bool((distances == 0.0).all())


def pinned_held(
    vertices: npt.NDArray[np.float64], kept: npt.NDArray[np.float64], fixed: npt.NDArray[np.bool_]
) -> bool:
    """Whether every vertex the *format* pins survived at exactly its old position.

    :func:`boundary_held` asks the same question of the topological boundary, which is the thing
    a library flag like ``preserve_border`` can express. That boundary is *usually* a superset
    of the pinned set -- a fragment cut at a cell plane has that plane as an open boundary --
    but it is not always one, and the exception is not exotic: **a closed surface merely tangent
    to a cell face** has a vertex on that face and no open boundary anywhere, so nothing pins it
    and a quadric collapse is free to move it wherever the error says.

    Moving it is two failures at once. It is a crack, because the neighbouring cell quantizes
    that same coordinate against a different box and the format promises the two agree. And when
    it moves *outward* it is worse than a crack: the vertex leaves the cell box, per-cell
    quantization cannot represent a coordinate outside its own cell, and the whole build fails
    at :func:`maille.encode_positions` -- correctly, but a long way from the cause.

    So the ``fixed`` mask is checked directly rather than trusted to a proxy. Exact equality for
    the same reason :func:`boundary_held` uses it: anything short of it is the gap.
    """
    fixed = np.asarray(fixed, dtype=bool)
    if fixed.shape != (len(vertices),) or not fixed.any():
        return True
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]

    distances = cKDTree(np.asarray(kept, dtype=np.float64)).query(np.asarray(vertices[fixed], dtype=np.float64))[0]
    return bool((distances == 0.0).all())


__all__ = ["boundary_held", "measure_deviation", "pinned_held"]
