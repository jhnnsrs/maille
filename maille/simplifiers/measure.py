"""How far a coarse surface strayed, and whether the boundary really did stay put.

Neither backend is handed an error metric by its library, so the number a planner spends its
budget against is computed here -- and the promise the format makes about cell faces is checked
here rather than assumed.
"""

from __future__ import annotations

import numpy as np

from maille.geometry import border_vertices


def measure_deviation(before: np.ndarray, after: np.ndarray) -> float:
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


def boundary_held(vertices: np.ndarray, faces: np.ndarray, kept: np.ndarray) -> bool:
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


__all__ = ["boundary_held", "measure_deviation"]
