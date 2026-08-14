"""``simplification: GREEDY`` -- shortest-edge collapse in pure numpy.

Worth choosing where a heavily pinned boundary keeps the quadric collapse from reaching the face
budget: this one will collapse an edge *onto* a locked vertex, where ``preserve_border`` leaves
the locked region alone entirely. Lower quality, and its error estimate is how far it moved a
vertex, which on a collapsing object is about that object's radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maille.geometry import decimate_fixed
from maille.simplifiers.protocol import Simplified

#: The ``simplification`` value that selects this backend.
SIMPLIFICATION_GREEDY = "GREEDY"


@dataclass(frozen=True)
class GreedyEdgeCollapse:
    """Shortest-edge collapse in pure numpy, pinning exactly the mask it is handed.

    ``check_interval`` is how many collapses to make between face counts (``None`` picks 16
    above 256 faces and 1 below, where overshoot would consume the mesh). ``placement`` is
    ``"midpoint"`` or ``"onto_fixed"``; the latter never moves a vertex at all, trading shape
    quality for every surviving vertex keeping its exact input position.
    """

    check_interval: int | None = None
    placement: str = "midpoint"
    name: str = SIMPLIFICATION_GREEDY
    #: This one is held by exactly the mask it is handed.
    uses_fixed_mask: bool = True

    def simplify(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        fixed: np.ndarray,
        target_faces: int,
    ) -> Simplified:
        """Collapse edges shortest-first until the face budget is met."""
        kept_vertices, kept_faces, moved = decimate_fixed(
            vertices,
            faces,
            fixed=fixed,
            target_faces=target_faces,
            check_interval=self.check_interval,
            placement=self.placement,
        )
        return Simplified(
            vertices=kept_vertices,
            faces=kept_faces,
            error=float(moved),
            reached_target=len(kept_faces) <= target_faces,
            backend=self.name,
        )


__all__ = ["SIMPLIFICATION_GREEDY", "GreedyEdgeCollapse"]
