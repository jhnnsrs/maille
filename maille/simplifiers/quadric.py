"""``simplification: QUADRIC`` -- quadric edge collapse via fast-simplification. The default.

fast-simplification is a dependency of maille rather than an extra -- making coarse levels is
what maille is for -- so it is imported plainly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fast_simplification
import numpy as np
import numpy.typing as npt

from maille.simplifiers.greedy import GreedyEdgeCollapse
from maille.simplifiers.measure import boundary_held, measure_deviation
from maille.simplifiers.protocol import Simplified, Simplifier

#: The ``simplification`` value that selects this backend, and the format's default.
SIMPLIFICATION_QUADRIC = "QUADRIC"


@dataclass(frozen=True)
class QuadricSimplifier:
    """Quadric edge collapse via fast-simplification -- Sven Forstmann's algorithm.

    ``preserve_border`` is what honours ``boundary: LOCKED``: it holds every vertex on the open
    boundary at exactly its input position. Interior vertices do move -- to wherever the quadric
    puts them, which is the whole point -- so unlike a subset-only simplifier this one improves
    the shape rather than just choosing which vertices to keep.

    ``aggression`` is the library's ``agg``: how hard it pushes per iteration. Higher reaches a
    target in fewer passes and strays further doing it; 7 is the library's default.

    ``verify_border`` checks afterwards that every boundary vertex is still present at its exact
    position, and falls back to ``fallback`` for that piece if one is not. The check costs one
    lookup per boundary vertex and is what turns a library flag into a guarantee.

    **What this costs, stated plainly.** ``preserve_border`` is all-or-nothing: it pins the
    whole topological boundary, and after the children of a coarse cell are welded that boundary
    still contains the level-0 seams *interior* to the cell -- T-junctions where two fragments
    triangulated the same cut plane differently. Those vertices do not need pinning at this
    level, and nothing across a face depends on them, but the library has no way to say so. On a
    heavily cut object they can be most of the boundary, and the collapse then falls well short
    of its face budget: measured on a 300x170x90 box cut into 34 fragments, 74 of 176 welded
    vertices were such seams and the level-2 reduction stopped at 74% where
    :class:`~maille.simplifiers.greedy.GreedyEdgeCollapse` -- which pins only *this* level's cell
    planes -- reached 9%.

    So the trade is a real one, not a formality: this backend gives better shapes and a much
    tighter error at the cost of sometimes missing the budget, and the builder warns when it
    does. Pass ``simplifier=GreedyEdgeCollapse()`` where the budget matters more than the shape.
    """

    aggression: float = 7.0
    preserve_border: bool = True
    verify_border: bool = True
    fallback: Simplifier = field(default_factory=GreedyEdgeCollapse)
    name: str = SIMPLIFICATION_QUADRIC
    #: Held by the topological boundary rather than by the mask it is handed -- a strict subset
    #: of it, and the more precise statement of what may not move. See `border_vertices`.
    uses_fixed_mask: bool = False

    def simplify(
        self,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        *,
        fixed: npt.NDArray[np.bool_],
        target_faces: int,
    ) -> Simplified:
        """Collapse to the face budget, then check the boundary really did stay put."""
        source_vertices = np.asarray(vertices, dtype=np.float64)
        source_faces = np.asarray(faces, dtype=np.int64)
        if len(source_faces) <= target_faces:
            return Simplified(source_vertices, source_faces, 0.0, True, self.name)

        # Indexed rather than unpacked: the library returns a third array when asked to report
        # its collapses, which it is not here.
        collapsed = fast_simplification.simplify(
            np.ascontiguousarray(source_vertices, dtype=np.float64),
            np.ascontiguousarray(source_faces, dtype=np.int32),
            target_count=max(1, int(target_faces)),
            agg=float(self.aggression),
            preserve_border=self.preserve_border,
        )
        kept_vertices = np.asarray(collapsed[0], dtype=np.float64)
        kept_faces = np.asarray(collapsed[1], dtype=np.int64)

        if not len(kept_faces):
            return Simplified(source_vertices[:0], kept_faces, 0.0, False, self.name)

        if self.verify_border and not boundary_held(source_vertices, source_faces, kept_vertices):
            # A boundary vertex moved or vanished, which is a crack between levels rather than a
            # quality regression. Hand the piece to the fallback rather than shipping it.
            return self.fallback.simplify(source_vertices, source_faces, fixed=fixed, target_faces=target_faces)

        return Simplified(
            vertices=kept_vertices,
            faces=kept_faces,
            error=measure_deviation(source_vertices, kept_vertices),
            reached_target=len(kept_faces) <= target_faces,
            backend=self.name,
        )


__all__ = ["SIMPLIFICATION_QUADRIC", "QuadricSimplifier"]
