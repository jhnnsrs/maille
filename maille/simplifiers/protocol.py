"""What a simplification backend is: one call, and what it hands back.

Structural, like the store and codec protocols -- a backend satisfies this by having the
method, not by inheriting anything -- so a caller can pass their own without importing a base
class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Simplified:
    """What a simplifier returns: the coarser surface, and how far it strayed making it."""

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]
    #: An upper bound on how far this surface deviates from the one handed in, in voxels. It
    #: becomes the decimation half of the cell's ``lod_error``, which is what a planner spends
    #: its budget against -- so a backend that cannot measure honestly should over-report.
    error: float
    #: Whether the face budget was actually met. A backend may stop early on its own terms.
    reached_target: bool
    #: Which backend produced this, for diagnostics and for the builder's warnings.
    backend: str = ""


@runtime_checkable
class Simplifier(Protocol):
    """The one operation a simplification backend provides.

    ``fixed`` is a boolean mask of vertices that must not move, and it is a *conservative*
    superset of the boundary that genuinely matters -- see
    :func:`maille.geometry.border_vertices`. A backend that never moves any vertex satisfies it
    trivially and need only ensure the boundary vertices survive.
    """

    @property
    def name(self) -> str:
        """What to call this backend in a manifest, a warning or a diagnostic."""
        ...

    @property
    def uses_fixed_mask(self) -> bool:
        """Whether ``fixed`` is what constrains this backend, or something narrower.

        The builder reports *why* a face budget was missed, and a pinned boundary is the usual
        reason -- but only a backend that actually reads ``fixed`` is constrained by the number
        of vertices in it. One that locks the topological border instead is held by a strict
        subset, so quoting the ``fixed`` count at it would explain the miss with a figure that
        does not describe the constraint.
        """
        ...

    def simplify(
        self,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        *,
        fixed: npt.NDArray[np.bool_],
        target_faces: int,
    ) -> Simplified:
        """Reduce ``faces`` toward ``target_faces`` without moving a fixed vertex."""
        ...


__all__ = ["Simplified", "Simplifier"]
