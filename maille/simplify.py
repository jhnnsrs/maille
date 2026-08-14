"""How a coarse level is made, and how to choose or replace the algorithm that makes it.

A coarser level is the same surfaces with fewer triangles, and the writer's whole obligation is
two-fold: hit a face budget (``decimation``), and leave the cut boundary exactly where it was
(``boundary: LOCKED``), so a fine cell drawn beside a coarse one meets it without a crack.

Everything else about *how* is a choice, so it is a strategy here rather than a hardcoded loop.
Two backends ship:

:class:`MeshoptSimplifier`
    The default when ``meshoptimizer`` is installed -- which it already is wherever the
    ``MESHOPT`` blob codec is used. Quadric-error simplification, and **it never invents a
    vertex position**: it returns an index buffer into the original vertex array, so every
    surviving vertex keeps its input coordinates bit-for-bit. That is what makes ``LOCKED``
    provable here rather than merely intended, and it leaves :func:`maille.snap_boundary`'s
    exact placements untouched. It also reports a real geometric deviation, which is a far
    tighter ``lod_error`` than the alternative below can offer.

:class:`GreedyEdgeCollapse`
    Shortest-edge collapse in pure numpy, for an install that has trimesh but not
    meshoptimizer. (Building a collection needs trimesh either way -- it is what cuts a mesh at
    the cell planes -- so this is the *simplifier* falling back, never the whole writer.) Lower
    quality, and its only honest error estimate is how far it moved a vertex, which when a small
    object collapses toward a point is about that object's own radius. Budgets against it are
    therefore coarse.

Two meshopt features are deliberately *not* exposed. ``simplify_sloppy`` takes no options at
all, so it cannot lock a border and cannot honour ``LOCKED`` -- and this spec version refuses
``OPEN``, so there is no configuration in which it would be legal. ``SIMPLIFY_PRUNE`` drops
small disconnected components, which is exactly the "an object silently disappears when you
zoom out" failure the builder works to prevent.

One meshopt feature is unavailable rather than declined: ``meshopt_simplifyWithAttributes``
takes an arbitrary ``vertex_lock`` array, which would let a caller lock any vertex set, but the
Python binding does not declare ``argtypes`` for it and the call raises ``ctypes.ArgumentError``
(checked against 0.2.x). ``SIMPLIFY_LOCK_BORDER`` is used instead, and it is arguably the more
precise condition anyway -- see :func:`maille.geometry.border_vertices`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maille.geometry import border_vertices, decimate_fixed

#: A ceiling high enough that the face target is always what stops meshopt, never the error.
#: It is passed as a C float, so it stays well inside that range.
_NO_ERROR_CEILING = 1.0e9


@dataclass(frozen=True)
class Simplified:
    """What a simplifier returns: the coarser surface, and how far it strayed making it."""

    vertices: np.ndarray
    faces: np.ndarray
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
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        fixed: np.ndarray,
        target_faces: int,
    ) -> Simplified:
        """Reduce ``faces`` toward ``target_faces`` without moving a fixed vertex."""
        ...


@dataclass(frozen=True)
class GreedyEdgeCollapse:
    """Shortest-edge collapse in pure numpy. The fallback when meshoptimizer is absent.

    Chosen automatically by :func:`auto_simplifier` in that case, and worth choosing explicitly
    when a heavily pinned boundary keeps meshopt from reaching the face budget: this one will
    collapse an edge *onto* a locked vertex, where meshopt leaves the locked region alone.

    ``check_interval`` is how many collapses to make between face counts (``None`` picks 16
    above 256 faces and 1 below, where overshoot would consume the mesh). ``placement`` is
    ``"midpoint"`` or ``"onto_fixed"``; the latter never moves a vertex, which trades shape
    quality for the same exact-position property meshopt has.
    """

    check_interval: int | None = None
    placement: str = "midpoint"
    name: str = "greedy-edge-collapse"
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


@dataclass(frozen=True)
class MeshoptSimplifier:
    """Quadric-error simplification via meshoptimizer. The default where it is installed.

    ``lock_border`` keeps the open boundary -- the cut curve -- out of the collapse, which is
    how ``LOCKED`` is honoured. ``verify_border`` checks afterwards that no boundary vertex was
    dropped and falls back to ``fallback`` for that piece if one was: since meshopt never moves
    a vertex, "no boundary vertex removed" is the whole of the promise, and checking it costs
    one array intersection.
    """

    lock_border: bool = True
    verify_border: bool = True
    #: The error ceiling handed to meshopt. Left high so the face budget is what stops it;
    #: lower it to stop early once a deviation is reached instead.
    target_error: float = _NO_ERROR_CEILING
    fallback: Simplifier = field(default_factory=GreedyEdgeCollapse)
    name: str = "meshopt"
    #: meshopt locks the *topological border*, not the mask it is handed -- a strict subset of
    #: it, and the more precise statement of what may not move. See `border_vertices`.
    uses_fixed_mask: bool = False

    def simplify(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        fixed: np.ndarray,
        target_faces: int,
    ) -> Simplified:
        """Simplify with meshopt, verifying the boundary survived."""
        from maille.codec import require_meshoptimizer

        module = require_meshoptimizer()

        source_vertices = np.asarray(vertices, dtype=np.float64)
        source_faces = np.asarray(faces, dtype=np.int64)
        if len(source_faces) <= target_faces:
            return Simplified(source_vertices, source_faces, 0.0, True, self.name)

        options = 0
        if self.lock_border:
            options |= module.SIMPLIFY_LOCK_BORDER
        # Report the error in voxels rather than as a fraction of the mesh's extent, so it is
        # directly comparable with a cell's quantization step and with an `error_budget`.
        options |= module.SIMPLIFY_ERROR_ABSOLUTE

        indices = np.ascontiguousarray(source_faces.reshape(-1), dtype=np.uint32)
        positions = np.ascontiguousarray(source_vertices, dtype=np.float32)
        destination = np.zeros(len(indices), dtype=np.uint32)
        reported = np.zeros(1, dtype=np.float32)

        count = module.simplify(
            destination,
            indices,
            positions,
            target_index_count=int(target_faces) * 3,
            target_error=float(self.target_error),
            options=options,
            result_error=reported,
        )

        kept = destination[:count].reshape(-1, 3).astype(np.int64)
        if not len(kept):
            return Simplified(source_vertices[:0], kept, 0.0, False, self.name)

        if self.verify_border and not self._boundary_survived(source_faces, kept):
            # Nothing here moved a vertex, so the only way to break LOCKED is to drop one off
            # the cut curve. Hand the piece to the fallback rather than shipping a crack.
            return self.fallback.simplify(
                source_vertices, source_faces, fixed=fixed, target_faces=target_faces
            )

        # meshopt indexes the *original* vertex array, so compacting is a subset -- and the
        # surviving positions are the float64 originals, never the float32 copy sent to C.
        used = np.unique(kept)
        remap = np.full(len(source_vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return Simplified(
            vertices=source_vertices[used],
            faces=remap[kept],
            error=float(reported[0]),
            reached_target=len(kept) <= target_faces,
            backend=self.name,
        )

    @staticmethod
    def _boundary_survived(source_faces: np.ndarray, kept: np.ndarray) -> bool:
        """Whether every vertex on the input's open boundary is still referenced."""
        boundary = border_vertices(source_faces)
        if not len(boundary):
            return True
        return bool(np.isin(boundary, np.unique(kept)).all())


def auto_simplifier() -> Simplifier:
    """The best backend available: meshopt where it is installed, greedy where it is not.

    Chosen per build rather than imported at module load, so installing the extra changes the
    behaviour of the next call rather than requiring a restart.
    """
    try:
        import meshoptimizer  # type: ignore  # noqa: F401
    except ImportError:
        return GreedyEdgeCollapse()
    return MeshoptSimplifier()


def simplify_to_target(
    simplifier: Simplifier,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    fixed: np.ndarray,
    target_faces: int,
) -> tuple[Simplified, bool]:
    """Simplify to ``target_faces``, relaxing the target if that target destroys the surface.

    Some surfaces -- a box, most clearly -- cannot be taken to an aggressive target by a
    simplifier without validity checks: every edge collapses into its neighbour until no
    non-degenerate triangle is left. Two responses are wrong and one is right.

    Dropping the piece is wrong: **a level is a standalone representation of the whole
    collection**, so an object missing from level 2 is not a coarser object, it is an object
    that disappears when a viewer zooms out -- with nothing raised anywhere, because a level
    that lost a row looks exactly like a level that never had one.

    Keeping the piece *undecimated* is also wrong, and worse than it looks: it makes the coarse
    level larger than the fine one it summarises, which is a coarse level that costs a fetch and
    saves nothing.

    So the target is doubled until something survives -- the tightest reduction this backend can
    actually reach on this surface. Only if nothing survives at any target is the original kept,
    which is the honest last resort.

    Returns the result and whether the target had to be relaxed.
    """
    attempt = max(1, int(target_faces))
    while attempt < len(faces):
        result = simplifier.simplify(vertices, faces, fixed=fixed, target_faces=attempt)
        if len(result.faces):
            return result, attempt != target_faces
        attempt *= 2

    return Simplified(vertices, faces, 0.0, False, getattr(simplifier, "name", "")), True


def resolve_simplifier(simplifier: Any) -> Simplifier:  # noqa: ANN401
    """Accept a backend, or ``None`` for whichever one is available."""
    if simplifier is None:
        return auto_simplifier()
    if not hasattr(simplifier, "simplify"):
        raise TypeError(
            f"A simplifier provides a `simplify(vertices, faces, *, fixed, target_faces)` method; "
            f"{type(simplifier).__name__} does not. Try maille.MeshoptSimplifier() or "
            f"maille.GreedyEdgeCollapse()."
        )
    return simplifier


__all__ = [
    "GreedyEdgeCollapse",
    "MeshoptSimplifier",
    "Simplified",
    "Simplifier",
    "auto_simplifier",
    "resolve_simplifier",
    "simplify_to_target",
]
