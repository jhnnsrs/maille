"""Which backend a name selects, and the retry policy every backend is driven through.

``simplification`` is a *value*, the way ``codec`` is: a name from a vocabulary the format
defines, resolved to an implementation here. ``QUADRIC`` is the default. Adding a backend is a
module beside this one and an entry in the table -- and passing an instance stays available for
the cases where the defaults need adjusting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from maille.errors import FormatError
from maille.simplifiers.greedy import SIMPLIFICATION_GREEDY, GreedyEdgeCollapse
from maille.simplifiers.protocol import Simplified, Simplifier
from maille.simplifiers.quadric import SIMPLIFICATION_QUADRIC, QuadricSimplifier

#: The backend used when none is named: quadric-optimal shapes with the boundary pinned.
SIMPLIFICATION_DEFAULT = SIMPLIFICATION_QUADRIC

#: Every backend that ships, keyed by the ``simplification`` value that selects it.
_SIMPLIFIERS: dict[str, type[Simplifier]] = {
    SIMPLIFICATION_QUADRIC: QuadricSimplifier,
    SIMPLIFICATION_GREEDY: GreedyEdgeCollapse,
}


def simplifier_for(simplification: str) -> Simplifier:
    """The backend a ``simplification`` value names, at its defaults.

    A fresh instance rather than a shared one: a backend is configuration as much as code --
    ``aggression``, ``placement``, a fallback -- and handing out one object would make a caller's
    adjustment everyone else's.
    """
    try:
        factory = _SIMPLIFIERS[simplification]
    except KeyError:
        raise FormatError(
            f"`simplification` is {simplification!r}; maille ships {', '.join(sorted(_SIMPLIFIERS))}."
        ) from None
    return factory()


def resolve_simplifier(simplifier: Any) -> Simplifier:  # noqa: ANN401
    """Accept a backend, the name of one, or ``None`` for the default.

    Three shapes, in the order a caller reaches for them: nothing at all, a name out of the
    vocabulary, or an instance they have configured themselves.
    """
    if simplifier is None:
        return simplifier_for(SIMPLIFICATION_DEFAULT)
    if isinstance(simplifier, str):
        return simplifier_for(simplifier)
    if not hasattr(simplifier, "simplify"):
        raise TypeError(
            f"A simplifier is one of {', '.join(sorted(_SIMPLIFIERS))}, or an object providing a "
            f"`simplify(vertices, faces, *, fixed, target_faces)` method; {type(simplifier).__name__} is "
            f"neither. Try maille.QuadricSimplifier() or maille.GreedyEdgeCollapse()."
        )
    return simplifier


def simplify_to_target(
    simplifier: Simplifier,
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    *,
    fixed: npt.NDArray[np.bool_],
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


__all__ = [
    "SIMPLIFICATION_DEFAULT",
    "resolve_simplifier",
    "simplifier_for",
    "simplify_to_target",
]
