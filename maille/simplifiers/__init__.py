"""How a coarse level is made, and how to choose or replace the algorithm that makes it.

A coarser level is the same surfaces with fewer triangles, and the writer's whole obligation is
two-fold: hit a face budget (``decimation``), and leave the cut boundary exactly where it was
(``boundary: LOCKED``), so a fine cell drawn beside a coarse one meets it without a crack.

Everything else about *how* is a choice, so the backend is named the way a codec is -- a value
out of a small vocabulary, resolved to an implementation -- rather than a hardcoded loop::

    maille.build_collection(objects, cell_size=...)                      # QUADRIC, the default
    maille.build_collection(objects, cell_size=..., simplifier="GREEDY")
    maille.build_collection(objects, cell_size=..., simplifier=maille.GreedyEdgeCollapse(placement="onto_fixed"))

Two backends ship:

``QUADRIC`` (:class:`~maille.simplifiers.quadric.QuadricSimplifier`)
    The default, backed by `fast-simplification
    <https://github.com/pyvista/fast-simplification>`_ -- Sven Forstmann's quadric collapse.
    It is run with ``preserve_border=True``, which pins every vertex on the open boundary at
    **exactly** its input position while leaving interior vertices free to move to the position
    the quadric says is best. That split is precisely what the format wants: the cut curve is
    the only thing a neighbouring cell shares, so it is the only thing that must not move, and
    holding the interior still as well would cost shape quality for nothing.

``GREEDY`` (:class:`~maille.simplifiers.greedy.GreedyEdgeCollapse`)
    Shortest-edge collapse in pure numpy, for the cases where a heavily pinned boundary stops
    the quadric collapse reaching a budget: this one will collapse an edge *onto* a locked
    vertex. Lower quality, and its error estimate is how far it moved a vertex, which on a
    collapsing object is about that object's radius.

The package is split the way the stores and codecs are: :mod:`~maille.simplifiers.protocol`
states what a backend is, :mod:`~maille.simplifiers.quadric` and
:mod:`~maille.simplifiers.greedy` are the two implementations,
:mod:`~maille.simplifiers.measure` is the error bound neither library provides, and
:mod:`~maille.simplifiers.registry` is the name-to-backend table and the retry policy the
builder drives them through.
"""

from maille.simplifiers.greedy import SIMPLIFICATION_GREEDY, GreedyEdgeCollapse
from maille.simplifiers.measure import boundary_held, measure_deviation, pinned_held
from maille.simplifiers.protocol import Simplified, Simplifier
from maille.simplifiers.quadric import SIMPLIFICATION_QUADRIC, QuadricSimplifier
from maille.simplifiers.registry import (
    SIMPLIFICATION_DEFAULT,
    resolve_simplifier,
    simplifier_for,
    simplify_to_target,
)

__all__ = [
    "SIMPLIFICATION_DEFAULT",
    "SIMPLIFICATION_GREEDY",
    "SIMPLIFICATION_QUADRIC",
    "GreedyEdgeCollapse",
    "QuadricSimplifier",
    "Simplified",
    "Simplifier",
    "boundary_held",
    "measure_deviation",
    "pinned_held",
    "resolve_simplifier",
    "simplifier_for",
    "simplify_to_target",
]
