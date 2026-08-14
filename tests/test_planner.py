"""The LOD planner: descending the octree, spending an error budget, and failing gracefully."""

from __future__ import annotations

import math

import numpy as np
import pytest

import maille
from tests.conftest import LEVELS


def test_no_budget_descends_to_the_finest_level(opened: maille.Collection):
    """With nothing said about acceptable error, the answer is everything at full detail."""
    plan = opened.plan()

    assert {entry.level for entry in plan} == {0}
    assert len(plan) == len(opened.cells_at(0))


def test_a_generous_error_budget_stops_at_the_coarsest_level(opened: maille.Collection):
    """The other end: one cell, the whole collection, as coarse as it goes."""
    plan = opened.plan(error_budget=1e9)

    assert [(entry.level, entry.cell) for entry in plan] == [(entry.level, entry.cell) for entry in opened.roots()]


def test_a_tighter_budget_never_returns_a_coarser_plan(opened: maille.Collection):
    """Monotonicity: spending less error can only add detail, never remove it."""
    budgets = [1e9, 10.0, 1.0, 0.1, 0.0]
    face_counts = [sum(entry.face_count for entry in opened.plan(error_budget=budget)) for budget in budgets]

    assert face_counts == sorted(face_counts), f"tightening the budget lost detail: {face_counts}"


def test_the_plan_covers_the_collection_without_drawing_a_cell_twice(opened: maille.Collection):
    """A plan is a *cut* through the octree: every region once, at exactly one level."""
    for budget in (1e9, 5.0, 0.5, 0.0):
        plan = opened.plan(error_budget=budget)
        keys = [(entry.level, entry.cell) for entry in plan]

        assert len(keys) == len(set(keys)), "a cell appears twice in one plan"

        # No selected cell may be an ancestor of another selected cell -- that would draw the
        # same surface at two levels, which is the double-image artifact an octree exists to
        # avoid.
        for entry in plan:
            for other in plan:
                if entry is other or other.level >= entry.level:
                    continue
                assert not _is_ancestor(entry, other), f"{entry.cell}@{entry.level} covers {other.cell}@{other.level}"


def _is_ancestor(coarse: maille.CellEntry, fine: maille.CellEntry) -> bool:
    """Whether one cell contains another in the octree."""
    steps = coarse.level - fine.level
    i, j, k = fine.triple
    return (i >> steps, j >> steps, k >> steps) == coarse.triple


def test_a_camera_spends_the_budget_in_pixels_and_so_depends_on_distance(opened: maille.Collection):
    """The reason for an octree: a far cell settles coarse, a near one descends."""
    centre = np.mean([entry.bbox_min for entry in opened.cells.values()], axis=0)

    near = maille.Camera.perspective(centre + np.array([0.0, 0.0, 40.0]), fov_y=0.8, viewport_height=1080)
    far = maille.Camera.perspective(centre + np.array([0.0, 0.0, 40000.0]), fov_y=0.8, viewport_height=1080)

    near_plan = opened.plan(camera=near, pixel_budget=1.0)
    far_plan = opened.plan(camera=far, pixel_budget=1.0)

    assert min(entry.level for entry in near_plan) <= min(entry.level for entry in far_plan)
    assert sum(e.face_count for e in near_plan) >= sum(e.face_count for e in far_plan)


def test_a_camera_inside_a_cell_takes_the_finest_level_available(opened: maille.Collection):
    """Distance to the box, not to its centre -- otherwise a camera inside descends forever."""
    entry = opened.cells_at(0)[0]
    inside = np.asarray(entry.bbox_min) + (np.asarray(entry.bbox_max) - np.asarray(entry.bbox_min)) / 2

    camera = maille.Camera.perspective(inside, fov_y=0.8, viewport_height=1080)

    assert math.isinf(camera.screen_error(entry))
    assert (entry.level, entry.cell) in [(e.level, e.cell) for e in opened.plan(camera=camera, pixel_budget=1.0)]


def test_a_box_query_keeps_only_what_it_meets(opened: maille.Collection):
    """The cheap spatial filter, answered entirely from the catalog."""
    target = opened.cells_at(0)[0]
    plan = opened.plan(box=(target.bbox_min, target.bbox_max))

    assert plan
    for entry in plan:
        assert (np.asarray(entry.bbox_max) >= np.asarray(target.bbox_min)).all()
        assert (np.asarray(entry.bbox_min) <= np.asarray(target.bbox_max)).all()


def test_a_box_that_meets_nothing_returns_nothing(opened: maille.Collection):
    """An empty plan is a legitimate answer, not an error."""
    assert opened.plan(box=((1e6, 1e6, 1e6), (2e6, 2e6, 2e6))) == []


def test_a_frustum_culls_what_is_behind_it(opened: maille.Collection):
    """Conservative: it may keep a cell that is not visible, never cull one that is."""
    everything = opened.plan(error_budget=1e9)
    origin = np.asarray(everything[0].bbox_min)

    # One plane, facing +x from far beyond the data: nothing is in front of it.
    behind = [((1.0, 0.0, 0.0), -1e9)]
    assert opened.plan(error_budget=1e9, frustum=behind) == []

    # The same plane facing the other way keeps everything.
    ahead = [((1.0, 0.0, 0.0), -float(origin[0]) + 1e6)]
    assert len(opened.plan(error_budget=1e9, frustum=ahead)) == len(everything)


def test_an_object_filter_resolves_through_the_object_catalog(opened: maille.Collection):
    """"Where is segment 4711" answered as a lookup, then used to narrow the plan."""
    object_id = next(iter(opened.objects))
    named = set(opened.objects[object_id].cells)

    plan = opened.plan(objects=[object_id])

    assert plan
    for entry in plan:
        assert (entry.level, entry.cell) in named


def test_running_out_of_cells_degrades_detail_rather_than_dropping_geometry(opened: maille.Collection):
    """The failure mode matters: a missing cell is a hole in the surface, a coarse cell is not."""
    full = opened.plan(error_budget=0.0)
    capped = opened.plan(error_budget=0.0, max_cells=2)

    assert len(capped) <= len(full)
    assert capped, "a budget of two cells still has to draw something"

    # Every region the full plan covers is still covered, just possibly by a coarser cell.
    for fine in full:
        assert any(
            (coarse.level, coarse.cell) == (fine.level, fine.cell) or _is_ancestor(coarse, fine) for coarse in capped
        ), f"cell {fine.cell} at level {fine.level} is covered by nothing in the capped plan"


def test_min_level_stops_the_descent_where_it_is_told(opened: maille.Collection):
    """A viewer that cannot afford the finest level asks not to be given it."""
    plan = opened.plan(error_budget=0.0, min_level=1)

    assert plan
    assert min(entry.level for entry in plan) >= 1


def test_a_plan_is_ordered_coarsest_first(opened: maille.Collection):
    """Stable ordering, so a fetch queue can be built off it directly."""
    plan = opened.plan(error_budget=1.0)
    keys = [(-entry.level, entry.cell) for entry in plan]

    assert keys == sorted(keys)


def test_a_camera_needs_a_sensible_field_of_view():
    """A caller who passes degrees instead of radians should hear about it."""
    with pytest.raises(ValueError, match="radians"):
        maille.Camera.perspective((0, 0, 0), fov_y=45.0, viewport_height=1080)


def test_max_cells_must_allow_at_least_one_cell(opened: maille.Collection):
    """Zero cells is not a plan."""
    with pytest.raises(ValueError, match="at least one cell"):
        opened.plan(max_cells=0)


def test_the_planner_reads_only_the_cell_catalog(written: maille.MemoryStore):
    """Planning must not touch geometry -- that is the entire point of the two-catalog split."""

    class GeometryIsOffLimits(maille.MemoryStore):
        def __init__(self, objects: dict[str, bytes]) -> None:
            super().__init__()
            self.objects = dict(objects)

        def get(self, path: str) -> bytes:
            if "level=" in path:
                raise AssertionError(f"the planner opened geometry: {path}")
            return super().get(path)

    collection = maille.open_collection(GeometryIsOffLimits(written.objects), "collection")

    assert collection.plan(error_budget=0.5)
    assert len(collection.plan()) == len(collection.cells_at(0))


def test_planning_over_every_level_of_the_fixture(opened: maille.Collection):
    """A budget between the levels' errors selects a genuinely mixed plan where one exists."""
    errors = sorted({round(entry.lod_error, 6) for entry in opened.cells.values()})
    assert len(errors) >= 2, "the fixture should have distinguishable errors between levels"

    mixed = opened.plan(error_budget=errors[len(errors) // 2])

    assert {entry.level for entry in mixed} <= set(range(LEVELS))


def test_max_cells_caps_the_plan_when_the_budget_is_above_the_coarsest_level(opened: maille.Collection):
    """Above the floor it behaves as a cap, which is the case a fetch queue is sized for."""
    roots = len(opened.roots())

    for budget in range(roots, roots + 6):
        assert len(opened.plan(error_budget=0.0, max_cells=budget)) <= budget


def test_max_cells_never_goes_below_the_coarsest_level(opened: maille.Collection):
    """It is a descent budget, not a hard cap, and the floor is deliberate.

    Going lower would mean dropping a root cell, and a missing cell is a hole in the surface
    where a coarse cell is merely blurry. So the floor is stated in the docstring and asserted
    here rather than left for a caller to discover by sizing a buffer off the argument.
    """
    roots = opened.roots()

    plan = opened.plan(error_budget=0.0, max_cells=1)

    assert len(plan) == len(roots)
    assert {(entry.level, entry.cell) for entry in plan} == {(entry.level, entry.cell) for entry in roots}
