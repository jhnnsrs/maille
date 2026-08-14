"""Choosing which cells to draw at which level.

This is the question the whole format exists to answer cheaply, and the reason the cell
catalog is a separate file: a planner reads one small table once and decides the entire frame
from it, without opening a single geometry file.

The descent
-----------
Start at the coarsest level and walk down. At each cell, ask whether *this* level is good
enough; if it is, take the cell and stop; if it is not, descend into the children that hold
geometry -- which ``child_mask`` names, so descending costs no listing and no second query.

"Good enough" is ``lod_error``, which the writer records as an upper bound on how far a vertex
at this level may sit from where it sits at level 0. Two ways to spend it:

- **in voxels** (``error_budget``): take the coarsest level whose error stays under a fixed
  distance. Scale-free, and what you want when there is no camera -- a batch export, a
  fixed-detail fetch, a test.
- **in pixels** (``camera``): project that distance through a perspective camera and take the
  coarsest level whose error is under a pixel budget. Error shrinks with distance, so a far
  cell settles at a coarse level and a near one descends -- the reason for an octree.

A cell whose descent is cut short by ``max_cells`` is still drawn, at the coarsest level it
reached. Running out of budget degrades detail rather than punching a hole in the geometry.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from maille.reader import CellEntry, Collection


@dataclass(frozen=True)
class Camera:
    """Where the viewer is, and how many pixels a voxel of error is worth from there.

    ``focal_pixels`` is the usual pinhole term, ``0.5 * viewport_height / tan(0.5 * fov_y)``:
    an object of size ``s`` at distance ``d`` covers ``s * focal_pixels / d`` pixels. Build one
    with :meth:`perspective` rather than working it out by hand.
    """

    position: tuple[float, float, float]
    focal_pixels: float

    @classmethod
    def perspective(
        cls,
        position: Sequence[float],
        *,
        fov_y: float,
        viewport_height: int,
    ) -> Camera:
        """A camera from a vertical field of view in radians and a viewport height in pixels."""
        if not 0.0 < fov_y < math.pi:
            raise ValueError(f"`fov_y` is a vertical field of view in radians, got {fov_y}.")
        focal = 0.5 * float(viewport_height) / math.tan(0.5 * float(fov_y))
        return cls(position=tuple(float(component) for component in position), focal_pixels=focal)  # type: ignore[arg-type]

    def screen_error(self, entry: CellEntry) -> float:
        """How many pixels this cell's LOD error is worth from here.

        Distance is measured to the cell's box, not to its centre: a cell the camera is inside
        would otherwise report a distance of nearly zero from one corner and be descended into
        forever.
        """
        low = np.asarray(entry.bbox_min, dtype=np.float64)
        high = np.asarray(entry.bbox_max, dtype=np.float64)
        eye = np.asarray(self.position, dtype=np.float64)
        outside = np.maximum(np.maximum(low - eye, eye - high), 0.0)
        distance = float(np.linalg.norm(outside))
        if distance <= 0.0:
            return math.inf  # inside the box: always the finest level available
        return entry.lod_error * self.focal_pixels / distance


def _overlaps(entry: CellEntry, box: tuple[Sequence[float], Sequence[float]] | None) -> bool:
    """Whether a cell's bounds meet a query box. Touching counts."""
    if box is None:
        return True
    low = np.asarray(box[0], dtype=np.float64)
    high = np.asarray(box[1], dtype=np.float64)
    return bool(
        (np.asarray(entry.bbox_max, dtype=np.float64) >= low).all()
        and (np.asarray(entry.bbox_min, dtype=np.float64) <= high).all()
    )


def _inside_frustum(entry: CellEntry, planes: Sequence[tuple[Sequence[float], float]] | None) -> bool:
    """Whether a cell's box is on the inside of every plane.

    Planes are ``(normal, offset)`` with a point ``p`` inside where ``dot(normal, p) + offset
    >= 0``. The test is the standard conservative one -- it uses the box corner furthest along
    the normal, so it never culls a cell that is partly visible, and may keep one that is not.
    """
    if not planes:
        return True
    low = np.asarray(entry.bbox_min, dtype=np.float64)
    high = np.asarray(entry.bbox_max, dtype=np.float64)
    for normal, offset in planes:
        direction = np.asarray(normal, dtype=np.float64)
        furthest = np.where(direction >= 0.0, high, low)
        if float(np.dot(direction, furthest)) + float(offset) < 0.0:
            return False
    return True


def plan_cells(
    collection: Collection,
    *,
    camera: Camera | None = None,
    pixel_budget: float = 1.0,
    error_budget: float | None = None,
    box: tuple[Sequence[float], Sequence[float]] | None = None,
    frustum: Sequence[tuple[Sequence[float], float]] | None = None,
    objects: Sequence[int] | None = None,
    max_cells: int | None = None,
    min_level: int = 0,
) -> list[CellEntry]:
    """Select the cells to draw, descending the octree from the coarsest level.

    Args:
        collection: The collection to plan over. Only its cell catalog is read.
        camera: Spend the error budget in pixels, from a viewpoint. Distance-dependent.
        pixel_budget: With a camera, how many pixels of LOD error are acceptable.
        error_budget: Without a camera, how many voxels of LOD error are acceptable. When
            neither is given, the finest level in range is selected.
        box: Keep only cells meeting this ``(low, high)`` box, in voxels.
        frustum: Keep only cells inside these ``(normal, offset)`` planes.
        objects: Keep only cells holding at least one of these object ids -- resolved through
            the object catalog, which is the lookup it exists for.
        max_cells: A **descent budget, not a hard cap**. Once selecting a cell's children would
            take the plan past it, the coarse cell is kept whole instead -- so the plan loses
            detail rather than geometry. It therefore cannot go below the number of cells at
            the coarsest level: the alternative would be dropping one, and a missing cell is a
            hole in the surface where a coarse cell is merely blurry. Size a fetch queue off
            ``len(plan)``, not off this number.
        min_level: Never descend below this level.

    Returns:
        The chosen cells, coarsest first and in Morton order within a level.
    """
    if max_cells is not None and max_cells < 1:
        raise ValueError(f"`max_cells` is at least one cell, got {max_cells}.")

    allowed: set[tuple[int, int]] | None = None
    if objects is not None:
        # The union of the cells the object catalog names for each requested object. A cell
        # holding any one of them is kept -- a cell is the smallest thing a reader fetches, so
        # narrowing further would mean fetching it and discarding part of it anyway.
        allowed = {key for object_id in objects for key in collection.objects[int(object_id)].cells}

    def wanted(entry: CellEntry) -> bool:
        if allowed is not None and (entry.level, entry.cell) not in allowed:
            return False
        return _overlaps(entry, box) and _inside_frustum(entry, frustum)

    def good_enough(entry: CellEntry) -> bool:
        if entry.level <= min_level:
            return True
        if camera is not None:
            return camera.screen_error(entry) <= pixel_budget
        if error_budget is not None:
            return entry.lod_error <= error_budget
        return False  # no budget stated: descend to the finest level available

    selected: dict[tuple[int, int], CellEntry] = {}

    # A max-heap on level so the descent is breadth-first from the coarsest level down: when
    # `max_cells` cuts it short, what has been selected is a complete covering at some mixture
    # of levels rather than a fully-refined corner and nothing elsewhere.
    queue: list[tuple[int, int, CellEntry]] = []
    for entry in collection.roots():
        if wanted(entry):
            heapq.heappush(queue, (-entry.level, entry.cell, entry))

    while queue:
        _, _, entry = heapq.heappop(queue)

        if good_enough(entry):
            selected[(entry.level, entry.cell)] = entry
            continue

        children = [
            child
            for key in entry.children()
            if (child := collection.cells.get(key)) is not None and wanted(child)
        ]
        if not children:
            # Nothing finer exists here -- this cell is as good as this collection gets.
            selected[(entry.level, entry.cell)] = entry
            continue

        if max_cells is not None and len(selected) + len(queue) + len(children) > max_cells:
            # Descending would overrun the budget, so keep the coarse cell whole. Degrading
            # detail is the right failure; dropping the cell would punch a hole in the surface.
            selected[(entry.level, entry.cell)] = entry
            continue

        for child in children:
            heapq.heappush(queue, (-child.level, child.cell, child))

    return sorted(selected.values(), key=lambda entry: (-entry.level, entry.cell))


__all__ = ["Camera", "plan_cells"]
