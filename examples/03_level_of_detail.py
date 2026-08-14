"""Level of detail: fetching the detail a view needs, and not the rest.

This is what the whole format is for. A collection is an octree: level 0 is full detail, and
each coarser level has one cell per eight finer ones and about a quarter of the faces. A
renderer reads the **cell catalog** once -- one small Parquet file, one row per cell, no
geometry -- and from it alone decides which cells to fetch at which level.

What this shows, in order:

1. **The pyramid.** What each level costs.
2. **A budget in voxels.** ``plan(error_budget=...)``: scale-free, for when there is no camera.
3. **A budget in pixels.** ``plan(camera=...)``: error shrinks with distance, so a far cell
   settles coarse and a near one descends. This is the mixed plan an octree exists to produce.
4. **Running out of budget.** ``max_cells`` degrades detail rather than punching holes.
5. **One object.** The object catalog turns "where is segment 4711" into a set of cell keys.

Run it (after example 1, or it will write the collection itself)::

    uv run python examples/03_level_of_detail.py
"""

from __future__ import annotations

import numpy as np
from common import ensure_collection, rule

import maille


def blob_bytes(collection: maille.Collection) -> dict[tuple[int, int], int]:
    """How many bytes of geometry each cell actually holds, keyed by ``(level, cell)``.

    Read from the geometry files once so the numbers below are the bytes a fetch would really
    move, rather than a proxy computed off the vertex counts.
    """
    sizes: dict[tuple[int, int], int] = {}
    for level in range(collection.grid.levels):
        table = collection.geometry(level)
        for row in range(table.num_rows):
            key = (level, table.column("cell")[row].as_py())
            sizes[key] = len(table.column("positions")[row].as_py()) + len(table.column("indices")[row].as_py())
    return sizes


def summarise(collection: maille.Collection, plan: list[maille.CellEntry], sizes: dict) -> tuple[int, int, str]:
    """The three numbers that matter about a plan: cells, faces, bytes -- plus its level mix."""
    faces = sum(entry.face_count for entry in plan)
    payload = sum(sizes.get((entry.level, entry.cell), 0) for entry in plan)
    mix = ", ".join(
        f"L{level}x{sum(1 for entry in plan if entry.level == level)}"
        for level in sorted({entry.level for entry in plan})
    )
    return faces, payload, mix


def main() -> None:
    """Walk through what an LOD fetch actually looks like."""
    collection = ensure_collection()
    sizes = blob_bytes(collection)
    full = collection.plan()  # no budget: everything, at full detail
    _, full_bytes, _ = summarise(collection, full, sizes)

    # ---------------------------------------------------------------- #
    # 1. The pyramid.
    # ---------------------------------------------------------------- #
    rule("1. What each level costs")
    print(f"{'level':>6}  {'cells':>6}  {'faces':>8}  {'bytes':>9}  {'lod_error (voxels)':>20}")
    for level in range(collection.grid.levels):
        entries = collection.cells_at(level)
        payload = sum(sizes[(level, entry.cell)] for entry in entries)
        errors = [entry.lod_error for entry in entries]
        print(f"{level:>6}  {len(entries):>6}  {sum(e.face_count for e in entries):>8}  {payload:>9,}  "
              f"{min(errors):>9.4f} .. {max(errors):<8.4f}")

    print(
        "\n`lod_error` is the writer's upper bound on how far a vertex at this level may sit\n"
        "from where it sits at level 0: the quantization step, plus the furthest any vertex\n"
        "moved during decimation. Every budget below is spent against it.\n"
        "\n"
        "It is deliberately conservative, and the coarse numbers are not a mistake -- when a\n"
        "small object collapses to a handful of triangles, its vertices really did move by\n"
        "something like its own radius. It is an upper bound rather than a measured Hausdorff\n"
        "distance, so it over-fetches rather than under-draws, which is the safe direction.\n"
        "Note also that a parent's error always dominates its children's: without that, a\n"
        "tighter budget could buy you a coarser cell."
    )

    # ---------------------------------------------------------------- #
    # 2. A budget in voxels.
    # ---------------------------------------------------------------- #
    rule("2. Spending the budget in voxels, with no camera at all")
    print(f"{'error_budget':>13}  {'cells':>6}  {'faces':>8}  {'bytes':>9}  {'of full':>8}  levels")
    for budget in (0.0, 1.0, 5.0, 20.0, 50.0, 200.0):
        plan = collection.plan(error_budget=budget)
        faces, payload, mix = summarise(collection, plan, sizes)
        print(f"{budget:>13.2f}  {len(plan):>6}  {faces:>8}  {payload:>9,}  {payload / full_bytes:>7.0%}  {mix}")

    print(
        "\nScale-free, and what you want for a batch export or a fixed-detail fetch. The plan\n"
        "is a *cut* through the octree: every region appears exactly once, at one level."
    )

    # ---------------------------------------------------------------- #
    # 3. A budget in pixels, from a camera.
    # ---------------------------------------------------------------- #
    rule("3. Spending the budget in pixels, from a viewpoint")
    low = np.min([entry.bbox_min for entry in collection.cells.values()], axis=0)
    high = np.max([entry.bbox_max for entry in collection.cells.values()], axis=0)
    centre = (low + high) / 2
    span = float(np.linalg.norm(high - low))
    print(f"the collection spans {np.round(low, 0).tolist()} .. {np.round(high, 0).tolist()} voxels\n")

    print("pulling back along +z, accepting 8 pixels of error:\n")
    print(f"{'distance':>9}  {'cells':>6}  {'faces':>8}  {'bytes':>9}  {'of full':>8}  levels")
    for factor in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        eye = centre + np.array([0.0, 0.0, span * factor])
        camera = maille.Camera.perspective(eye, fov_y=0.8, viewport_height=1080)
        plan = collection.plan(camera=camera, pixel_budget=8.0)
        faces, payload, mix = summarise(collection, plan, sizes)
        print(f"{span * factor:>9.0f}  {len(plan):>6}  {faces:>8}  {payload:>9,}  {payload / full_bytes:>7.0%}  {mix}")

    print(
        "\nThe payload falls away as the camera pulls back --\n"
        "without the renderer asking a server for anything, and without a single geometry file\n"
        "being opened to decide it."
    )

    # A camera *inside* the scene is where a mixed plan shows up: the objects it is standing
    # among need level 0, the ones at the far end of the volume do not.
    rule("   the same thing, from inside the scene")
    eye = np.array([low[0] + 60.0, centre[1], centre[2]])
    camera = maille.Camera.perspective(eye, fov_y=0.8, viewport_height=1080)

    print(f"{'pixel_budget':>13}  {'cells':>6}  {'faces':>8}  {'bytes':>9}  {'of full':>8}  levels")
    for budget in (0.5, 2.0, 8.0, 40.0, 200.0):
        plan = collection.plan(camera=camera, pixel_budget=budget)
        faces, payload, mix = summarise(collection, plan, sizes)
        print(f"{budget:>13.1f}  {len(plan):>6}  {faces:>8}  {payload:>9,}  {payload / full_bytes:>7.0%}  {mix}")

    mixed = collection.plan(camera=camera, pixel_budget=40.0)
    if len({entry.level for entry in mixed}) > 1:
        print("\nA mixed plan: near cells at a fine level, far ones coarse, in one fetch. The near/far")
        print("split, by distance from the eye:")
        for entry in sorted(mixed, key=lambda e: np.linalg.norm(np.asarray(e.bbox_min) - eye))[:4]:
            distance = float(np.linalg.norm(np.asarray(entry.bbox_min) - eye))
            print(f"  {distance:7.0f} voxels away -> level {entry.level}  ({entry.face_count:5d} faces)")
        print("  ...")
        for entry in sorted(mixed, key=lambda e: -np.linalg.norm(np.asarray(e.bbox_min) - eye))[:2]:
            distance = float(np.linalg.norm(np.asarray(entry.bbox_min) - eye))
            print(f"  {distance:7.0f} voxels away -> level {entry.level}  ({entry.face_count:5d} faces)")

    print(
        "\nThis is why the boundary is LOCKED. Two neighbouring cells here are drawn at\n"
        "different levels, and their shared face has to line up exactly or there is a hole\n"
        "through the surface -- so vertices on a cell face are pinned and never move during\n"
        "decimation. No renderer can stitch that after the fact."
    )

    # ---------------------------------------------------------------- #
    # 4. Running out of budget.
    # ---------------------------------------------------------------- #
    rule("4. Running out of budget degrades detail, never geometry")
    print(f"{'max_cells':>10}  {'cells':>6}  {'faces':>8}  levels")
    for cap in (1, 4, 12, 30, 200):
        plan = collection.plan(error_budget=0.0, max_cells=cap)
        faces, _, mix = summarise(collection, plan, sizes)
        print(f"{cap:>10}  {len(plan):>6}  {faces:>8}  {mix}")

    print(
        "\nIt is a descent budget rather than a hard cap, and it floors at the coarsest level:\n"
        "going lower would mean dropping a cell, and a missing cell is a hole in the surface\n"
        "where a coarse cell is merely blurry. Size a fetch queue off len(plan)."
    )

    # ---------------------------------------------------------------- #
    # 5. One object.
    # ---------------------------------------------------------------- #
    rule("5. Fetching one object out of the collection")
    object_id = sorted(collection.objects)[7]
    entry = collection.objects[object_id]

    isolated = collection.plan(objects=[object_id], error_budget=0.0)
    faces, payload, mix = summarise(collection, isolated, sizes)
    print(f"object {object_id}: the object catalog names {len(entry.cells)} cells for it")
    print(f"  fetching it at full detail: {len(isolated)} cells, {faces} faces, {payload:,} bytes "
          f"({payload / full_bytes:.1%} of the collection)")

    coarse = collection.plan(objects=[object_id], error_budget=200.0)
    faces, payload, mix = summarise(collection, coarse, sizes)
    print(f"  ... and coarsely        : {len(coarse)} cells, {faces} faces, {payload:,} bytes  [{mix}]")

    print(
        "\nThe forward direction -- which objects are in this cell -- already lives in the\n"
        "geometry row as `object_ids`. The object catalog exists to carry the reverse, so\n"
        "isolating, colouring or picking one object is a lookup rather than a scan over\n"
        "every geometry row in the collection."
    )


if __name__ == "__main__":
    main()
