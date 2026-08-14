"""Collection in, trimesh out.

The other direction: open a collection written by example 1 and get ``trimesh.Trimesh``
objects back out of it. Three ways, because a collection can be sliced three ways:

1. **One object, whole** -- ``collection.object_mesh(id)`` looks the object up in the object
   catalog, fetches only the cells that hold a piece of it, and welds them. This is the
   "isolate segment 4711" case, and it is a lookup rather than a scan.
2. **One cell** -- ``collection.read_cell(level, cell)`` decodes a cell's concatenated
   vertices and faces, and ``cell.object_mesh(id)`` slices one object back out of it.
3. **A whole level** -- every cell at a level, as one scene. The batch-export case.

It also measures the round trip, because the format is lossy in a bounded way: positions are
quantized to 16 bits against each cell's own box, so a level-0 vertex moves by at most one
quantum of that cell.

Run it (after example 1, or it will write the collection itself)::

    uv run python examples/02_maille_to_trimesh.py
"""

from __future__ import annotations

import numpy as np
from common import OUTPUT, demo_objects, ensure_collection, megabytes, rule


def main() -> None:
    """Read a collection back out into trimesh objects, three ways."""
    collection = ensure_collection()

    rule("What the store says about itself")
    # Everything needed to decode the geometry travels next to the geometry -- that is what
    # makes a collection one store rather than a handful, and it is read from `meshed.json`
    # alone, before a single Parquet file is opened.
    print(f"grid      : cell_size={collection.grid.cell_size} (x, y, z), levels={collection.grid.levels}")
    print(f"axes      : {list(collection.axes) if collection.axes else '(none declared -- optional, and never read)'}")
    print(f"encoding  : codec={collection.encoding.codec}, positions={collection.encoding.positions}")
    print(f"counts    : {collection.manifest.counts}")
    print(f"objects   : {len(collection.objects)} in the identity index")
    print(f"cells     : {len(collection.cells)} in the spatial index")

    # ---------------------------------------------------------------- #
    # 1. One object, whole.
    # ---------------------------------------------------------------- #
    rule("One object, reassembled across every cell that holds it")
    sources = demo_objects()
    object_id = sorted(collection.objects)[3]
    entry = collection.objects[object_id]

    print(f"object {object_id}: the catalog says it lives in {len(entry.cells)} cells")
    for level in range(collection.grid.levels):
        print(f"  level {level}: cells {entry.cells_at(level)}")

    mesh = collection.object_mesh(object_id, level=0)  # a maille.Mesh
    as_trimesh = mesh.as_trimesh()  # ... and now a trimesh.Trimesh

    print(f"\nreassembled: {len(as_trimesh.vertices)} vertices, {len(as_trimesh.faces)} faces")
    print(f"type       : {type(as_trimesh).__module__}.{type(as_trimesh).__name__}")
    print(f"volume     : {as_trimesh.volume:.1f} voxels^3 (the source's is {sources[object_id].volume:.1f})")

    # ---------------------------------------------------------------- #
    # How lossy is it? Bounded, and by a number you can state up front.
    # ---------------------------------------------------------------- #
    rule("The round trip, measured")
    source = sources[object_id]
    quantum = np.asarray(collection.grid.cell_extent(0)) / 65535

    print(f"one quantum at level 0 : {np.round(quantum, 5).tolist()} voxels")
    print(f"source bounds          : {np.round(source.bounds[0], 4).tolist()} .. {np.round(source.bounds[1], 4).tolist()}")
    print(f"decoded bounds         : {np.round(mesh.bounds[0], 4).tolist()} .. {np.round(mesh.bounds[1], 4).tolist()}")
    print(f"worst corner error     : {float(np.abs(np.asarray(mesh.bounds) - np.asarray(source.bounds)).max()):.6f} voxels")
    print(
        "\nPositions are quantized per cell, against the cell's own grid box rather than the\n"
        "data's bounding box -- which is what lets a decoder invert them from `level` and\n"
        "`cell` alone, with nothing else to look up."
    )

    # ---------------------------------------------------------------- #
    # 2. One cell, and one object out of a shared cell.
    # ---------------------------------------------------------------- #
    rule("One cell, and one object sliced back out of it")
    shared = sorted(
        (cell for cell in collection.cells.values() if cell.object_count > 1),
        key=lambda cell: (-cell.object_count, cell.level, cell.cell),
    )
    target = shared[0] if shared else collection.cells_at(0)[0]

    decoded = collection.read_cell(target.level, target.cell)
    print(f"cell {target.cell} at level {target.level}: {len(decoded.vertices)} vertices, "
          f"{len(decoded.faces)} faces, {len(decoded)} objects {decoded.object_ids}")

    for held in decoded.object_ids:
        piece = decoded.object_mesh(held)
        print(f"  object {held:5d}: {len(piece.vertices):4d} vertices, {len(piece.faces):4d} faces")
    print(
        "\nThe stored indices are cell-global -- an object's faces are offset by its own vertex\n"
        "start, not local to it -- so slicing one object out means subtracting that start back\n"
        "off. `DecodedCell.object_mesh` is what does it."
    )

    # ---------------------------------------------------------------- #
    # 3. A whole level, as one trimesh scene.
    # ---------------------------------------------------------------- #
    rule("A whole level, as one scene")
    import trimesh

    for level in range(collection.grid.levels):
        pieces = [
            collection.read_cell(entry.level, entry.cell).mesh().as_trimesh()
            for entry in collection.cells_at(level)
        ]
        combined = trimesh.util.concatenate(pieces)
        print(f"level {level}: {len(collection.cells_at(level)):3d} cells -> one mesh of "
              f"{len(combined.faces):6d} faces")

        if level == collection.grid.levels - 1:
            exported = OUTPUT / f"level-{level}.obj"
            exported.write_text(trimesh.exchange.obj.export_obj(combined))
            print(f"        exported the coarsest level to {exported.name} ({megabytes(exported.stat().st_size)})")

    print(
        "\nThat is the batch-export case, and note what it costs: reading every cell at a level\n"
        "is the one access pattern the octree does not help with. Fetching for a *view* is\n"
        "example 3."
    )


if __name__ == "__main__":
    main()
