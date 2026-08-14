"""Trimesh in, collection out.

Takes a dict of ``trimesh.Trimesh`` objects -- the shape a mesh extractor hands you after
marching cubes over a label volume -- and writes a maille collection: an octree of surfaces as
a self-describing tree of Parquet files.

Two ways to do it, both here:

1. ``write_meshes`` -- build and write in one call. What most writers want.
2. ``build_collection`` then ``collection.write`` -- the same thing in two steps, so the three
   frames can be inspected or checked before any bytes are spent on a store.

Run it::

    uv run python examples/01_trimesh_to_maille.py

It writes to ``examples/out/segmentation`` and prints the tree that landed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

import maille

#: Where this script writes. It is in .gitignore; delete it whenever you like.
OUTPUT = Path(__file__).parent / "out"

#: The collection the other two examples read.
COLLECTION = "segmentation"

#: In voxels, one per component, in the same order as the vertices below. Anisotropic because
#: real acquisitions are -- one axis is usually sampled coarser than the other two -- and a cell
#: that matches the source array's chunk shape is what makes a viewer fetch image chunks and
#: mesh cells over the same regions.
CELL_SIZE = (128, 128, 64)

#: An axis order a *caller* might declare. maille never reads it -- it is carried into the
#: manifest for whatever owns the coordinate system. The last section shows the pass-through.
AXES = ("z", "y", "x")

LEVELS = 3


def demo_objects(count: int = 32) -> dict[int, trimesh.Trimesh]:
    """A segmentation-shaped scene: ``{instance_id: trimesh.Trimesh}``, in voxel coordinates.

    A segmentation is what maille is for, so the demo scene is shaped like one: a few dozen
    small objects scattered through a volume, keyed by the instance ids they would carry in a
    label image. It is spread along **x** on purpose -- that is what lets a single camera see
    some objects close up and others far away, which is the whole point of example 3.

    Every object sits in the positive octant, which the format requires: a cell index is
    non-negative, so geometry is shifted into the positive octant before it is addressed.
    """
    objects: dict[int, trimesh.Trimesh] = {}
    for index in range(count):
        # A slow spiral down +x, so distance from a camera at one end varies a lot.
        x = 120.0 + index * 90.0
        y = 260.0 + 120.0 * math.sin(index * 0.7)
        z = 130.0 + 60.0 * math.cos(index * 0.5)

        instance_id = 1000 + index * 7  # ids as a label image would carry them: sparse, unsorted
        if index % 3 == 0:
            body = trimesh.creation.icosphere(radius=26.0, subdivisions=3)
        elif index % 3 == 1:
            body = trimesh.creation.box(extents=[46.0, 30.0, 22.0])
        else:
            body = trimesh.creation.capsule(radius=14.0, height=40.0, count=[24, 24])
        objects[instance_id] = body.apply_translation([x, y, z])

    return objects


def store() -> maille.DirectoryStore:
    """The store this script writes into: a plain directory, so you can go and look at it.

    Swapping this for ``obstore.store.S3Store(...)`` is the only change needed to write the
    same tree to S3 -- maille asks a store for ``put``/``get``/``list`` and nothing else.
    """
    return maille.DirectoryStore(OUTPUT, create=True)


def megabytes(size: int) -> str:
    """Format a byte count the way a fetch budget is usually discussed."""
    return f"{size / 1_000_000:6.2f} MB"


def rule(title: str) -> None:
    """Print a section heading, so a long run stays readable."""
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    """Write the demo scene as a collection, then show what was written."""
    objects = demo_objects()

    rule("What goes in")
    print(f"{len(objects)} objects, keyed by the instance ids a label image would carry")
    for instance_id, mesh in list(objects.items())[:3]:
        low, high = mesh.bounds
        print(f"  id {instance_id:5d}: {len(mesh.faces):5d} faces, bounds {np.round(low, 1)} .. {np.round(high, 1)}")
    print(f"  ... and {len(objects) - 3} more")
    print(f"\ntotal input geometry: {sum(len(mesh.faces) for mesh in objects.values())} faces")

    # ---------------------------------------------------------------- #
    # 1. The one-call form.
    # ---------------------------------------------------------------- #
    rule("Writing")
    manifest = maille.write_meshes(
        objects,
        store(),
        prefix=COLLECTION,
        # In voxels, one size per component, in the same order as the vertices. maille never
        # asks which physical axis a component is -- (z, y, x) data stays (z, y, x) -- it only
        # requires the two to agree. Pass the source array's chunk shape when you know it: that
        # is the value worth matching, and no amount of looking at meshes reveals it.
        cell_size=CELL_SIZE,
        levels=LEVELS,
        # `codec` and `compression` both default to NONE, so a blob is the raw little-endian
        # layout a consumer uploads as-is. `compression="ZSTD"` and `codec="MESHOPT"` (the
        # latter needing the `meshopt` extra) trade a decoder on the reading side for size.
    )

    print(f"wrote {OUTPUT / COLLECTION}")
    print(f"  spec version : {manifest.spec_version}")
    print(f"  grid         : cell_size={manifest.grid.cell_size} (x, y, z), levels={manifest.grid.levels}")
    print(f"  axes         : {list(manifest.axes) if manifest.axes else '(none declared)'}")
    print(f"  encoding     : {manifest.encoding.to_dict()}")
    print(f"  counts       : {manifest.counts}")

    rule("The tree that landed")
    root = OUTPUT / COLLECTION
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(root)!s:32s} {megabytes(path.stat().st_size)}")
    print(
        "\nThe manifest is written LAST, and that is the completion protocol: a prefix has no\n"
        "atomic 'upload finished' flag, so a tree without `maille.json` is an interrupted write\n"
        "rather than a collection -- and opening one is refused instead of half-succeeding."
    )

    # ---------------------------------------------------------------- #
    # 2. The two-step form: build, inspect, then write.
    # ---------------------------------------------------------------- #
    rule("The same thing in two steps, when you want to look first")
    collection = maille.build_collection(objects, axes=AXES, cell_size=CELL_SIZE, levels=LEVELS)

    print(f"cell catalog   : {collection.cell_catalog.num_rows} rows -- the spatial index")
    print(f"                 columns: {', '.join(collection.cell_catalog.column_names[:6])}, ...")
    print(f"object catalog : {collection.object_catalog.num_rows} rows -- the identity index, inverted")
    print(f"                 columns: {', '.join(collection.object_catalog.column_names[:4])}, ...")
    for level, shard in collection.shards:
        faces = sum(shard.column("index_count").to_pylist()) // 3
        blobs = sum(
            len(shard.column("positions")[row].as_py()) + len(shard.column("indices")[row].as_py())
            for row in range(shard.num_rows)
        )
        print(f"level {level}        : {shard.num_rows:3d} cells, {faces:6d} faces, {megabytes(blobs)} of blobs")

    print(
        "\nLevel 0 carries more faces than went in, which is expected: an object wider than a\n"
        "cell is cut at the cell planes, and cutting a triangle that straddles a plane replaces\n"
        "it with several. That cut is also what buys `boundary: LOCKED` -- it puts vertices\n"
        "exactly on the planes, and every coarser level's planes are a subset of these, so two\n"
        "cells drawn at different levels meet without a crack. Each coarser level is then a\n"
        "quarter of the one below, which is the `decimation: QUARTER` the manifest declares."
    )

    # A frame can be checked against the columns its role requires before an upload is spent on
    # it -- the earliest point the mistake is catchable and the only point it is cheap.
    maille.validate_columns(collection.cell_catalog, "cell_catalog")
    print("\nframes check out against the columns the format requires")

    collection.write(store(), prefix="segmentation-again")
    print(f"written a second copy to {OUTPUT / 'segmentation-again'}")

    # ---------------------------------------------------------------- #
    # The one optional declaration, and who it is for.
    # ---------------------------------------------------------------- #
    rule("Declaring an axis order, if you are the layer that knows one")
    named = maille.build_collection(objects, axes=AXES, cell_size=CELL_SIZE, levels=1)
    anonymous = maille.build_collection(objects, cell_size=CELL_SIZE, levels=1)

    print(f"with    axes={list(AXES)} -> manifest axes: {named.manifest.axes}")
    print(f"without axes             -> manifest axes: {anonymous.manifest.axes}")
    print(f"same geometry either way : {named.shards[0][1].to_pylist() == anonymous.shards[0][1].to_pylist()}")
    print(
        "\nmaille never reads `axes` -- not when quantizing, not when addressing a cell, not\n"
        "when planning. Everything it computes is positional (x, y, z). Naming those axes is a\n"
        "statement about how this collection relates to something else (the image it came from,\n"
        "the coordinate graph it is placed in), which belongs to whatever owns that coordinate\n"
        "system. So it is carried through when you pass it, and simply absent when you do not --\n"
        "which says 'no claim made' rather than something possibly false."
    )

    # ---------------------------------------------------------------- #
    # Somewhere else entirely: the store is the only thing that changes.
    # ---------------------------------------------------------------- #
    rule("The same tree, somewhere else")
    print(
        "maille asks a store for put/get/list and nothing else, which is deliberately the shape\n"
        "obstore already has. So the same call writes to S3 with no adapter:\n"
        "\n"
        "    from obstore.store import S3Store\n"
        "    maille.write_meshes(objects, S3Store('my-bucket', ...), prefix=key, cell_size=...)\n"
        "\n"
        "and `maille.MemoryStore()` keeps the whole thing in a dict, which is what the tests use."
    )

    in_memory = maille.MemoryStore()
    maille.write_collection(collection, in_memory, "in-memory")
    print(f"\nsame collection, in memory: {len(in_memory.objects)} objects, "
          f"{megabytes(sum(len(body) for body in in_memory.objects.values()))} total")


if __name__ == "__main__":
    main()
