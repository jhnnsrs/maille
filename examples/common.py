"""The scene the examples share, and where they put what they write.

A segmentation is what maille is for, so the demo scene is shaped like one: a few dozen small
objects scattered through a volume, keyed by the instance ids they would carry in a label
image. It is spread along **x** on purpose -- that is what lets a single camera see some
objects close up and others far away, which is the whole point of example 3.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import maille

#: Where the examples write. Add to .gitignore; delete it whenever you like.
OUTPUT = Path(__file__).parent / "out"

#: The collection every example after the first one reads.
COLLECTION = "segmentation"

#: In voxels, one per component, in the same order as the vertices below. Anisotropic because
#: real acquisitions are -- one axis is usually sampled coarser than the other two -- and a cell
#: that matches the source array's chunk shape is what makes a viewer fetch image chunks and
#: mesh cells over the same regions.
CELL_SIZE = (128, 128, 64)

#: An axis order a *caller* might declare. maille never reads it -- it is carried into the
#: manifest for whatever owns the coordinate system, and these examples have no such layer,
#: so they mostly leave it out. Example 1 shows the pass-through.
AXES = ("z", "y", "x")

LEVELS = 3


def demo_objects(count: int = 32) -> dict[int, Any]:
    """A segmentation-shaped scene: ``{instance_id: trimesh.Trimesh}``, in voxel coordinates.

    Every object sits in the positive octant, which the format requires: a cell index is
    non-negative, so geometry is shifted into the positive octant before it is addressed.
    """
    import trimesh

    objects: dict[int, Any] = {}
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


def store(create: bool = True) -> maille.DirectoryStore:
    """The store the examples write into: a plain directory, so you can go and look at it.

    Swapping this for ``obstore.store.S3Store(...)`` is the only change needed to write the
    same tree to S3 -- maille asks a store for ``put``/``get``/``list`` and nothing else.
    """
    return maille.DirectoryStore(OUTPUT, create=create)


def ensure_collection() -> maille.Collection:
    """Write the demo collection if it is not there yet, and open it either way."""
    if not (OUTPUT / COLLECTION / "meshed.json").is_file():
        print(f"No collection at {OUTPUT / COLLECTION}, writing one first...")
        maille.write_meshes(
            demo_objects(),
            store(),
            prefix=COLLECTION,
            cell_size=CELL_SIZE,
            levels=LEVELS,
        )

    return maille.open_collection(store(), COLLECTION)


def megabytes(size: int) -> str:
    """Format a byte count the way a fetch budget is usually discussed."""
    return f"{size / 1_000_000:6.2f} MB"


def rule(title: str) -> None:
    """Print a section heading, so a long run stays readable."""
    print(f"\n{title}\n{'-' * len(title)}")
