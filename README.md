# maille

[![PyPI version](https://badge.fury.io/py/maille.svg)](https://pypi.org/project/maille/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://pypi.org/project/maille/)
![Maintainer](https://img.shields.io/badge/maintainer-jhnnsrs-blue)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/maille.svg)](https://pypi.python.org/pypi/maille/)
[![PyPI status](https://img.shields.io/pypi/status/maille.svg)](https://pypi.python.org/pypi/maille/)

**A level-of-detail wire format for meshes.** maille turns a pile of surfaces into an
octree-partitioned, self-describing tree of Parquet files, so a viewer fetches the detail the
view actually needs instead of the whole thing.

It is a *serializer*. It has no client, no network code and no opinion about where the bytes
go: you hand it a store, and the same tree lands on a local disk or in an S3 prefix.

## Why

A segmentation of a large volume is tens of thousands of surfaces and hundreds of megabytes of
triangles. A viewer that wants to draw it has two bad options — download everything, or ask a
server to prepare something. maille takes the third: **partition once, at write time**, into a
structure a dumb object store can serve and a renderer can plan against.

- **An octree of cells.** Level 0 is full detail; each coarser level has one cell per eight
  finer ones and a quarter of the faces. A renderer picks a level per cell, per frame.
- **Two catalogs, answering opposite questions.** `cells.parquet` is the spatial index — read
  once at mount, it decides which cells to fetch at which level without opening any geometry.
  `objects.parquet` is the identity index, inverted — it answers *"where is segment 4711?"*
  with a set of cell keys, making isolation and picking a lookup rather than a scan.
- **No cracks between levels.** Vertices on cell faces are pinned, so a fine cell drawn next to
  a coarse one meets it exactly. That property is what makes drawing a single object out of the
  collection viable, and it is the writer's whole job.

## Install

```bash
pip install maille                 # the format itself: numpy + pyarrow
pip install 'maille[mesh]'         # + everything needed to write a collection
pip install 'maille[complete]'     # + obstore, for S3 and friends
```

The core imports two dependencies: reading a mesh in, compressing a blob and talking to S3 are
optional extras, imported lazily and refused with a message naming the extra. `[mesh]` is the
one most writers want — it carries trimesh (which cuts a mesh at the cell planes) *and*
meshoptimizer, because `codec` defaults to `MESHOPT` and an install that could not honour the
default would be a trap.

## Examples

Three runnable scripts in [`examples/`](examples/), in order — trimesh to the format, the
format back to trimesh, and what the level-of-detail machinery actually buys:

```bash
uv run python examples/01_trimesh_to_maille.py
uv run python examples/02_maille_to_trimesh.py
uv run python examples/03_level_of_detail.py
```

## Writing

```python
import trimesh
from obstore.store import LocalStore
import maille

objects = {
    7: trimesh.creation.icosphere(radius=18.0).apply_translation([200, 160, 60]),
    3: trimesh.creation.box(extents=[40, 24, 16]).apply_translation([90, 70, 40]),
}

manifest = maille.write_meshes(
    objects,
    LocalStore("/data"),          # or S3Store(...), or maille.DirectoryStore("/data")
    prefix="my-collection",
    cell_size=(128, 128, 64),     # in voxels, in the same component order as the vertices
    levels=3,
)
```

Objects are keyed by the id they carry in whatever they were extracted from — a label volume's
instance ids, say — and those ids are written through unchanged. Each value is a
`trimesh.Trimesh`, a `maille.Mesh`, or a plain `(vertices, faces)` pair of arrays.

To inspect or check the frames before spending the writes, build and write in two steps:

```python
collection = maille.build_collection(objects, cell_size=(128, 128, 64))
collection.cell_catalog      # pyarrow.Table -- the spatial index
collection.object_catalog    # pyarrow.Table -- the identity index
collection.shards            # [(level, pyarrow.Table)]
collection.write(store, "my-collection")
```

## The tree it writes

```
my-collection/
  meshed.json                  <- the manifest, written LAST
  catalog/cells.parquet        <- one row per (level, cell)
  catalog/objects.parquet      <- one row per object
  level=0/part-00000.parquet   <- the geometry, finest level
  level=1/part-00000.parquet
  level=2/part-00000.parquet
```

**The manifest lands last, and that is the completion protocol.** A prefix has no atomic
"upload finished" flag: a `PutObject` either happened or it did not, but a tree is a sequence
of writes that can stop anywhere. So every file the manifest names is written before the
manifest is, and a prefix without one is an *interrupted write* rather than a collection —
which turns "this thing is half written" from something a renderer discovers into something
opening it rejects.

## Reading and planning

```python
collection = maille.open_collection(LocalStore("/data"), "my-collection")

collection.grid.cell_size      # (128, 128, 64)
collection.encoding.codec      # 'MESHOPT'
collection.cells               # the whole spatial index, {(level, cell): CellEntry}

# Which cells, at which level, for this view? Answered from the catalog alone.
camera = maille.Camera.perspective((0, 0, 500), fov_y=0.8, viewport_height=1080)
for entry in collection.plan(camera=camera, pixel_budget=1.0):
    cell = collection.read_cell(entry.level, entry.cell)
    draw(cell.vertices, cell.faces)

# Or spend the budget in voxels, with no camera at all.
collection.plan(error_budget=0.5)

# One object, reassembled across every cell that holds a piece of it.
mesh = collection.object_mesh(7)

# One object out of a shared cell, with its indices re-based.
piece = collection.read_cell(0, 3).object_mesh(7)
```

The planner descends from the coarsest level, keeps a cell when its LOD error fits the budget,
and otherwise descends into the children `child_mask` names — so descending costs no listing
and no second query. It also takes a query `box`, `frustum` planes, an `objects` filter
resolved through the object catalog, and a `max_cells` cap that **degrades detail rather than
dropping geometry**: running out of budget gives you a coarser cell, never a hole.

## Stores

maille asks a store for three methods — `put(path, data)`, `get(path)`, `list(prefix)` — and
nothing else. That is deliberately the shape [obstore][obstore] already has, so its `S3Store`,
`LocalStore`, `GCSStore`, `AzureStore` and `MemoryStore` all work **as they are**, with no
adapter and without obstore being a dependency of maille.

For a plain path and no dependencies, `maille.DirectoryStore("/data")` does the same job;
`maille.MemoryStore()` is there for tests.

## Coordinates

**maille addresses components by position, never by name.** Vertices, `cell_size` and the
`bbox_*` columns are component 0, 1 and 2 — and nothing in the writer, the octree or the
planner asks what those components mean. Feed them in whatever order your data already has,
as long as you feed them *consistently*, and you get the same octree either way.

That matters because meshes usually come out of marching cubes over a `(z, y, x)` array, and
the imaging stack around them is `(z, y, x)` throughout. There is no house convention to
transpose into first:

```python
# vertices from a (z, y, x) volume, and the chunk shape of that same volume
maille.write_meshes(objects, store, prefix=key, cell_size=(64, 128, 128), axes=["z", "y", "x"])
```

The `x`/`y`/`z` in the `bbox_min_x` / `bbox_max_z` column names are **labels for slots 0, 1
and 2**, fixed by the Parquet schema a server checks with a `DESCRIBE`. They are not a claim
about which physical axis each slot holds. That claim is exactly what the optional `axes`
field carries, and why it is the caller's to make.

Two consequences worth knowing:

- **An order mistake here cannot misplace geometry.** Clipping and quantization read the same
  `cell_size`, so a mismatched one gives you a differently *shaped* octree — cells that fit
  the data's anisotropy less well, so the tree narrows a fetch less — while every vertex still
  decodes exactly where it started. The failure mode is efficiency, not correctness.
- **`cell_size` is still worth matching to your source array's chunk shape.** In whatever
  order that shape is in. A cell that matches the chunking means a viewer fetching image
  chunks and mesh cells pulls the same regions, and nothing about the meshes themselves can
  reveal it.

## What the format promises, and what it does not

Two declarations are true by construction and unverifiable by anything downstream, so they are
kept by the writer or not at all:

- **`boundary: LOCKED`** — every object is cut *once*, at the level-0 planes; coarser levels
  are assembled by welding children and decimated with on-plane vertices held fixed. See
  `maille/geometry.py` for the argument, including the residual case that follows from 65535
  being odd.
- **`decimation: QUARTER`** — level `L` targets `(1/4)**L` of the level-0 face count. When a
  collection cannot reach it — usually a cell size small relative to the objects, so every cut
  vertex is pinned — maille warns and says which cause the numbers support.

What it does *not* do: it holds every object in memory and writes one file per level, so it is
sized for thousands of objects rather than millions; it does not reorder rows into Morton order
within a level; and `lod_error` is an upper bound rather than a measured Hausdorff distance.

The byte format is documented in `maille/codec.py`, the tree layout in `maille/manifest.py`.

[obstore]: https://developmentseed.org/obstore/
