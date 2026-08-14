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
pip install maille              # everything needed to build and read collections
pip install 'maille[obstore]'   # + obstore, to write to S3 and friends
pip install 'maille[meshopt]'   # + the optional MESHOPT blob codec
```

Everything that takes part in building a collection is a dependency rather than an extra:
trimesh cuts a mesh at the cell planes and `fast-simplification` makes the coarse levels.
scipy and shapely are named explicitly because trimesh does not require them and the clipper
does — `slice_mesh_plane` imports `scipy.spatial.cKDTree` and `trimesh.path.polygons` (an
unguarded `from shapely import ops`) in its body, verified by blocking each in turn.

What is genuinely optional is what a *consumer* would otherwise need code for: reaching a
remote store, and decoding a compressed blob.

maille ships a `py.typed` marker, so your type checker sees its annotations: vertices are
`NDArray[np.float64]`, faces `NDArray[np.int64]`, and the three pluggable pieces — stores,
codecs and simplifiers — are structural protocols you can satisfy without importing a base
class. The package itself is checked with basedpyright in strict mode.

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
  maille.json                  <- the manifest, written LAST
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

**The geometry lands before the catalog that points into it.** Each level part is written with
one Parquet row group per byte-budgeted run of cells, and the cell catalog records, per cell,
the part and row group holding it — facts about bytes that do not exist until the bytes do. The
manifest then records each file's length, because that is the one thing a reader cannot
discover: maille asks a store for `put`/`get`/`list` and nothing more, and a Parquet footer
lives at the end of a file you have to be able to seek to.

## Reading and planning

```python
collection = maille.open_collection(LocalStore("/data"), "my-collection")

collection.grid.cell_size      # (128, 128, 64)
collection.encoding.codec      # 'NONE'
collection.cells               # the whole spatial index, {(level, cell): CellEntry}

# Which cells, at which level, for this view? Answered from the catalog alone.
camera = maille.Camera.perspective((0, 0, 500), fov_y=0.8, viewport_height=1080)
plan = collection.plan(camera=camera, pixel_budget=1.0)

# Each cell costs the row group holding it, not the level it came from. `read_cells` goes
# further: it reads a row group once however many of the planned cells share it.
for cell in collection.read_cells([(entry.level, entry.cell) for entry in plan]):
    draw(cell.vertices, cell.faces)

collection.release()           # drop the cached levels; the catalogs stay

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
dropping geometry**: running out of budget gives you a coarser cell, never a hole. Every
`CellEntry` also carries `blob_bytes`, so a plan can be budgeted in bytes before a single fetch.

## Reading without blocking

A frame is forty cells, and forty sequential round trips to an object store is not a frame:

```python
collection = await maille.aopen_collection(S3Store(...), "my-collection")
plan = collection.plan(camera=camera)
cells = await collection.aread_cells([(e.level, e.cell) for e in plan], concurrency=16)
```

The work arrives in two waves, because the second cannot be known without the first: the
footers of every part the plan touches, then the byte span of every row group it needs — each
wave issued at once. Only then is anything parsed, and the decode goes to a worker thread.

pyarrow's reader is synchronous and maille does not pretend otherwise; what is asynchronous is
the part that is actually I/O. A store carrying `get_async` / `get_range_async` (obstore does)
has them used directly; one without them has its sync methods run in a thread, which for a
network round trip overlaps just as well.

## Stores

maille asks a store for three methods — `put(path, data)`, `get(path)`, `list(prefix)` — and
uses a fourth if it is there: `get_range(path, start=..., length=...)`. That is deliberately the
shape [obstore][obstore] already has, so its `S3Store`, `LocalStore`, `GCSStore`, `AzureStore`
and `MemoryStore` all work **as they are**, with no adapter and without obstore being a
dependency of maille.

`get_range` is the optional one because it is what a hand-rolled store is likeliest to be
missing, and its absence has to degrade rather than fail: without it, reading a cell falls back
to fetching its whole level part and slicing — correct, and exactly what maille did before the
locator existed. With it, a cell costs its row group.

For a plain path and no dependencies, `maille.DirectoryStore("/data")` does the same job;
`maille.MemoryStore()` is there for tests. Both implement `get_range`.

## Simplification

Each coarser level is the same surfaces with fewer triangles, and which algorithm does that
reduction is named the way a codec is — a value out of a small vocabulary:

```python
maille.build_collection(objects, cell_size=(128, 128, 64))                  # QUADRIC, the default
maille.build_collection(objects, cell_size=..., simplifier="GREEDY")
maille.build_collection(objects, cell_size=..., simplifier=maille.GreedyEdgeCollapse())  # configured
maille.build_collection(objects, cell_size=..., decimation=maille.Decimation.half())
```

| Backend | What it is |
| --- | --- |
| `"QUADRIC"` (`QuadricSimplifier`) | **The default**, backed by [fast-simplification][fs]. Quadric edge collapse run with `preserve_border=True`, which pins every vertex on the cut curve at *exactly* its input position while letting interior vertices move to the shape-optimal spot. That split is what the format wants: the cut curve is the only thing a neighbour shares, so it is the only thing that must not move. |
| `"GREEDY"` (`GreedyEdgeCollapse`) | Shortest-edge collapse in pure numpy. Lower quality and a much looser error estimate, but it pins only *this* level's cell planes, so it reduces harder on heavily cut objects. |

The trade is real and worth knowing before you pick. `preserve_border` is all-or-nothing: after
a coarse cell welds its children, the topological boundary still contains the level-0 seams
*interior* to that cell, and those get pinned too even though nothing across a face depends on
them. On a heavily cut object that can be most of the boundary, and the collapse then falls
short of its budget — measured on one box cut into 34 fragments, the quadric backend stopped at
74% where the greedy collapse reached 9%. maille warns when a level misses its budget and names
which cause the numbers support.

[fs]: https://github.com/pyvista/fast-simplification

`Decimation` controls how much survives each level — `quarter()` (the default), `half()`,
`eighth()`, or `custom(ratio)` — plus `floor_faces`, the smallest budget any one object's piece
is given. The ratio and the name written into `encoding.decimation` are **required to agree**:
declaring `QUARTER` while reducing by half would be a claim about the geometry that nothing
downstream could test.

Bring your own by implementing one method — `maille.Simplifier` is a structural protocol, so
there is nothing to inherit:

```python
class MySimplifier:
    name = "mine"
    uses_fixed_mask = True
    def simplify(self, vertices, faces, *, fixed, target_faces) -> maille.Simplified: ...
```

## Checking one

Two of the declarations above are kept by the writer or not at all, and no renderer can see
whether they were — it only ever looks at one cell. Given the whole collection they are
perfectly checkable, so maille ships the thing that checks:

```python
report = maille.verify(collection, tier="geometry")
if not report:
    print(report)          # every failed check, with examples
```

Three tiers, because they cost very different amounts:

| tier | reads | answers |
| --- | --- | --- |
| `structure` | the manifest and two catalogs | do the files exist, are the recorded lengths right, does `child_mask` name the children that exist, do the locators point inside real row groups, do the two catalogs agree |
| `blobs` | every cell | does every blob decode to the counts its row claims, do indices stay inside their vertex array, is every vertex inside its own cell box |
| `geometry` | every level, compared | `boundary: LOCKED` — are on-plane vertices held fixed across levels; `decimation` — is a coarse level actually smaller; is `lod_error` a real bound on deviation from the level-0 surface |

`structure` reads two small files and no geometry, which makes it cheap enough to run at
registration time — the point in a pipeline where rejecting a bad collection is free.

Nothing raises. A verifier that stops at the first problem tells you about one thing when you
wanted all of them, so every check runs and the report carries the lot.

## Blob encoding

Two independent knobs, and **both default to `NONE`** — a blob is then the raw little-endian
layout the format describes, which a consumer reads out of the Parquet column and uploads to
the GPU with nothing in between:

```python
maille.build_collection(objects, cell_size=...)                              # NONE / NONE
maille.build_collection(objects, cell_size=..., compression="ZSTD")          # smaller on disk
maille.build_collection(objects, cell_size=..., codec="MESHOPT")             # needs [meshopt]
```

| | what it is | measured on the demo scene |
| --- | --- | --- |
| `codec: NONE` | positions are 6 bytes a vertex, indices 4 bytes an index | 366 KB |
| `compression: ZSTD` | each blob is a zstd frame; its length comes from the row's vertex/index count, since the framing carries none | 355 KB |
| `codec: MESHOPT` | glTF's `EXT_meshopt_compression`, needing a decoder on the reading side | 237 KB |

The honest summary: ZSTD buys ~3% because the Parquet file is already zstd-compressed around
the blobs, so it is mostly redundant. MESHOPT buys 35% and costs the consumer a decoder.
`NONE`/`NONE` costs nothing and needs nothing, which is why it is the default.

`codec: MESHOPT` with `compression: ZSTD` is refused rather than merely discouraged: the ZSTD
framing derives a blob's uncompressed length from the row's counts, and a meshopt blob has no
fixed size per element, so the pair is undecodable.

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
maille.write_meshes(objects, store, prefix=key, cell_size=(64, 128, 128))
```

The `x`/`y`/`z` in the `bbox_min_x` / `bbox_max_z` column names are **labels for slots 0, 1
and 2**, fixed by the Parquet schema a server checks with a `DESCRIBE`. They are not a claim
about which physical axis each slot holds, and the manifest makes no such claim either: naming
these axes says how the collection relates to the image it came from or the coordinate graph it
sits in, which is knowledge the layer that owns that coordinate system has and maille does not.
A field here would be a claim nothing in the format could check, use or contradict.

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

- **`sortKey: MORTON`** — rows within a level are written in ascending Morton order. That is
  what makes a row group a spatially compact set of cells rather than an arbitrary one, and so
  what makes a row group the right unit for a reader to fetch.

What it does *not* do: it holds every object in memory and builds one shard per level, so it is
sized for thousands of objects rather than millions; and `lod_error` is an upper bound rather
than a measured Hausdorff distance.

The byte format is documented in `maille/codecs/`, how space is divided in `maille/octree.py`,
and the tree layout in `maille/manifest.py`. The three pluggable pieces are packages with the
same shape — a protocol module and one module per implementation: `maille/stores/`,
`maille/codecs/` and `maille/simplifiers/`.

[obstore]: https://developmentseed.org/obstore/
