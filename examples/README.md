# Examples

Three runnable scripts, in order. Each one prints what it did, so reading the output is most
of the point.

```bash
uv sync
uv run python examples/01_trimesh_to_maille.py
uv run python examples/02_maille_to_trimesh.py
uv run python examples/03_level_of_detail.py
```

They write to `examples/out/`, which is a plain directory you can go and look at. Delete it
whenever you like — examples 2 and 3 rebuild the collection if it is not there.

| Script | What it shows |
| --- | --- |
| [`01_trimesh_to_maille.py`](01_trimesh_to_maille.py) | `{id: trimesh.Trimesh}` → a collection on disk. Both the one-call `write_meshes` form and the two-step build-then-inspect-then-write form, plus what the tree looks like and what each level costs. |
| [`02_maille_to_trimesh.py`](02_maille_to_trimesh.py) | A collection → `trimesh.Trimesh` objects. One object reassembled across its cells, one object sliced out of a shared cell, and a whole level exported as OBJ. Measures the round-trip error. |
| [`03_level_of_detail.py`](03_level_of_detail.py) | The point of the format: spending an error budget in voxels or in pixels, what each plan actually costs to fetch, mixed-level plans from a camera inside the scene, fetching one object alone, and what the two simplification backends cost against each other. |

Each script stands on its own: no shared module, nothing to read first. That means each one
carries its own copy of the demo scene — a segmentation-shaped set of 32 objects spread along
`x`, keyed by sparse instance ids the way a label image would key them — so you can lift a
single file out of here and run it anywhere maille is installed.

The three copies of `demo_objects` must stay byte-identical: examples 2 and 3 read the tree
example 1 wrote, and example 2 measures the decoded meshes *against* that scene. A copy that
drifts does not fail — it quietly compares against a different mesh.

## What the output looks like

Both excerpts are from the bundled scene, unchanged, with every default in place — run the
scripts and you should see these exact numbers. The byte column moves with `codec` and
`compression`; the face and level columns move with the simplifier and the decimation
schedule, which is rather the point of example 3's last section.

Example 2, reading one object back:

```
reassembled: 897 vertices, 1790 faces
volume     : 72988.5 voxels^3 (the source's is 72988.6)
worst corner error     : 0.000899 voxels
```

Example 3, the LOD trade — a camera pulling back, accepting 8 pixels of error:

```
 distance   cells     faces      bytes   of full  levels
      143      83     30686    511,098      97%  L0x77, L1x6
     1428      66     21436    357,954      68%  L0x52, L1x14
     2857      41     13568    222,960      42%  L0x26, L1x13, L2x2
     5714       6      2420     37,356       7%  L2x6
```

Every one of those decisions comes out of `catalog/cells.parquet` alone — one small file, one
row per cell — without opening a single geometry file to make it.

## Writing somewhere other than a directory

`maille.DirectoryStore` is used throughout so the output is inspectable, but it is the only
line that changes:

```python
from obstore.store import S3Store

maille.write_meshes(objects, S3Store("my-bucket", ...), prefix=key, cell_size=...)
```

maille asks a store for `put`/`get`/`list` and nothing else, which is deliberately the shape
obstore already has — so its `S3Store`, `GCSStore`, `AzureStore` and `MemoryStore` all work
with no adapter.
