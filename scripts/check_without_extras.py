"""Check that maille works, and fails helpfully, on a bare numpy + pyarrow install.

Run by the `test_without_extras` CI job. The extras are imported lazily so that `import
maille` costs two dependencies -- but "lazily" is a claim that rots the moment someone adds a
top-level import, and the ordinary test suite installs everything and so cannot notice. This
script is what notices.
"""

import sys

import maille

# 1. The package imports, and the parts that are pure declaration work.
manifest = maille.Manifest(
    grid=maille.Grid(cell_size=(128, 128, 64), levels=2),
    encoding=maille.Encoding(codec=maille.CODEC_NONE),
    axes=("z", "y", "x"),  # optional, and carried rather than read
)
assert maille.Manifest.from_json(manifest.to_json()).grid.cell_size == (128, 128, 64)
print(f"manifest round-trips at spec version {maille.SPEC_VERSION}")

# 2. A store works: it is plain filesystem and dict code with no optional dependency behind it.
store = maille.MemoryStore()
store.put("catalog/cells.parquet", b"not really parquet")
assert maille.store.get_bytes(store, "catalog/cells.parquet") == b"not really parquet"
print("stores work with no extras")

# 3. The blob codec works under `codec: NONE`, which is the codec that needs no extra.
blob = maille.encode_positions(
    [[0.0, 0.0, 0.0], [128.0, 128.0, 64.0]],
    cell=0, level=0, cell_size=(128, 128, 64), codec=maille.CODEC_NONE,
)
decoded = maille.decode_positions(blob, cell=0, level=0, cell_size=(128, 128, 64), codec=maille.CODEC_NONE)
assert decoded.shape == (2, 3), decoded
print("the NONE codec round-trips with no extras")

# 4. What *does* need an extra says so by name, rather than raising a bare ImportError from
#    three frames deeper.
for describe, call in [
    ("trimesh", lambda: maille.build_collection({1: object()}, cell_size=(64, 64, 64))),
    ("meshoptimizer", lambda: maille.encode_positions(
        [[0.0, 0.0, 0.0]], cell=0, level=0, cell_size=(64, 64, 64), codec=maille.CODEC_MESHOPT
    )),
]:
    try:
        call()
    except maille.MissingExtraError as error:
        assert "pip install" in str(error), f"the {describe} error does not say how to fix it: {error}"
        print(f"missing {describe} is refused by name")
    except Exception as error:  # noqa: BLE001 - any other error is the failure this job catches
        print(f"FAIL: missing {describe} raised {type(error).__name__} instead of MissingExtraError: {error}")
        sys.exit(1)
    else:
        print(f"FAIL: {describe} appears to be installed, so this job is not testing what it claims")
        sys.exit(1)

# 5. And the fallback simplifier is what gets chosen, so a caller with trimesh but no
#    meshoptimizer can still build coarse levels. Note that *building a collection* needs
#    trimesh either way -- it is the clipper, not an alternative to one -- so the backend is
#    exercised directly here rather than through `build_collection`.
import numpy as np

from maille.simplify import GreedyEdgeCollapse, auto_simplifier

assert isinstance(auto_simplifier(), GreedyEdgeCollapse), "the fallback backend was not chosen"

vertices = np.array(
    [[0.0, 0, 0], [40, 0, 0], [40, 30, 0], [0, 30, 0], [0, 0, 20], [40, 0, 20], [40, 30, 20], [0, 30, 20]]
)
faces = np.array(
    [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
     [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
)
reduced = auto_simplifier().simplify(
    vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=6
)
assert len(reduced.faces), "the fallback simplifier produced nothing"
print(f"the {auto_simplifier().name} backend simplifies {len(faces)} faces to {len(reduced.faces)} with no extras")

print("maille works, and refuses helpfully, on numpy + pyarrow alone")
