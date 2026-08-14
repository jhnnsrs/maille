"""maille: a level-of-detail mesh wire format.

A mesh collection is an **octree of surfaces** written as one self-describing tree, so a
renderer can fetch the detail it needs for the view it has instead of the whole thing::

    <prefix>/
      meshed.json                    <- the manifest, written LAST
      catalog/cells.parquet          <- the spatial index, one row per (level, cell)
      catalog/objects.parquet        <- the identity index, one row per object
      level=0/part-00000.parquet     <- the geometry, finest level
      level=1/part-00000.parquet
      level=2/part-00000.parquet

Writing one::

    import trimesh
    from obstore.store import LocalStore
    import maille

    objects = {1: trimesh.creation.icosphere(radius=4.0), 2: ...}
    manifest = maille.write_meshes(
        objects,
        LocalStore("/data"),          # or an S3Store, or maille.DirectoryStore
        prefix="my-collection",
        cell_size=(128, 128, 64),     # in voxels, in your data's own component order
    )

Reading it back::

    collection = maille.open_collection(LocalStore("/data"), "my-collection")
    for entry in collection.plan(camera=maille.Camera.perspective((0, 0, 500), fov_y=0.8, viewport_height=1080)):
        cell = collection.read_cell(entry.level, entry.cell)
        draw(cell.vertices, cell.faces)

    sphere = collection.object_mesh(1)   # one object, reassembled across its cells

Coordinates
-----------
**maille addresses components by position, never by name.** Vertices, ``cell_size`` and the
``bbox_*`` columns are components 0, 1 and 2, and nothing in the writer, the octree or the
planner asks what they mean. Feed them in whatever order your data already has -- meshes off a
``(z, y, x)`` volume stay ``(z, y, x)`` -- as long as you feed them *consistently*, and the
octree comes out the same either way.

The ``x``/``y``/``z`` in the ``bbox_min_x`` column names are labels for those three slots,
fixed by the Parquet schema a server checks. They are not a claim about which physical axis
each slot holds; the optional ``axes`` field is what carries that claim, for whatever layer
owns the coordinate system.

An order mistake here cannot misplace geometry: clipping and quantization read the same
``cell_size``, so a mismatched one yields a differently *shaped* octree rather than displaced
vertices. Still worth matching ``cell_size`` to the source array's chunk shape, in whatever
order that shape is in -- a cell that matches the chunking means a viewer fetching image chunks
and mesh cells pulls the same regions.

Simplification
--------------
A coarse level is made by a pluggable backend. :class:`MeshoptSimplifier` is the default where
``meshoptimizer`` is installed -- quadric-error, and it never invents a vertex position, which
is what makes ``boundary: LOCKED`` provable rather than intended. :class:`GreedyEdgeCollapse`
is the pure-numpy fallback, so the two-dependency core can still build a multi-level tree::

    maille.build_collection(objects, cell_size=..., simplifier=maille.GreedyEdgeCollapse())
    maille.build_collection(objects, cell_size=..., decimation=maille.Decimation.half())

How much survives each level is :class:`Decimation`, defaulting to a quarter. Whatever it is,
the manifest declares what was actually done: a ratio and its declaration are required to
agree, because nothing downstream can re-derive one from the other.

The byte format is documented in :mod:`maille.codec`; the boundary and decimation arguments in
:mod:`maille.geometry`; the tree layout in :mod:`maille.manifest`.
"""

from maille.build import MeshCollection, build_collection, choose_cell_size
from maille.codec import (
    QUANT_MAX,
    cell_box,
    decode_indices,
    decode_positions,
    encode_indices,
    encode_positions,
    morton_decode,
    morton_encode,
    morton_encode_one,
)
from maille.errors import (
    FormatError,
    MailleError,
    MissingExtraError,
    PartitioningError,
    UnfinishedCollectionError,
)
from maille.frames import DEFAULT_ROW_GROUP_BYTES, REQUIRED_COLUMNS, arrow_schemas, validate_columns
from maille.geometry import decimate_fixed, snap_boundary
from maille.manifest import (
    BOUNDARY_LOCKED,
    CELL_CATALOG_PATH,
    CODEC_MESHOPT,
    CODEC_NONE,
    COMPRESSION_NONE,
    DECIMATION_CUSTOM,
    DECIMATION_EIGHTH,
    DECIMATION_HALF,
    DECIMATION_QUARTER,
    INDICES_UINT32,
    MANIFEST_NAME,
    OBJECT_CATALOG_PATH,
    POSITIONS_UINT16_QUANTIZED_PER_CELL,
    SPEC_VERSION,
    Decimation,
    Encoding,
    FileEntry,
    Grid,
    Manifest,
    level_part_path,
    level_prefix,
)
from maille.planner import Camera, plan_cells
from maille.reader import CellEntry, Collection, DecodedCell, ObjectEntry, open_collection
from maille.simplify import (
    GreedyEdgeCollapse,
    MeshoptSimplifier,
    Simplified,
    Simplifier,
    auto_simplifier,
)
from maille.sources import Mesh, MeshSource, coerce_mesh
from maille.store import DirectoryStore, MailleStore, MemoryStore, RangeReadable, StoreFile
from maille.writer import awrite_collection, write_collection, write_meshes

__all__ = [
    "BOUNDARY_LOCKED",
    "CELL_CATALOG_PATH",
    "CODEC_MESHOPT",
    "CODEC_NONE",
    "COMPRESSION_NONE",
    "DECIMATION_CUSTOM",
    "DECIMATION_EIGHTH",
    "DECIMATION_HALF",
    "DECIMATION_QUARTER",
    "DEFAULT_ROW_GROUP_BYTES",
    "INDICES_UINT32",
    "MANIFEST_NAME",
    "OBJECT_CATALOG_PATH",
    "POSITIONS_UINT16_QUANTIZED_PER_CELL",
    "QUANT_MAX",
    "REQUIRED_COLUMNS",
    "SPEC_VERSION",
    "Camera",
    "CellEntry",
    "Collection",
    "Decimation",
    "DecodedCell",
    "DirectoryStore",
    "Encoding",
    "FileEntry",
    "FormatError",
    "GreedyEdgeCollapse",
    "Grid",
    "MailleError",
    "MailleStore",
    "Manifest",
    "MemoryStore",
    "Mesh",
    "MeshCollection",
    "MeshSource",
    "MeshoptSimplifier",
    "MissingExtraError",
    "ObjectEntry",
    "PartitioningError",
    "RangeReadable",
    "Simplified",
    "Simplifier",
    "StoreFile",
    "UnfinishedCollectionError",
    "arrow_schemas",
    "auto_simplifier",
    "awrite_collection",
    "build_collection",
    "cell_box",
    "choose_cell_size",
    "coerce_mesh",
    "decimate_fixed",
    "decode_indices",
    "decode_positions",
    "encode_indices",
    "encode_positions",
    "level_part_path",
    "level_prefix",
    "morton_decode",
    "morton_encode",
    "morton_encode_one",
    "open_collection",
    "plan_cells",
    "snap_boundary",
    "validate_columns",
    "write_collection",
    "write_meshes",
]
