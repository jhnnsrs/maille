"""The byte format: Morton codes, per-cell quantization, and the blob codecs.

**This package is the wire format.** A decoder in any language needs everything stated here and
nothing else, which is why the inverses live next to the encoders rather than in a test.

Where each part is
------------------
:mod:`~maille.codecs.blobs` is the quantization either side of a codec, and the dispatch from a
manifest's ``codec`` value to an implementation. :mod:`~maille.codecs.protocol` states what an
implementation is; :mod:`~maille.codecs.raw` and :mod:`~maille.codecs.meshopt` are the two that
ship, and :mod:`~maille.codecs.compression` is the blob compression declared alongside them.

How space is divided -- the octree levels, the Morton code a ``cell`` is, and the box it stands
for -- is :mod:`maille.octree`, not this package. Quantization needs a cell's box and imports it
from there; nothing in the addressing needs to know how a blob is packed.

Components are referred to as ``x``, ``y`` and ``z`` throughout, and in the ``bbox_*`` column
names, purely as labels for slots 0, 1 and 2 -- the code never asks which physical axis a slot
holds, and a collection whose components are ``(z, y, x)`` encodes and decodes identically. What
a slot *means* is not stated anywhere in the format: it is a claim about the collection's
relation to something else, and it belongs to whatever owns that coordinate system.

positions
---------
``UINT16_QUANTIZED_PER_CELL``. Three ``uint16`` per vertex, quantized against the cell's own
grid box, written little-endian and interleaved -- so the blob is exactly ``6 * vertex_count``
bytes and a reader can hand the column straight to a vertex buffer.

indices
-------
``UINT32``, three per triangle, indexing the cell's **concatenated** vertex array. The blob is a
flat little-endian ``uint32`` triangle list, in the order the writer emitted -- which is what
keeps ``object_index_offsets`` meaningful.

normals
-------
Omitted, and therefore absent from both the ``encoding`` object and the shard columns. An
omitted normal encoding means the renderer computes vertex normals itself.

codec / compression
-------------------
``codec`` defaults to ``NONE``: a blob is the renderer's buffer verbatim, which is the point
rather than a shortfall -- a consumer reads the column and uploads it, with no decoder in front
of the geometry at all. ``MESHOPT`` is the other value the format defines, behind an optional
extra.

The field is **required and always stated**, for the reason it always was: nothing in the bytes
reveals how they were packed, so a reader handed a manifest without it would be guessing, and a
guess here is not an error but geometry that decodes to garbage. A third codec would also arrive
as a *value* -- a module beside these and an entry in the table in
:mod:`~maille.codecs.blobs` -- not as a new field.

``compression`` likewise defaults to ``NONE``. The Parquet file around the blobs is already
zstd-compressed, which is why the raw blobs cost roughly 50% rather than the 2.3x their sizes
suggest.
"""

from maille.codecs.blobs import (
    QUANT_MAX,
    codec_for,
    decode_indices,
    decode_positions,
    encode_indices,
    encode_positions,
)
from maille.codecs.compression import compress, decompress, require_known_compression
from maille.codecs.meshopt import MeshoptCodec, require_meshoptimizer
from maille.codecs.protocol import BlobCodec
from maille.codecs.raw import RawCodec

__all__ = [
    "QUANT_MAX",
    "BlobCodec",
    "MeshoptCodec",
    "RawCodec",
    "codec_for",
    "compress",
    "decode_indices",
    "decode_positions",
    "decompress",
    "encode_indices",
    "encode_positions",
    "require_known_compression",
    "require_meshoptimizer",
]
