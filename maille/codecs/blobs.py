"""Per-cell quantization, and the dispatch from a manifest's ``codec`` to an implementation.

This is the format-level pair a writer and a reader call, and it is where the part that is
*not* the codec's business lives: quantizing a vertex against its cell's grid box on the way
in, inverting that on the way out, and refusing a vertex that does not belong to the cell.

positions
---------
``UINT16_QUANTIZED_PER_CELL``. Each vertex becomes three ``uint16`` quantized against **the
cell's grid box** -- not the data's bounding box, so a decoder needs only ``level`` and
``cell`` to invert it::

    origin = morton_to_triple(cell) * cell_size * 2**level
    extent = cell_size * 2**level
    p = origin + q / 65535 * extent

indices
-------
``UINT32``, three per triangle, indexing the cell's **concatenated** vertex array -- so an
object's indices are offset-corrected by its own vertex start, not local to the object.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from maille.codecs.meshopt import MeshoptCodec
from maille.codecs.protocol import BlobCodec
from maille.codecs.raw import RawCodec
from maille.errors import FormatError, PartitioningError
from maille.manifest import CODEC_MESHOPT, CODEC_NONE, COMPRESSION_NONE
from maille.octree import cell_box

#: The quantization denominator. Note it is **odd**, which is the whole reason
#: :func:`maille.geometry.snap_boundary` exists.
QUANT_MAX = 65535

#: Every codec the format defines, keyed by the ``encoding.codec`` value that selects it. A new
#: codec is an entry here and a module beside this one; nothing above the package changes.
_CODECS: dict[str, BlobCodec] = {
    CODEC_NONE: RawCodec(),
    CODEC_MESHOPT: MeshoptCodec(),
}


def codec_for(codec: str) -> BlobCodec:
    """The implementation a manifest's ``codec`` value names, or a refusal naming what exists.

    A codec the format does not define is refused rather than guessed at: nothing in the bytes
    reveals how they were packed, so a guess here is not an error but geometry that decodes to
    garbage.
    """
    try:
        return _CODECS[codec]
    except KeyError:
        raise FormatError(f"`codec` is {codec!r}; the format defines {CODEC_NONE} and {CODEC_MESHOPT}.") from None


def encode_positions(
    vertices: np.ndarray,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
) -> bytes:
    """Pack vertices into the format's per-cell quantized uint16 blob.

    A vertex outside the cell is an **error, not something to clamp**. Clamping is what makes a
    partitioning bug invisible: the stray vertex is flattened onto the cell face, the blob is
    the right length, the columns are the right type, a schema check passes, and the only
    symptom is geometry quietly welded to a wall. Nothing downstream can detect it, so it is
    caught here or not at all.
    """
    implementation = codec_for(codec)
    origin, extent = cell_box(cell, level, cell_size)
    normalized = (np.asarray(vertices, dtype=np.float64) - origin) / extent

    # A boundary vertex is pinned to exactly 0.0 or 1.0, so the slack only has to absorb
    # floating-point noise -- a fraction of a quantum, not a quantum.
    tolerance = 0.5 / QUANT_MAX
    if normalized.size and (normalized.min() < -tolerance or normalized.max() > 1.0 + tolerance):
        stray = int(((normalized < -tolerance) | (normalized > 1.0 + tolerance)).any(axis=1).sum())
        raise PartitioningError(
            f"{stray} vertex/vertices fall outside cell {cell} at level {level}, whose box is "
            f"{origin.tolist()} + {extent.tolist()} voxels (worst normalized coordinate "
            f"{normalized.min():.6f} .. {normalized.max():.6f}). Quantization is per cell, so a "
            f"vertex outside the cell cannot be represented -- this is a partitioning bug, not a "
            f"rounding one."
        )

    quantized = np.rint(np.clip(normalized, 0.0, 1.0) * QUANT_MAX).astype("<u2")
    return implementation.encode_positions(quantized, compression=compression)


def decode_positions(
    blob: bytes,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    vertex_count: int | None = None,
) -> np.ndarray:
    """Unpack a ``positions`` blob back into an ``(n, 3)`` float array of voxel coordinates.

    The executable half of the format documentation, and what the round-trip check asserts
    against. ``vertex_count`` is accepted and checked rather than needed: a raw blob is
    self-describing at six bytes a vertex, so a count that disagrees with it means the row and
    its geometry have come apart. Under ``compression: ZSTD`` it stops being optional: a
    compressed frame carries no reliable content size, so the count *is* the length.
    """
    implementation = codec_for(codec)
    if compression != COMPRESSION_NONE and vertex_count is None:
        raise FormatError(
            "`vertex_count` is required to decode a compressed positions blob: it is how the "
            "uncompressed length is known, since the format's ZSTD framing carries no size of its own."
        )
    origin, extent = cell_box(cell, level, cell_size)
    quantized = implementation.decode_positions(blob, compression=compression, vertex_count=vertex_count)
    return origin + np.asarray(quantized, dtype=np.float64) / QUANT_MAX * extent


def encode_indices(
    faces: np.ndarray,
    *,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    vertex_count: int | None = None,
) -> bytes:
    """Pack triangle indices into the format's uint32 blob.

    Triangle order is the order handed in, which is what keeps ``object_index_offsets``
    meaningful: an object's range into the concatenated arrays has to still be its own after a
    round trip.
    """
    implementation = codec_for(codec)
    faces = np.asarray(faces, dtype="<u4").reshape(-1, 3)
    return implementation.encode_indices(faces, compression=compression, vertex_count=vertex_count)


def decode_indices(
    blob: bytes,
    *,
    codec: str = CODEC_NONE,
    compression: str = COMPRESSION_NONE,
    index_count: int | None = None,
) -> np.ndarray:
    """Unpack an ``indices`` blob back into an ``(m, 3)`` triangle array.

    ``index_count`` is checked rather than needed, for the same reason as ``vertex_count`` on
    the positions side: four bytes an index makes a raw blob self-describing, so a row that says
    otherwise is a row that no longer belongs to this geometry -- and under ``compression: ZSTD``
    it becomes required, being the only statement of the uncompressed length.
    """
    implementation = codec_for(codec)
    if compression != COMPRESSION_NONE and index_count is None:
        raise FormatError(
            "`index_count` is required to decode a compressed indices blob: it is how the "
            "uncompressed length is known, since the format's ZSTD framing carries no size of its own."
        )
    return implementation.decode_indices(blob, compression=compression, index_count=index_count)


__all__ = [
    "QUANT_MAX",
    "codec_for",
    "decode_indices",
    "decode_positions",
    "encode_indices",
    "encode_positions",
]
