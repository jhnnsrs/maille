"""The byte format: Morton codes, per-cell quantization, and the two blob codecs.

**This module is the wire format.** A decoder in any language needs everything stated here and
nothing else, which is why the inverses live next to the encoders rather than in a test.

The octree
----------
Level 0 is the finest. A cell at level ``L`` spans ``cell_size * 2**L`` voxels per axis, so
level ``L`` has one cell for every ``2**3`` cells at level ``L-1``. A point lands in the cell
whose index triple is ``floor(p / (cell_size * 2**L))``.

``cell`` is the Morton code of that triple with **component 0 least significant**: bit ``3*n``
is bit ``n`` of ``i``, bit ``3*n+1`` is bit ``n`` of ``j``, bit ``3*n+2`` is bit ``n`` of
``k``. Each index is capped at 17 bits so the code stays under the format's ``2**53`` limit.

Components are referred to as ``x``, ``y`` and ``z`` throughout this module, and in the
``bbox_*`` column names, purely as labels for slots 0, 1 and 2 -- the code never asks which
physical axis a slot holds, and a collection whose components are ``(z, y, x)`` encodes and
decodes identically. What a slot *means* is the optional ``axes`` declaration in the manifest,
which nothing here reads.

positions
---------
``UINT16_QUANTIZED_PER_CELL``. Each vertex becomes three ``uint16`` quantized against **the
cell's grid box** -- not the data's bounding box, so a decoder needs only ``level`` and
``cell`` to invert it::

    origin = morton_to_triple(cell) * cell_size * 2**level
    extent = cell_size * 2**level
    p = origin + q / 65535 * extent

Under ``codec: NONE`` those three ``uint16`` are written little-endian and interleaved
``x, y, z``, so the blob is exactly ``6 * vertex_count`` bytes.

Under ``codec: MESHOPT`` the quantized vertices are **padded to four components**
``(x, y, z, 0)`` and run through meshopt's vertex codec. The padding is not decoration:
meshopt requires a stride divisible by 4 and three ``uint16`` is 6, so gltfpack pads quantized
positions the same way under ``EXT_meshopt_compression``. The encoded buffer does not carry
its own length -- decoding needs the row's ``vertex_count``.

indices
-------
``UINT32``, three per triangle, indexing the cell's **concatenated** vertex array -- so an
object's indices are offset-corrected by its own vertex start, not local to the object.

Under ``codec: NONE`` the blob is a flat little-endian ``uint32`` triangle list. Under
``codec: MESHOPT`` it goes through meshopt's index codec, which **preserves triangle order**
-- what keeps ``object_index_offsets`` meaningful -- but may return each triangle rotated to a
different starting vertex. Same three corners, same winding, identical surface; only a decoder
comparing index buffers byte for byte would call that a difference.

normals
-------
Omitted, and therefore absent from both the ``encoding`` object and the shard columns. An
omitted normal encoding means the renderer computes vertex normals itself.

codec / compression
-------------------
``codec`` defaults to ``MESHOPT`` and is applied, not merely declared: it is the codec glTF's
``EXT_meshopt_compression`` uses, so a web renderer already ships the decoder, and on real
geometry it cuts the payload roughly in half (measured 0.86x on positions, 0.36x on indices,
0.46x together). ``codec: NONE`` writes the blobs raw and is one argument away.

``compression`` is ``NONE``. meshopt already entropy-codes, and stacking ZSTD on top is a
separate framing this format does not specify -- declaring one that is not applied would be a
lie no check could catch. (For reference, ZSTD over the meshopt blobs buys another ~44% on the
reference fixture, so it is worth defining one day; it is simply not defined today.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from maille.errors import MissingExtraError, PartitioningError
from maille.manifest import CODEC_MESHOPT, CODEC_NONE

#: The largest Morton code the format allows, and the per-axis index width that keeps us under
#: it: three axes at 17 bits interleave to 51.
MORTON_BITS = 17
MAX_MORTON = 1 << 53

#: A dense ordinal is 24 bits in the format.
MAX_ORDINAL = 1 << 24

#: The quantization denominator. Note it is **odd**, which is the whole reason
#: :func:`maille.geometry.snap_boundary` exists.
QUANT_MAX = 65535

#: meshopt's vertex codec takes a stride that is a multiple of 4; three uint16 is 6, so a
#: quantized position is padded to four components. gltfpack does the same.
_MESHOPT_STRIDE = 8


def require_meshoptimizer() -> Any:  # noqa: ANN401
    """Import meshoptimizer lazily, raising a helpful error if the extra is missing."""
    try:
        import meshoptimizer  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise MissingExtraError(
            "meshoptimizer is required for `codec='MESHOPT'`. Install it with "
            "`pip install maille[meshopt]`, or build with `codec='NONE'` to write the blobs "
            "uncompressed."
        ) from error
    return meshoptimizer


def _meshopt_decode_vertices(module: Any, blob: bytes, vertex_count: int) -> np.ndarray:  # noqa: ANN401
    """Decode a meshopt vertex buffer to ``(n, 4)`` uint16, avoiding the binding's unsafe path.

    ``meshoptimizer.decode_vertex_buffer`` takes a ``dtype`` argument, and **it must not be
    used**: that branch allocates ``vertex_count`` *elements* while the C function writes
    ``vertex_count * vertex_size`` *bytes*, so anything but a 4-byte dtype overruns the heap
    (verified against 0.2.30a0 -- it corrupts the allocator rather than raising). The default
    branch sizes its buffer correctly and hands back a float32 view of those bytes, so the
    bytes are recovered from it and reinterpreted here.
    """
    decoded = module.decode_vertex_buffer(vertex_count, _MESHOPT_STRIDE, blob)
    return np.frombuffer(decoded.tobytes(), dtype="<u2").reshape(vertex_count, 4)


# --------------------------------------------------------------------------- #
# Morton codes
# --------------------------------------------------------------------------- #


def _spread_bits(value: np.ndarray) -> np.ndarray:
    """Insert two zero bits after each of the low 17 bits, for a 3D Morton interleave."""
    result = np.zeros_like(value, dtype=np.int64)
    for bit in range(MORTON_BITS):
        result |= ((value >> bit) & 1) << (3 * bit)
    return result


def morton_encode(triples: np.ndarray) -> np.ndarray:
    """Morton codes for an ``(n, 3)`` array of ``(i, j, k)`` cell indices, x least significant."""
    triples = np.asarray(triples, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"Cell indices come as an (n, 3) array, got {triples.shape}.")
    if triples.min(initial=0) < 0:
        raise ValueError("Cell indices are non-negative; shift the geometry into the positive octant first.")
    if triples.max(initial=0) >= (1 << MORTON_BITS):
        raise ValueError(
            f"A cell index reached {int(triples.max())}, past the {MORTON_BITS}-bit limit that keeps "
            f"a Morton code under 2**53. Use a larger cell_size or fewer levels."
        )
    codes = _spread_bits(triples[:, 0]) | (_spread_bits(triples[:, 1]) << 1) | (_spread_bits(triples[:, 2]) << 2)
    if codes.size and int(codes.max()) >= MAX_MORTON:
        raise ValueError("A Morton code exceeded the format's 2**53 limit.")
    return codes


def morton_encode_one(triple: Sequence[int]) -> int:
    """The Morton code of a single ``(i, j, k)`` cell index triple."""
    return int(morton_encode(np.asarray([triple], dtype=np.int64))[0])


def morton_decode(code: int) -> tuple[int, int, int]:
    """The ``(i, j, k)`` cell index triple behind a Morton code. The inverse of :func:`morton_encode`."""
    i = j = k = 0
    for bit in range(MORTON_BITS):
        i |= ((code >> (3 * bit)) & 1) << bit
        j |= ((code >> (3 * bit + 1)) & 1) << bit
        k |= ((code >> (3 * bit + 2)) & 1) << bit
    return i, j, k


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #


def cell_box(cell: int, level: int, cell_size: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """The grid box of a cell: its origin and its extent, in voxels, ``(x, y, z)``."""
    sizes = np.asarray(cell_size, dtype=np.int64)
    extent = sizes.astype(np.float64) * (2**int(level))
    origin = np.asarray(morton_decode(int(cell)), dtype=np.float64) * extent
    return origin, extent


def encode_positions(
    vertices: np.ndarray,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_MESHOPT,
) -> bytes:
    """Pack vertices into the format's per-cell quantized uint16 blob.

    A vertex outside the cell is an **error, not something to clamp**. Clamping is what makes a
    partitioning bug invisible: the stray vertex is flattened onto the cell face, the blob is
    the right length, the columns are the right type, a schema check passes, and the only
    symptom is geometry quietly welded to a wall. Nothing downstream can detect it, so it is
    caught here or not at all.
    """
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
    if codec == CODEC_NONE:
        return quantized.reshape(-1).tobytes()

    # meshopt's vertex codec requires a stride that is a multiple of 4, and three uint16 is 6.
    # Padding to a fourth component is what gltfpack does for quantized positions under
    # EXT_meshopt_compression, so this is the conventional shape rather than an invention.
    module = require_meshoptimizer()
    padded = np.zeros((len(quantized), 4), dtype="<u2")
    padded[:, :3] = quantized
    return bytes(module.encode_vertex_buffer(np.ascontiguousarray(padded), len(padded), _MESHOPT_STRIDE))


def decode_positions(
    blob: bytes,
    *,
    cell: int,
    level: int,
    cell_size: Sequence[int],
    codec: str = CODEC_MESHOPT,
    vertex_count: int | None = None,
) -> np.ndarray:
    """Unpack a ``positions`` blob back into an ``(n, 3)`` float array of voxel coordinates.

    The executable half of the format documentation, and what the round-trip check asserts
    against. ``vertex_count`` is required under ``MESHOPT`` -- the encoded buffer does not
    carry its own length, which is why the geometry row has the column.
    """
    origin, extent = cell_box(cell, level, cell_size)
    if codec == CODEC_NONE:
        quantized = np.frombuffer(blob, dtype="<u2").reshape(-1, 3).astype(np.float64)
    else:
        if vertex_count is None:
            raise ValueError(
                "`vertex_count` is required to decode a MESHOPT positions blob; it is the geometry row's `vertex_count`."
            )
        module = require_meshoptimizer()
        quantized = _meshopt_decode_vertices(module, blob, vertex_count)[:, :3].astype(np.float64)
    return origin + quantized / QUANT_MAX * extent


def encode_indices(faces: np.ndarray, *, codec: str = CODEC_MESHOPT, vertex_count: int | None = None) -> bytes:
    """Pack triangle indices into the format's uint32 blob.

    Under ``MESHOPT`` the triangle *order* is preserved -- which is what keeps
    ``object_index_offsets`` meaningful -- but each triangle may come back rotated to a
    different starting vertex. Same three corners, same winding, so the surface is identical;
    a decoder that compares index buffers byte for byte will see a difference that is not one.
    """
    faces = np.asarray(faces, dtype="<u4").reshape(-1, 3)
    if codec == CODEC_NONE:
        return faces.reshape(-1).tobytes()

    module = require_meshoptimizer()
    if vertex_count is None:
        vertex_count = int(faces.max()) + 1 if faces.size else 0
    return bytes(module.encode_index_buffer(np.ascontiguousarray(faces.reshape(-1)), faces.size, vertex_count))


def decode_indices(blob: bytes, *, codec: str = CODEC_MESHOPT, index_count: int | None = None) -> np.ndarray:
    """Unpack an ``indices`` blob back into an ``(m, 3)`` triangle array.

    ``index_count`` is required under ``MESHOPT``; it is the geometry row's ``index_count``.

    Unlike the vertex path, this calls the binding directly. ``decode_index_buffer`` takes the
    index *size* in bytes as its second argument rather than a dtype, so it sizes its own
    output buffer correctly -- the hazard documented on :func:`_meshopt_decode_vertices` has no
    analogue here, and the asymmetry is deliberate rather than an oversight.
    """
    if codec == CODEC_NONE:
        return np.frombuffer(blob, dtype="<u4").reshape(-1, 3).astype(np.int64)

    if index_count is None:
        raise ValueError(
            "`index_count` is required to decode a MESHOPT indices blob; it is the geometry row's `index_count`."
        )
    module = require_meshoptimizer()
    return module.decode_index_buffer(index_count, 4, blob).reshape(-1, 3).astype(np.int64)


__all__ = [
    "MAX_MORTON",
    "MAX_ORDINAL",
    "MORTON_BITS",
    "QUANT_MAX",
    "cell_box",
    "decode_indices",
    "decode_positions",
    "encode_indices",
    "encode_positions",
    "morton_decode",
    "morton_encode",
    "morton_encode_one",
    "require_meshoptimizer",
]
