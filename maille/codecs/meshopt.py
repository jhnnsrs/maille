"""``codec: MESHOPT`` -- meshoptimizer's vertex and index codecs, behind an optional extra.

Trades a decoder on the reading side for size. It needs ``meshoptimizer``, which maille does not
depend on: the name is only resolved when a manifest actually asks for this codec, and the
missing-extra error says which extra to install.

The declared ``compression`` never reaches these methods: ``MESHOPT`` with ``ZSTD`` is refused
when the manifest is validated -- a compressed frame carries no reliable length and a meshopt
buffer has no fixed relation to its counts, so between them nothing states the uncompressed
size. Both blobs also stop being self-describing, which is why the counts from the geometry row
become required here rather than merely checked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from maille.errors import FormatError, MissingExtraError
from maille.manifest import CODEC_MESHOPT, CODEC_NONE

#: meshopt's vertex codec takes a stride that is a multiple of 4; three uint16 is 6, so a
#: quantized position is padded to four components. gltfpack does the same.
STRIDE = 8


def require_meshoptimizer() -> Any:  # noqa: ANN401
    """Import meshoptimizer, naming the extra when `codec: MESHOPT` was asked for without it."""
    try:
        import meshoptimizer  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise MissingExtraError(
            f"`codec={CODEC_MESHOPT!r}` needs meshoptimizer, which is an optional extra: install it with "
            f"`pip install maille[meshopt]`. The format's default is `codec={CODEC_NONE!r}`, which writes the "
            f"blobs raw and needs nothing."
        ) from error
    return meshoptimizer


def _decode_vertices(module: Any, blob: bytes, vertex_count: int) -> npt.NDArray[np.uint16]:  # noqa: ANN401
    """Decode a meshopt vertex buffer to ``(n, 4)`` uint16, avoiding the binding's unsafe path.

    ``meshoptimizer.decode_vertex_buffer`` takes a ``dtype`` argument, and **it must not be
    used**: that branch allocates ``vertex_count`` *elements* while the C function writes
    ``vertex_count * vertex_size`` *bytes*, so anything but a 4-byte dtype overruns the heap
    (verified against 0.2.30a0 -- it corrupts the allocator rather than raising). The default
    branch sizes its buffer correctly and hands back a float32 view of those bytes, so the
    bytes are recovered from it and reinterpreted here.
    """
    decoded = module.decode_vertex_buffer(vertex_count, STRIDE, blob)
    return np.frombuffer(decoded.tobytes(), dtype="<u2").reshape(vertex_count, 4)


class MeshoptCodec:
    """meshoptimizer's vertex and index codecs, over the format's quantized values."""

    name = CODEC_MESHOPT

    def __repr__(self) -> str:
        """Name the manifest value this implements."""
        return f"MeshoptCodec({self.name!r})"

    def encode_positions(self, quantized: npt.NDArray[np.uint16], *, compression: str) -> bytes:
        """Pad each triple to four components and run meshopt's vertex codec over them.

        The padding is not an invention: meshopt's codec requires a stride that is a multiple of
        4, and three uint16 is 6, so gltfpack pads quantized positions the same way under
        ``EXT_meshopt_compression``.
        """
        del compression
        module = require_meshoptimizer()
        quantized = np.ascontiguousarray(quantized, dtype="<u2").reshape(-1, 3)
        padded = np.zeros((len(quantized), 4), dtype="<u2")
        padded[:, :3] = quantized
        return bytes(module.encode_vertex_buffer(np.ascontiguousarray(padded), len(padded), STRIDE))

    def decode_positions(
        self, blob: bytes, *, compression: str, vertex_count: int | None
    ) -> npt.NDArray[np.uint16]:
        """Decode the vertex buffer and drop the padding component."""
        del compression
        if vertex_count is None:
            raise FormatError(
                "`vertex_count` is required to decode a MESHOPT positions blob; the encoded buffer does not "
                "carry its own length, which is why the geometry row has the column."
            )
        module = require_meshoptimizer()
        return _decode_vertices(module, blob, vertex_count)[:, :3]

    def encode_indices(self, faces: npt.NDArray[np.uint32], *, compression: str, vertex_count: int | None) -> bytes:
        """Run meshopt's index codec over the triangle list, in the order it was handed over."""
        del compression
        module = require_meshoptimizer()
        faces = np.ascontiguousarray(faces, dtype="<u4").reshape(-1, 3)
        if vertex_count is None:
            vertex_count = int(faces.max()) + 1 if faces.size else 0
        return bytes(module.encode_index_buffer(np.ascontiguousarray(faces.reshape(-1)), faces.size, vertex_count))

    def decode_indices(
        self, blob: bytes, *, compression: str, index_count: int | None
    ) -> npt.NDArray[np.int64]:
        """Decode the index buffer back into ``(m, 3)`` triangles."""
        del compression
        if index_count is None:
            raise FormatError(
                "`index_count` is required to decode a MESHOPT indices blob; it is the geometry row's "
                "`index_count`."
            )
        module = require_meshoptimizer()
        return module.decode_index_buffer(index_count, 4, blob).reshape(-1, 3).astype(np.int64)


__all__ = ["STRIDE", "MeshoptCodec", "require_meshoptimizer"]
