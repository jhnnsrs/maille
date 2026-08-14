"""``codec: NONE`` -- the blob is the renderer's buffer verbatim.

The format's default, and the point of it rather than a shortfall: a consumer reads the column
and uploads it, with no decoder in front of the geometry at all. Positions are three
little-endian ``uint16`` interleaved, so the blob is exactly ``6 * vertex_count`` bytes;
indices are a flat little-endian ``uint32`` triangle list at ``4 * index_count``.

Both blobs being self-describing at a fixed width is what lets the counts in the geometry row be
*checked* here rather than needed -- a row that disagrees with its blob is a row and a geometry
that came from different writes. Under ``compression: ZSTD`` the check turns into a requirement,
since the compressed frame carries no reliable length of its own.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from maille.codecs.compression import compress, decompress
from maille.errors import FormatError
from maille.manifest import CODEC_NONE


class RawCodec:
    """The identity codec: quantized values, little-endian, in the order they were handed over."""

    name = CODEC_NONE

    def __repr__(self) -> str:
        """Name the manifest value this implements."""
        return f"RawCodec({self.name!r})"

    def encode_positions(self, quantized: npt.NDArray[np.uint16], *, compression: str) -> bytes:
        """Interleave the ``uint16`` triples and hand back the bytes."""
        return compress(np.ascontiguousarray(quantized, dtype="<u2").reshape(-1).tobytes(), compression)

    def decode_positions(
        self, blob: bytes, *, compression: str, vertex_count: int | None
    ) -> npt.NDArray[np.uint16]:
        """Read the blob back as ``(n, 3)`` ``uint16``, checking the row's count against it."""
        blob = decompress(blob, compression, 6 * int(vertex_count or 0))
        quantized = np.frombuffer(blob, dtype="<u2").reshape(-1, 3)
        if vertex_count is not None and len(quantized) != vertex_count:
            raise FormatError(
                f"This positions blob holds {len(quantized)} vertices and its row declares {vertex_count}. "
                f"A blob is 6 bytes a vertex, so the two cannot disagree unless the row and the geometry "
                f"belong to different writes."
            )
        return quantized

    def encode_indices(self, faces: npt.NDArray[np.uint32], *, compression: str, vertex_count: int | None) -> bytes:
        """Flatten the triangle list to little-endian ``uint32``.

        ``vertex_count`` is unused here and accepted only so the two codecs take the same
        arguments -- a raw index needs nothing but its own width.
        """
        del vertex_count
        return compress(np.ascontiguousarray(faces, dtype="<u4").reshape(-1).tobytes(), compression)

    def decode_indices(
        self, blob: bytes, *, compression: str, index_count: int | None
    ) -> npt.NDArray[np.int64]:
        """Read the blob back as ``(m, 3)`` triangles, checking the row's count against it."""
        blob = decompress(blob, compression, 4 * int(index_count or 0))
        triangles = np.frombuffer(blob, dtype="<u4").reshape(-1, 3).astype(np.int64)
        if index_count is not None and triangles.size != index_count:
            raise FormatError(
                f"This indices blob holds {triangles.size} indices and its row declares {index_count}. "
                f"A blob is 4 bytes an index, so the two cannot disagree unless the row and the geometry "
                f"belong to different writes."
            )
        return triangles


__all__ = ["RawCodec"]
