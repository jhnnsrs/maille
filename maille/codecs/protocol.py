"""What a blob codec is, stated as the four calls the format needs from one.

A codec's job starts *after* quantization and ends before dequantization: it turns an ``(n, 3)``
array of ``uint16`` quantized positions, or an ``(m, 3)`` array of ``uint32`` indices, into the
bytes that go in the column, and back. Everything a codec would have to guess -- the cell's box,
the vertex count, the declared compression -- is handed to it, so an implementation is only ever
the packing itself.

Which one runs is a *value* in the manifest, never a new field: ``encoding.codec`` names it, and
:func:`maille.codecs.blobs.codec_for` resolves that name to an implementation. Adding one means
a module here and an entry in that table, and nothing above the package changes.

The two that ship are :mod:`maille.codecs.raw` (``NONE``, the format's default -- a blob is the
renderer's buffer verbatim) and :mod:`maille.codecs.meshopt` (``MESHOPT``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class BlobCodec(Protocol):
    """The four calls a codec provides, plus the manifest value that selects it.

    Structural, like the store protocol: an implementation satisfies this by having the
    methods, not by inheriting anything.
    """

    #: The ``encoding.codec`` value that selects this implementation.
    name: str

    def encode_positions(self, quantized: npt.NDArray[np.uint16], *, compression: str) -> bytes:
        """Pack an ``(n, 3)`` array of ``uint16`` quantized coordinates into a blob."""
        ...

    def decode_positions(
        self, blob: bytes, *, compression: str, vertex_count: int | None
    ) -> npt.NDArray[np.uint16]:
        """Unpack a positions blob back into the ``(n, 3)`` ``uint16`` array that went in.

        ``vertex_count`` comes from the geometry row. Whether it is required or merely checked
        is the codec's own business: a raw blob is self-describing at six bytes a vertex, an
        encoded one is not.
        """
        ...

    def encode_indices(self, faces: npt.NDArray[np.uint32], *, compression: str, vertex_count: int | None) -> bytes:
        """Pack an ``(m, 3)`` triangle array into a blob, preserving triangle order.

        Order is not cosmetic: ``object_index_offsets`` names ranges into the concatenated
        arrays, so a codec that reordered triangles would break slicing one object out of a
        shared cell.
        """
        ...

    def decode_indices(
        self, blob: bytes, *, compression: str, index_count: int | None
    ) -> npt.NDArray[np.int64]:
        """Unpack an indices blob back into an ``(m, 3)`` triangle array."""
        ...


__all__ = ["BlobCodec"]
