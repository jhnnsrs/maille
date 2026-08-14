"""The blob compression the manifest declares, applied and undone.

``compression`` is ``NONE`` by default: the Parquet file around the blobs is already
zstd-compressed, which is why raw blobs cost roughly 50% rather than the 2.3x their sizes
suggest. ``ZSTD`` is the other value the format defines, and it is a blob-level frame written
*inside* that Parquet column -- which is why the uncompressed length has to be supplied from
the row rather than discovered from the frame.
"""

from __future__ import annotations

from maille.errors import FormatError
from maille.manifest import COMPRESSION_NONE, COMPRESSION_ZSTD


def require_known_compression(compression: str) -> None:
    """Refuse a compression this format does not define."""
    if compression not in (COMPRESSION_NONE, COMPRESSION_ZSTD):
        raise FormatError(
            f"`compression` is {compression!r}; the format defines {COMPRESSION_NONE} and {COMPRESSION_ZSTD}."
        )


def compress(payload: bytes, compression: str) -> bytes:
    """Apply the declared blob compression. ``NONE`` is a pass-through."""
    if compression == COMPRESSION_NONE:
        return payload
    require_known_compression(compression)
    import pyarrow as pa

    return bytes(pa.compress(payload, codec="zstd", asbytes=True))


def decompress(payload: bytes, compression: str, expected: int) -> bytes:
    """Undo :func:`compress`.

    ``expected`` is the uncompressed length, and it is **required rather than discovered**:
    ZSTD frames need not carry their content size, so the format supplies it from the row
    instead -- ``6 * vertex_count`` for positions, ``4 * index_count`` for indices. That is what
    those columns were always for, and it is why the framing needs no length prefix of its own.
    """
    if compression == COMPRESSION_NONE:
        return payload
    require_known_compression(compression)
    import pyarrow as pa

    return bytes(pa.decompress(payload, decompressed_size=expected, codec="zstd"))


__all__ = ["compress", "decompress", "require_known_compression"]
