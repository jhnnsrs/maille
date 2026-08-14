"""A seekable file over a store, so pyarrow can range-read one row group out of a part."""

from __future__ import annotations

import os

from maille.errors import FormatError
from maille.stores.access import get_range_bytes
from maille.stores.protocol import MailleStore, validate_relative


class StoreFile:
    """A seekable, read-only binary file over a store, so pyarrow can range-read a Parquet part.

    This exists because a row group's bytes are **not a Parquet file**: they carry no footer,
    so a reader handed only that window has nothing to parse. Handing pyarrow a file-like
    object instead lets ``ParquetFile.read_row_group`` decide which windows it needs -- the
    footer once, then the column chunks of the one row group -- and every one of those reads
    lands on :func:`maille.stores.access.get_range_bytes`.

    ``size`` is passed in rather than discovered: the protocol has no ``head``, and the writer
    already knew each part's length when it serialized it, so the manifest carries it. That
    also keeps this working against a store that can neither stat nor list.

    **It can also be filled from outside**, with :meth:`prime`. pyarrow's reader is synchronous
    and there is no honest way to make it otherwise, so the async read path inverts the
    problem: work out which windows a row group needs, fetch them all concurrently, hand them
    here, and let the parse run against memory. The reads are async; the parse is not, and does
    not need to be.
    """

    #: How much of the tail to pull on the first read into it. A Parquet footer is read as a
    #: length-then-metadata pair, so serving it from one cached window turns three round trips
    #: into one; 64 KiB covers the footer of any part maille writes.
    TAIL_BYTES = 64 * 1024

    def __init__(self, store: MailleStore, path: str, size: int) -> None:
        """Bind one object in a store as a file of a known length."""
        self.store = store
        self.path = validate_relative(path)
        self.size = int(size)
        self._position = 0
        self._closed = False
        self._tail: bytes | None = None
        self._tail_start = max(0, self.size - self.TAIL_BYTES)
        #: Windows fetched ahead of the parse, as ``(start, payload)``, newest first. There are
        #: never many -- a footer and the row groups of one batch -- so a scan is the right
        #: lookup and an interval tree would be furniture.
        self._primed: list[tuple[int, bytes]] = []

    def __repr__(self) -> str:
        """Show the object and its length."""
        return f"StoreFile({self.path!r}, size={self.size})"

    def readable(self) -> bool:
        """Always: this is a read-only view."""
        return True

    def seekable(self) -> bool:
        """Always -- which is the entire reason it exists."""
        return True

    def writable(self) -> bool:
        """Never."""
        return False

    @property
    def closed(self) -> bool:
        """Whether the file has been closed."""
        return self._closed

    def close(self) -> None:
        """Close the file. Nothing is held open underneath, so this only flips the flag."""
        self._closed = True

    def tell(self) -> int:
        """The current offset."""
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Move the offset, clamped to the object, and return where it landed."""
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._position + offset
        elif whence == os.SEEK_END:
            target = self.size + offset
        else:
            raise FormatError(f"`whence` is one of SEEK_SET, SEEK_CUR or SEEK_END, got {whence!r}.")
        self._position = max(0, min(int(target), self.size))
        return self._position

    def read(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes from the current offset, to the end when negative."""
        if self._closed:
            raise ValueError(f"{self!r} is closed.")
        remaining = self.size - self._position
        count = remaining if size is None or size < 0 else min(int(size), remaining)
        if count <= 0:
            return b""
        payload = self._window(self._position, count)
        self._position += count
        return payload

    def prime(self, start: int, payload: bytes) -> None:
        """Hand this file a window someone else already fetched."""
        self._primed.insert(0, (int(start), bytes(payload)))

    def tail_window(self) -> tuple[int, int]:
        """The ``(start, length)`` of the region a footer parse will read."""
        return self._tail_start, self.size - self._tail_start

    def has_tail(self) -> bool:
        """Whether the footer region is already in hand."""
        return self._tail is not None

    def prime_tail(self, payload: bytes) -> None:
        """Fill the footer region from an async fetch."""
        self._tail = bytes(payload)

    def holds(self, start: int, length: int) -> bool:
        """Whether this window is already in hand, so fetching it again would be waste.

        Worth asking before a prefetch rather than after: a part smaller than the tail window
        is pulled whole when its footer is read, which covers every row group in it, and a
        prefetch that did not check would re-fetch the entire part one row group at a time.
        """
        if self._tail is not None and start >= self._tail_start:
            return True
        return self._served(start, length) is not None

    def _served(self, start: int, length: int) -> bytes | None:
        """A primed window covering this request, if one is held."""
        for base, payload in self._primed:
            if base <= start and start + length <= base + len(payload):
                offset = start - base
                return payload[offset : offset + length]
        return None

    def _window(self, start: int, length: int) -> bytes:
        """One window: primed, or from the cached tail, or fetched."""
        served = self._served(start, length)
        if served is not None:
            return served
        if start >= self._tail_start:
            if self._tail is None:
                self._tail = get_range_bytes(self.store, self.path, self._tail_start, self.size - self._tail_start)
            offset = start - self._tail_start
            return self._tail[offset : offset + length]
        return get_range_bytes(self.store, self.path, start, length)


__all__ = ["StoreFile"]
