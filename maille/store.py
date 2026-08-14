"""The store a collection is written into, and the little that maille asks of one.

A meshed collection is **a tree, not a file**: a manifest, two catalogs and one directory per
octree level. That shape is why the store is a parameter rather than a path -- the same tree
has to land on a local disk during development and in an S3 prefix in production, and nothing
above this module should care which.

What maille asks for is three methods::

    store.put(path, data)     # bytes -> path, relative to the store's own root
    store.get(path)           # -> bytes, or anything with a .bytes() method
    store.list(prefix)        # -> the paths under a prefix

and one it will *use if it is there*::

    store.get_range(path, start=..., length=...)   # -> the bytes in that window

That is deliberately the shape obstore already has, so ``S3Store``, ``LocalStore``,
``GCSStore``, ``AzureStore`` and ``MemoryStore`` are all usable **as they are** -- there is no
adapter, and obstore is not a dependency of maille. :class:`DirectoryStore` is here so that a
plain filesystem path works with no dependencies at all.

``get_range`` is optional because it is the one method a hand-rolled store is likely to be
missing, and its absence must degrade rather than fail: :func:`get_range_bytes` falls back to
fetching the whole object and slicing it, which is exactly what maille did everywhere before
range reads existed. What it buys when present is the whole point of the format -- reading one
cell out of a level costs one row group rather than the level.

Paths are always ``/``-joined and relative to the store, never absolute and never
``..``-relative: a collection names files inside its own tree, and a writer that could escape
it would be a path-traversal surface in whatever is holding the credentials.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from maille.errors import FormatError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


@runtime_checkable
class MailleStore(Protocol):
    """The three operations maille needs from a store.

    Structural on purpose: an obstore store satisfies it without inheriting anything, and so
    does any object exposing the same three methods.
    """

    def put(self, path: str, data: Any) -> Any:  # noqa: ANN401
        """Write ``data`` at ``path``, replacing whatever was there."""
        ...

    def get(self, path: str) -> Any:  # noqa: ANN401
        """Read ``path``. May return bytes, or a result object carrying ``.bytes()``."""
        ...

    def list(self, prefix: str | None = None) -> Any:  # noqa: ANN401
        """Enumerate what exists under ``prefix``."""
        ...


@runtime_checkable
class RangeReadable(Protocol):
    """The optional fourth method, spelled the way obstore spells it.

    Kept separate from :class:`MailleStore` so that a store without it still satisfies the
    protocol maille requires -- ``isinstance(store, RangeReadable)`` is then the question
    :func:`get_range_bytes` asks before deciding whether to fetch a window or the whole object.
    """

    def get_range(self, path: str, *, start: int, length: int | None = None) -> Any:  # noqa: ANN401
        """Read ``length`` bytes from ``start``, or to the end when ``length`` is None."""
        ...


@runtime_checkable
class AsyncReadable(Protocol):
    """The async pair, again spelled the way obstore spells them.

    Optional for the same reason ``get_range`` is: a store without them still works, its reads
    are simply run in a worker thread instead. What they buy is that a viewer fetching forty
    cells for a frame issues forty overlapping requests rather than forty sequential ones,
    which on an object store is the difference between a frame and a stall.
    """

    async def get_async(self, path: str) -> Any:  # noqa: ANN401
        """Read a whole object."""
        ...

    async def get_range_async(self, path: str, *, start: int, length: int | None = None) -> Any:  # noqa: ANN401
        """Read one window of an object."""
        ...


def join(*parts: str) -> str:
    """Join store path segments with ``/``, dropping empty ones.

    The prefix is often ``""`` (the store is already rooted at the collection), so joining has
    to survive that without emitting a leading slash -- an S3 key beginning with ``/`` is a
    different, usually empty, object.
    """
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def validate_relative(path: str) -> str:
    """Refuse a path that leaves the collection's own tree."""
    if path.startswith("/"):
        raise FormatError(f"A collection names files inside its own tree, so {path!r} cannot be absolute.")
    if any(segment == ".." for segment in path.split("/")):
        raise FormatError(f"A collection names files inside its own tree, so {path!r} cannot escape it with '..'.")
    return path


def put_bytes(store: MailleStore, path: str, data: bytes) -> None:
    """Write bytes to a store, whatever flavour of store it is."""
    store.put(validate_relative(path), data)


def get_bytes(store: MailleStore, path: str) -> bytes:
    """Read a whole object from a store as bytes.

    obstore hands back a ``GetResult`` rather than bytes, and a hand-rolled store usually hands
    back bytes; both are accepted here so neither has to know about the other.
    """
    path = validate_relative(path)
    return _payload(store.get(path), store, path)


def _payload(result: Any, store: MailleStore, path: str) -> bytes:  # noqa: ANN401
    """Coerce whatever a store handed back into bytes."""
    if isinstance(result, (bytes, bytearray, memoryview)):
        return bytes(result)
    # `callable()` narrows to a callable returning `object`, so the payload is re-widened
    # rather than the call being cast at each site.
    reader: Any = getattr(result, "bytes", None)
    if callable(reader):
        payload: Any = reader()
        return bytes(payload)
    read: Any = getattr(result, "read", None)
    if callable(read):
        payload = read()
        return bytes(payload)
    try:
        # obstore's ranged reads hand back a `Bytes` -- no `.bytes()`, but a buffer, which is
        # the cheapest thing it could return and the last shape worth accepting.
        return bytes(memoryview(result))
    except TypeError as error:
        raise FormatError(
            f"{type(store).__name__} returned {type(result).__name__}, which is not bytes, does not carry a "
            f".bytes() or .read() method and is not a buffer, so maille cannot read {path!r} from it."
        ) from error


def get_range_bytes(store: MailleStore, path: str, start: int, length: int) -> bytes:
    """Read ``length`` bytes from ``start`` of ``path``, however little the store can do.

    A store carrying ``get_range`` serves the window; one without it has the whole object
    fetched and sliced. **The fallback is correct but not cheap**, and it is the difference
    between reading a cell and reading its level -- which is why the format bothers to record
    where a cell's row group is at all.
    """
    if start < 0 or length < 0:
        raise FormatError(f"A byte window starts at or after 0 and has a length of at least 0, got {start}:{length}.")
    validate_relative(path)
    ranged: Any = getattr(store, "get_range", None)
    if callable(ranged):
        window = ranged(path, start=start, length=length)
        return _payload(window, store, path)
    return get_bytes(store, path)[start : start + length]


async def aget_bytes(store: MailleStore, path: str) -> bytes:
    """Read a whole object without blocking the event loop."""
    path = validate_relative(path)
    native: Any = getattr(store, "get_async", None)
    if callable(native):
        return _payload(await native(path), store, path)
    return await asyncio.to_thread(get_bytes, store, path)


async def aget_range_bytes(store: MailleStore, path: str, start: int, length: int) -> bytes:
    """Read one window without blocking the event loop.

    Uses the store's own async method when it has one, and otherwise runs the sync path in a
    worker thread. The thread is not a pretence: the work is a network round trip, so handing
    it to a thread genuinely overlaps it with the others in flight.
    """
    if start < 0 or length < 0:
        raise FormatError(f"A byte window starts at or after 0 and has a length of at least 0, got {start}:{length}.")
    path = validate_relative(path)
    native: Any = getattr(store, "get_range_async", None)
    if callable(native):
        return _payload(await native(path, start=start, length=length), store, path)
    return await asyncio.to_thread(get_range_bytes, store, path, start, length)


class StoreFile:
    """A seekable, read-only binary file over a store, so pyarrow can range-read a Parquet part.

    This exists because a row group's bytes are **not a Parquet file**: they carry no footer,
    so a reader handed only that window has nothing to parse. Handing pyarrow a file-like
    object instead lets ``ParquetFile.read_row_group`` decide which windows it needs -- the
    footer once, then the column chunks of the one row group -- and every one of those reads
    lands on :func:`get_range_bytes`.

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


def list_paths(store: MailleStore, prefix: str = "") -> list[str]:
    """Every object path under ``prefix``, flattened and sorted.

    Stores differ in what they yield -- obstore streams *batches* of metadata dicts, others
    yield paths one at a time -- so this normalises all of it to a sorted list of strings.
    """
    listing = store.list(prefix or None)
    paths: list[str] = []

    def absorb(item: Any) -> None:  # noqa: ANN401
        if isinstance(item, str):
            paths.append(item)
            return
        if isinstance(item, dict) and "path" in item:
            paths.append(str(item["path"]))
            return
        path_attribute = getattr(item, "path", None)
        if isinstance(path_attribute, str):
            paths.append(path_attribute)
            return
        if isinstance(item, Iterable):
            for nested in item:  # a batch of metadata, which is how obstore streams
                absorb(nested)
            return
        raise FormatError(
            f"{type(store).__name__}.list yielded {type(item).__name__}, which maille cannot read a path out of."
        )

    for item in listing:
        absorb(item)
    return sorted(paths)


class DirectoryStore:
    """A store backed by a local directory, so maille works with no dependencies at all.

    obstore's ``LocalStore`` does the same job and does it better (it is the same code path as
    every other backend). This exists so that a caller with a path and no obstore is not
    obliged to install one, and so the test suite has a store that touches a real filesystem.
    """

    def __init__(self, root: str | os.PathLike[str], *, create: bool = True) -> None:
        """Bind a directory as the root of a store, creating it unless told not to."""
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Show the directory the store is rooted at."""
        return f"DirectoryStore({str(self.root)!r})"

    def _resolve(self, path: str) -> Path:
        """Turn a store-relative path into a filesystem path inside the root."""
        return self.root / validate_relative(path)

    def put(self, path: str, data: Any) -> None:  # noqa: ANN401
        """Write bytes to a file, creating the level directory on the way."""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))

    def get(self, path: str) -> bytes:
        """Read a file whole."""
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"No object at {path!r} in {self!r}.")
        return target.read_bytes()

    def get_range(self, path: str, *, start: int, length: int | None = None) -> bytes:
        """Read one window of a file, seeking rather than reading up to it."""
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"No object at {path!r} in {self!r}.")
        with target.open("rb") as handle:
            handle.seek(int(start))
            return handle.read(-1 if length is None else int(length))

    def list(self, prefix: str | None = None) -> Iterator[str]:
        """Yield the store-relative path of every file under ``prefix``."""
        base = self.root if not prefix else self._resolve(prefix)
        if base.is_file():
            yield str(base.relative_to(self.root)).replace(os.sep, "/")
            return
        if not base.is_dir():
            return
        for entry in sorted(base.rglob("*")):
            if entry.is_file():
                yield str(entry.relative_to(self.root)).replace(os.sep, "/")


class MemoryStore:
    """A store that keeps objects in a dict. For tests, and for building a tree to inspect."""

    def __init__(self) -> None:
        """Start empty."""
        self.objects: dict[str, bytes] = {}

    def __repr__(self) -> str:
        """Show how many objects are held."""
        return f"MemoryStore({len(self.objects)} objects)"

    def put(self, path: str, data: Any) -> None:  # noqa: ANN401
        """Hold bytes under a path."""
        self.objects[validate_relative(path)] = bytes(data)

    def get(self, path: str) -> bytes:
        """Return the bytes held under a path."""
        try:
            return self.objects[validate_relative(path)]
        except KeyError as error:
            raise FileNotFoundError(f"No object at {path!r} in {self!r}.") from error

    def get_range(self, path: str, *, start: int, length: int | None = None) -> bytes:
        """Return one window of the bytes held under a path."""
        body = self.get(path)
        return body[int(start) :] if length is None else body[int(start) : int(start) + int(length)]

    def list(self, prefix: str | None = None) -> Iterator[str]:
        """Yield every held path under ``prefix``."""
        head = (prefix or "").strip("/")
        for path in sorted(self.objects):
            if not head or path == head or path.startswith(head + "/"):
                yield path


__all__ = [
    "AsyncReadable",
    "DirectoryStore",
    "MailleStore",
    "MemoryStore",
    "RangeReadable",
    "StoreFile",
    "aget_bytes",
    "aget_range_bytes",
    "get_bytes",
    "get_range_bytes",
    "join",
    "list_paths",
    "put_bytes",
    "validate_relative",
]
