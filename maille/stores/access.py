"""Reading and writing through a store, whatever flavour of store it turns out to be.

Everything above this module calls these functions rather than a store's methods directly, and
that is what absorbs the differences between backends: obstore hands back a ``GetResult``, a
hand-rolled store hands back bytes, and a store missing ``get_range`` has to degrade to fetching
the whole object rather than failing. None of that belongs in the reader.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from maille.errors import FormatError
from maille.stores.protocol import MailleStore, validate_relative


def put_bytes(store: MailleStore, path: str, data: bytes) -> None:
    """Write bytes to a store, whatever flavour of store it is."""
    store.put(validate_relative(path), data)


def get_bytes(store: MailleStore, path: str) -> bytes:
    """Read a whole object from a store as bytes.

    obstore hands back a ``GetResult`` rather than bytes, and a hand-rolled store usually hands
    back bytes; both are accepted here so neither has to know about the other.
    """
    path = validate_relative(path)
    return payload_bytes(store.get(path), store, path)


def payload_bytes(result: Any, store: MailleStore, path: str) -> bytes:  # noqa: ANN401
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
        return payload_bytes(window, store, path)
    return get_bytes(store, path)[start : start + length]


async def aget_bytes(store: MailleStore, path: str) -> bytes:
    """Read a whole object without blocking the event loop."""
    path = validate_relative(path)
    native: Any = getattr(store, "get_async", None)
    if callable(native):
        # `callable()` narrows to a callable returning `object`, so the awaitable is re-widened
        # rather than the await being cast at the call site.
        pending: Any = native(path)
        return payload_bytes(await pending, store, path)
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
        pending: Any = native(path, start=start, length=length)
        return payload_bytes(await pending, store, path)
    return await asyncio.to_thread(get_range_bytes, store, path, start, length)


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


__all__ = [
    "aget_bytes",
    "aget_range_bytes",
    "get_bytes",
    "get_range_bytes",
    "list_paths",
    "payload_bytes",
    "put_bytes",
]
