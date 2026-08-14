"""What maille asks of a store, and the path rules every one of them is held to.

Three protocols, none of which anything inherits from: what a store has to offer, and the two
things it may offer that maille will use if they are there. Keeping the optional halves in
separate protocols is what lets ``isinstance`` be the question asked before a range read or a
concurrent fetch is attempted -- a store missing them still satisfies the one that matters.

The implementations that ship with maille are :mod:`maille.stores.directory` and
:mod:`maille.stores.memory`; the helpers that call through these methods are in
:mod:`maille.stores.access`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from maille.errors import FormatError


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
    :func:`maille.stores.access.get_range_bytes` asks before deciding whether to fetch a window
    or the whole object.
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


__all__ = [
    "AsyncReadable",
    "MailleStore",
    "RangeReadable",
    "join",
    "validate_relative",
]
