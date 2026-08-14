"""The store a collection is written into, and the little that maille asks of one.

A meshed collection is **a tree, not a file**: a manifest, two catalogs and one directory per
octree level. That shape is why the store is a parameter rather than a path -- the same tree
has to land on a local disk during development and in an S3 prefix in production, and nothing
above this module should care which.

What maille asks for is three methods::

    store.put(path, data)     # bytes -> path, relative to the store's own root
    store.get(path)           # -> bytes, or anything with a .bytes() method
    store.list(prefix)        # -> the paths under a prefix

That is deliberately the shape obstore already has, so ``S3Store``, ``LocalStore``,
``GCSStore``, ``AzureStore`` and ``MemoryStore`` are all usable **as they are** -- there is no
adapter, and obstore is not a dependency of maille. :class:`DirectoryStore` is here so that a
plain filesystem path works with no dependencies at all.

Paths are always ``/``-joined and relative to the store, never absolute and never
``..``-relative: a collection names files inside its own tree, and a writer that could escape
it would be a path-traversal surface in whatever is holding the credentials.
"""

from __future__ import annotations

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
    result = store.get(validate_relative(path))
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
    raise FormatError(
        f"{type(store).__name__}.get returned {type(result).__name__}, which is neither bytes nor "
        f"carries a .bytes() or .read() method, so maille cannot read {path!r} from it."
    )


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

    def list(self, prefix: str | None = None) -> Iterator[str]:
        """Yield every held path under ``prefix``."""
        head = (prefix or "").strip("/")
        for path in sorted(self.objects):
            if not head or path == head or path.startswith(head + "/"):
                yield path


__all__ = [
    "DirectoryStore",
    "MailleStore",
    "MemoryStore",
    "get_bytes",
    "join",
    "list_paths",
    "put_bytes",
    "validate_relative",
]
