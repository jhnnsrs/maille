"""A store backed by a local directory, so maille works with no dependencies at all."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maille.stores.protocol import validate_relative

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


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


__all__ = ["DirectoryStore"]
