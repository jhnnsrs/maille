"""A store that keeps objects in a dict. For tests, and for building a tree to inspect."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from maille.stores.protocol import validate_relative

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


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


__all__ = ["MemoryStore"]
