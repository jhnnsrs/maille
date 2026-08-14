"""The store protocol: on disk, in memory, and through obstore, with one code path.

The point of the protocol is that maille never learns which kind of store it has. So the
central test here writes the same collection through three unrelated stores and asserts the
trees are byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import maille
from maille.stores import (
    DirectoryStore,
    MailleStore,
    get_bytes,
    join,
    list_paths,
    validate_relative,
)


def test_obstore_satisfies_the_protocol_with_no_adapter():
    """The reason the protocol has this exact shape.

    obstore's stores expose ``put``/``get``/``list`` as methods, so an ``S3Store`` or a
    ``LocalStore`` is usable as it is -- which is what lets maille reach S3 without depending
    on obstore.
    """
    obstore = pytest.importorskip("obstore")

    store = obstore.store.MemoryStore()

    assert isinstance(store, MailleStore), "obstore no longer satisfies the protocol maille writes through"


@pytest.mark.parametrize("kind", ["memory", "directory", "obstore-local", "obstore-memory"])
def test_the_same_tree_lands_in_every_kind_of_store(collection: maille.MeshCollection, tmp_path: Path, kind: str):
    """On disk or on S3, the layout is the layout. Nothing above the store knows which it has."""
    if kind == "memory":
        store: object = maille.MemoryStore()
    elif kind == "directory":
        store = DirectoryStore(tmp_path / "dir")
    else:
        obstore = pytest.importorskip("obstore")
        store = (
            obstore.store.LocalStore(str(tmp_path / "obs"), mkdir=True)
            if kind == "obstore-local"
            else obstore.store.MemoryStore()
        )

    maille.write_collection(collection, store, "collection")  # type: ignore[arg-type]

    written = list_paths(store, "collection")  # type: ignore[arg-type]
    assert written == [
        "collection/catalog/cells.parquet",
        "collection/catalog/objects.parquet",
        "collection/level=0/part-00000.parquet",
        "collection/level=1/part-00000.parquet",
        "collection/level=2/part-00000.parquet",
        "collection/maille.json",
    ]

    opened = maille.open_collection(store, "collection")  # type: ignore[arg-type]
    assert opened.manifest.grid.cell_size == (128, 128, 64)
    assert len(opened.cells) == collection.cell_catalog.num_rows


def test_a_directory_store_writes_the_tree_a_shell_can_see(collection: maille.MeshCollection, tmp_path: Path):
    """The layout is meant to be legible on disk, not only through an API."""
    maille.write_collection(collection, DirectoryStore(tmp_path), "c")

    assert (tmp_path / "c" / "maille.json").is_file()
    assert (tmp_path / "c" / "catalog" / "cells.parquet").is_file()
    assert (tmp_path / "c" / "level=1" / "part-00000.parquet").is_file()


def test_writing_at_an_empty_prefix_does_not_produce_a_leading_slash():
    """A store already rooted at the collection is the common case for a granted prefix."""
    assert join("", "catalog/cells.parquet") == "catalog/cells.parquet"
    assert join("a", "", "b") == "a/b"
    assert join("/a/", "/b/") == "a/b"


def test_a_collection_may_not_name_a_file_outside_its_own_tree():
    """A writer that could escape its prefix is a path-traversal surface in whatever holds the
    credentials, so the escape is refused at the boundary rather than trusted not to happen."""
    store = maille.MemoryStore()

    with pytest.raises(maille.FormatError, match="cannot be absolute"):
        store.put("/etc/passwd", b"x")
    with pytest.raises(maille.FormatError, match="escape it"):
        store.put("a/../../b", b"x")
    assert validate_relative("level=0/part-00000.parquet")


def test_get_bytes_accepts_both_flavours_of_result():
    """obstore hands back a result object, a hand-rolled store hands back bytes."""

    class ResultLike:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def bytes(self) -> bytes:
            return self.payload

    class Wrapped:
        def put(self, path: str, data: object) -> None: ...

        def get(self, path: str) -> object:
            return ResultLike(b"payload")

        def list(self, prefix: str | None = None) -> list[str]:
            return []

    plain = maille.MemoryStore()
    plain.put("a", b"payload")

    assert get_bytes(Wrapped(), "a") == b"payload"  # type: ignore[arg-type]
    assert get_bytes(plain, "a") == b"payload"


def test_a_store_whose_get_returns_something_unreadable_says_so():
    """Better than an AttributeError three frames deeper."""

    class Odd:
        def put(self, path: str, data: object) -> None: ...

        def get(self, path: str) -> object:
            return 42

        def list(self, prefix: str | None = None) -> list[str]:
            return []

    with pytest.raises(maille.FormatError, match="is not bytes"):
        get_bytes(Odd(), "a")  # type: ignore[arg-type]


def test_listing_is_flattened_however_the_store_streams_it():
    """obstore streams batches of metadata; others yield paths. Both normalise to sorted paths."""

    class Batched:
        def put(self, path: str, data: object) -> None: ...

        def get(self, path: str) -> object:
            return b""

        def list(self, prefix: str | None = None) -> object:
            return iter([[{"path": "b/2", "size": 1}, {"path": "b/1", "size": 1}]])

    assert list_paths(Batched(), "b") == ["b/1", "b/2"]  # type: ignore[arg-type]


def test_a_memory_store_lists_only_under_the_prefix_it_is_asked_about():
    """A prefix is a directory, so `coll` must not match `collection`."""
    store = maille.MemoryStore()
    store.put("coll/a", b"1")
    store.put("collection/b", b"2")

    assert list_paths(store, "coll") == ["coll/a"]
