"""Reading without blocking the loop, and whether the reads actually overlap.

A frame is forty cells. Forty sequential round trips to an object store is not a frame, so the
async path exists to put them in flight together -- and "together" is the part worth testing.
Wall-clock is the tempting measurement and the wrong one: it is flaky under load and it passes
on a fast local store no matter how sequential the code is.

So these count **peak concurrency** instead. A store that records how many reads are in flight
at once answers the question directly and deterministically: if the reads overlapped, the peak
is greater than one, and no amount of machine speed changes that.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import numpy as np
import pytest
import trimesh

import maille
from tests.conftest import AXES, CELL_SIZE, LEVELS

ROW_GROUP_BYTES = 16 * 1024


class ConcurrentStore(maille.MemoryStore):
    """A store with async reads that records how many were ever in flight at once."""

    def __init__(self) -> None:
        """Start empty, with nothing in flight."""
        super().__init__()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def _tracked(self, work: Any) -> bytes:  # noqa: ANN401
        self.in_flight += 1
        self.calls += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            # A real store would be waiting on a socket here. Yielding to the loop is what
            # makes this store behave like one: without it every await completes before the
            # next begins and nothing is ever concurrent, however the caller arranged it.
            await asyncio.sleep(0.005)
            return work()
        finally:
            self.in_flight -= 1

    async def get_async(self, path: str) -> bytes:
        """Read a whole object, tracked."""
        return await self._tracked(lambda: self.get(path))

    async def get_range_async(self, path: str, *, start: int, length: int | None = None) -> bytes:
        """Read one window, tracked."""
        return await self._tracked(lambda: self.get_range(path, start=start, length=length))

    def forget(self) -> None:
        """Reset the record."""
        self.in_flight = self.peak = self.calls = 0


@pytest.fixture(scope="session")
def many_objects() -> dict[int, Any]:
    """Enough objects, spread far enough apart, to fill several cells and several row groups."""
    return {
        1000 + index * 7: trimesh.creation.icosphere(radius=30.0, subdivisions=3).apply_translation(
            [60.0 + index * 70.0, 80.0, 60.0]
        )
        for index in range(24)
    }


@pytest.fixture(scope="session")
def many(many_objects: dict[int, Any]) -> maille.MeshCollection:
    """The larger collection, built once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return maille.build_collection(many_objects, axes=AXES, cell_size=CELL_SIZE, levels=LEVELS)


@pytest.fixture()
def served(many: maille.MeshCollection) -> ConcurrentStore:
    """The collection in a store whose reads are async and observable."""
    store = ConcurrentStore()
    maille.write_collection(many, store, "c", row_group_bytes=ROW_GROUP_BYTES)
    return store


def test_opening_a_collection_without_blocking_reads_the_same_manifest(served: ConcurrentStore):
    """``aopen_collection`` differs from ``open_collection`` in when it yields, not in what it reads."""
    synchronous = maille.open_collection(served, "c")

    asynchronous = asyncio.run(maille.aopen_collection(served, "c"))

    assert asynchronous.manifest.to_dict() == synchronous.manifest.to_dict()
    assert asynchronous.grid == synchronous.grid
    assert asynchronous.prefix == synchronous.prefix


def test_opening_an_interrupted_write_is_refused_asynchronously_too(served: ConcurrentStore):
    """The completion protocol is a property of the tree, not of which reader looked at it."""
    del served.objects["c/maille.json"]

    with pytest.raises(maille.UnfinishedCollectionError, match="manifest last"):
        asyncio.run(maille.aopen_collection(served, "c"))


def test_the_async_reader_returns_exactly_what_the_sync_one_does(served: ConcurrentStore):
    """Two paths to the same bytes; a difference between them is a bug in one of them."""
    reference = maille.open_collection(served, "c")
    keys = [(entry.level, entry.cell) for entry in reference.plan(error_budget=0.3)]
    expected = list(reference.read_cells(keys))

    got = asyncio.run(_read(served, keys))

    assert [(cell.level, cell.cell) for cell in got] == keys, "the caller's order is the answer's order"
    for actual, wanted in zip(got, expected):
        assert np.array_equal(actual.vertices, wanted.vertices)
        assert np.array_equal(actual.faces, wanted.faces)
        assert actual.object_ids == wanted.object_ids
        assert actual.object_vertex_offsets == wanted.object_vertex_offsets


def test_the_reads_actually_overlap(served: ConcurrentStore):
    """The whole point. Sequential awaits would leave the peak at one."""
    collection = maille.open_collection(served, "c")
    keys = [(entry.level, entry.cell) for entry in collection.plan(error_budget=0.3)]
    assert len(keys) > 4, "the fixture must plan enough cells for overlap to be observable"

    served.forget()
    asyncio.run(_read(served, keys))

    assert served.peak > 1, (
        f"{served.calls} reads were issued and never more than {served.peak} was in flight at "
        f"once, so they ran one after another -- which is the thing the async path is for"
    )


def test_the_concurrency_bound_is_respected(served: ConcurrentStore):
    """A viewer must be able to stop maille from opening a hundred sockets at once."""
    collection = maille.open_collection(served, "c")
    keys = [(entry.level, entry.cell) for entry in collection.plan(error_budget=0.3)]

    served.forget()
    asyncio.run(_read(served, keys, concurrency=2))

    assert served.peak <= 2, f"asked for 2 at a time, saw {served.peak}"


def test_a_store_with_no_async_methods_still_reads(many: maille.MeshCollection):
    """Async is optional on a store, so its absence runs the sync path in a thread."""
    store = maille.MemoryStore()
    maille.write_collection(many, store, "c", row_group_bytes=ROW_GROUP_BYTES)
    reference = maille.open_collection(store, "c")
    keys = [(entry.level, entry.cell) for entry in reference.plan(error_budget=0.3)]

    got = asyncio.run(_read(store, keys))

    for actual, wanted in zip(got, reference.read_cells(keys)):
        assert np.array_equal(actual.vertices, wanted.vertices)
        assert np.array_equal(actual.faces, wanted.faces)


def test_reading_one_cell_asynchronously_matches_reading_it_synchronously(served: ConcurrentStore):
    """``aread_cell`` is the single-key convenience and must not diverge from the plural form."""
    collection = maille.open_collection(served, "c")
    entry = min((e for e in collection.cells.values() if e.level == 0), key=lambda e: e.cell)

    got = asyncio.run(_one(served, entry.level, entry.cell))
    expected = collection.read_cell(entry.level, entry.cell)

    assert np.array_equal(got.vertices, expected.vertices)
    assert np.array_equal(got.faces, expected.faces)


def test_an_unknown_cell_is_refused_before_anything_is_fetched(served: ConcurrentStore):
    """The catalog already knows the answer, so the loop should never see a request."""
    collection = maille.open_collection(served, "c")
    collection.cells  # noqa: B018 - reading the catalog is setup
    served.forget()

    async def attempt() -> None:
        await collection.aread_cells([(0, 999_999)])

    with pytest.raises(KeyError, match="holds no cell"):
        asyncio.run(attempt())
    assert served.calls == 0, "a cell the catalog does not name must cost no fetch at all"


def test_two_overlapping_reads_open_each_part_once(served: ConcurrentStore):
    """A viewer issues more than one read per frame, against one open collection.

    Without a guard both would find the part unopened, both would fetch its footer, and both
    would store a handle -- leaving the reader parsing through one file while prefetches prime
    the other. The output stays correct and every prefetch after that is wasted, which is a
    performance bug that no correctness test would ever notice.
    """
    reference = maille.open_collection(served, "c")
    keys = [(entry.level, entry.cell) for entry in reference.plan(error_budget=0.3)]
    half = len(keys) // 2
    expected = list(reference.read_cells(keys))

    async def both() -> tuple[list[maille.DecodedCell], int, int]:
        collection = await maille.aopen_collection(served, "c")
        first, second = await asyncio.gather(
            collection.aread_cells(keys[:half]), collection.aread_cells(keys[half:])
        )
        return first + second, len(collection._parquet_files), len(collection._part_files)

    got, parts, handles = asyncio.run(both())

    assert parts == handles, "a part was opened twice, so its prefetches prime a file nothing reads"
    for actual, wanted in zip(got, expected):
        assert np.array_equal(actual.vertices, wanted.vertices)


def test_the_native_async_methods_are_the_ones_used(many: maille.MeshCollection):
    """obstore has ``get_range_async``; maille must call it rather than thread its sync twin."""
    obstore = pytest.importorskip("obstore")

    store = obstore.store.MemoryStore()
    maille.write_collection(many, store, "c", row_group_bytes=ROW_GROUP_BYTES)
    seen: list[dict[str, Any]] = []
    native = store.get_range_async

    async def spy(path: str, **window: Any) -> Any:  # noqa: ANN401
        seen.append(window)
        return await native(path, **window)

    store.get_range_async = spy
    collection = maille.open_collection(store, "c")
    keys = [(entry.level, entry.cell) for entry in collection.plan(error_budget=0.3)]

    asyncio.run(_read(store, keys))

    assert seen, "no ranged async read was issued, so the sync path was taken instead"
    assert all("start" in window for window in seen)


async def _read(
    store: maille.MailleStore, keys: list[tuple[int, int]], *, concurrency: int = 16
) -> list[maille.DecodedCell]:
    """Open and read, the way a viewer would."""
    collection = await maille.aopen_collection(store, "c")
    return await collection.aread_cells(keys, concurrency=concurrency)


async def _one(store: maille.MailleStore, level: int, cell: int) -> maille.DecodedCell:
    """Open and read a single cell."""
    collection = await maille.aopen_collection(store, "c")
    return await collection.aread_cell(level, cell)
