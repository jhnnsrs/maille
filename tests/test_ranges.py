"""Fetching one cell instead of its level: the locator, the row group, and what it costs.

This file is about a *number* rather than a shape. Everything else in the suite checks that
what comes back is right; these check how much had to move to get it, because "a viewer
fetches the detail the view needs instead of the whole thing" is the claim the format is for,
and it is the one claim that stays true-looking while silently regressing.

The measurements are split cold from warm on purpose. A cold read pays for the Parquet footer
of the part it opens; a warm one does not, because the footer is parsed once per part and
reused. Only the warm number is "a row group's worth", and a test that adds the two together
passes just as happily when the footer has grown to dominate the transfer -- which is exactly
how chunking too finely fails.
"""

from __future__ import annotations

import numpy as np
import pytest

import maille
from tests.conftest import AXES, CELL_SIZE, LEVELS, AccountingStore

#: Small enough that the fixture's level 0 lands in several row groups. The default is sized
#: for real collections, and a fixture big enough to split at the default would make the suite
#: slow to prove a point that the budget itself is what varies.
ROW_GROUP_BYTES = 4 * 1024


@pytest.fixture()
def chunked(collection: maille.MeshCollection) -> AccountingStore:
    """The fixture collection, written with row groups small enough to be worth locating."""
    store = AccountingStore()
    maille.write_collection(collection, store, "c", row_group_bytes=ROW_GROUP_BYTES)
    return store


@pytest.fixture()
def measured(chunked: AccountingStore) -> maille.Collection:
    """The chunked collection, opened, with the catalogs already read and the record cleared."""
    opened = maille.open_collection(chunked, "c")
    opened.cells  # noqa: B018 - the catalog read is setup, not what is being measured
    chunked.forget()
    return opened


def a_level_zero_cell(collection: maille.Collection) -> maille.CellEntry:
    """The first level-0 cell in Morton order, which is a cell like any other."""
    return min((entry for entry in collection.cells.values() if entry.level == 0), key=lambda e: e.cell)


def test_the_level_actually_splits_into_row_groups(chunked: AccountingStore):
    """Every measurement below is vacuous if the level came out as one row group."""
    manifest = maille.open_collection(chunked, "c").manifest
    parts = manifest.level_files(0) or []

    assert sum(entry.row_groups or 0 for entry in parts) > 1, (
        "this fixture is supposed to exercise the locator, and a single row group per level "
        "would make every 'one row group' assertion below trivially equal to 'the whole level'"
    )


def test_reading_a_cell_costs_its_row_group_rather_than_its_level(
    chunked: AccountingStore, measured: maille.Collection
):
    """The whole point: a cell is fetched, not the level containing it."""
    entry = a_level_zero_cell(measured)
    level_bytes = sum(len(body) for path, body in chunked.objects.items() if "level=0" in path)

    measured.read_cell(entry.level, entry.cell)
    cold = chunked.bytes_read()
    chunked.forget()

    measured.read_cell(entry.level, entry.cell)
    warm = chunked.bytes_read()

    assert warm < cold, "the second read reuses the footer the first one parsed"
    assert warm < level_bytes, "a warm cell read must not cost the level"
    assert cold <= level_bytes, (
        f"a cold read moved {cold} bytes against a level of {level_bytes}: opening the part cost "
        f"more than downloading it, which is what chunking too finely looks like"
    )


def test_a_located_read_beats_reading_the_level_whole(
    chunked: AccountingStore, measured: maille.Collection
):
    """The comparison against the fallback, which is the thing the locator replaced."""
    entry = a_level_zero_cell(measured)

    measured.read_cell(entry.level, entry.cell)
    chunked.forget()
    measured.read_cell(entry.level, entry.cell)
    located = chunked.bytes_read()

    chunked.forget()
    measured.geometry(0)
    whole = chunked.bytes_read()

    assert located < whole, f"a located read moved {located} bytes, reading the level moved {whole}"


def test_reading_a_plan_shares_a_row_group_between_the_cells_in_it(
    chunked: AccountingStore, measured: maille.Collection
):
    """A plan is Morton-ordered and so is the geometry, so its cells cluster into row groups.

    ``read_cells`` exists to spend that adjacency: cells sharing a row group are decoded from
    one read of it. Reading the same keys one at a time is the baseline it has to beat.
    """
    keys = [(entry.level, entry.cell) for entry in measured.plan(error_budget=0.4)]
    assert len(keys) > 1, "the fixture must plan more than one cell for this to mean anything"

    batched = list(measured.read_cells(keys))
    batched_reads = len(chunked.reads)

    one_at_a_time = maille.open_collection(chunked, "c")
    one_at_a_time.cells  # noqa: B018
    chunked.forget()
    serial = [one_at_a_time.read_cell(level, cell) for level, cell in keys]

    assert batched_reads <= len(chunked.reads), (
        f"grouping by row group made {batched_reads} reads where reading one cell at a time made "
        f"{len(chunked.reads)}; the grouping is not paying for itself"
    )
    for got, expected in zip(batched, serial):
        assert np.array_equal(got.vertices, expected.vertices)
        assert np.array_equal(got.faces, expected.faces)


def test_read_cells_yields_in_the_order_it_was_asked_for(measured: maille.Collection):
    """Grouping is an implementation detail, so it must not reorder the caller's selection."""
    keys = [(entry.level, entry.cell) for entry in measured.plan(error_budget=0.4)]
    reversed_keys = list(reversed(keys))

    got = [(cell.level, cell.cell) for cell in measured.read_cells(reversed_keys)]

    assert got == reversed_keys


def test_every_cell_with_geometry_is_located(measured: maille.Collection):
    """A catalog row without a locator is a cell a reader has to go looking for."""
    unlocated = [
        (entry.level, entry.cell)
        for entry in measured.cells.values()
        if entry.part is None or entry.row_group is None
    ]

    assert not unlocated, f"these cells carry no locator: {unlocated[:5]}"


def test_a_locator_points_at_the_cell_that_claims_it(measured: maille.Collection):
    """The locator is only useful if it is right, and a wrong one returns the wrong geometry.

    Nothing about a wrong ``row_group`` raises on its own -- it hands back a neighbouring
    cell's blob, which decodes perfectly well against the wrong box. So the check is that every
    cell read through its locator comes back as itself.
    """
    for entry in sorted(measured.cells.values(), key=lambda e: (e.level, e.cell)):
        decoded = measured.read_cell(entry.level, entry.cell)

        assert (decoded.level, decoded.cell) == (entry.level, entry.cell)
        assert len(decoded.vertices) == entry.vertex_count
        assert len(decoded.faces) * 3 == entry.index_count


def test_blob_bytes_is_what_the_cell_actually_carries(measured: maille.Collection):
    """A planner budgets a fetch with this, so it has to be the encoded size, not a guess."""
    table = measured.geometry(0)
    sizes = {
        int(cell): len(positions or b"") + len(indices or b"")
        for cell, positions, indices in zip(
            table.column("cell").to_pylist(),
            table.column("positions").to_pylist(),
            table.column("indices").to_pylist(),
        )
    }

    for entry in measured.cells.values():
        if entry.level == 0:
            assert entry.blob_bytes == sizes[entry.cell]


def test_a_collection_whose_manifest_records_no_length_still_reads(collection: maille.MeshCollection):
    """A hand-written manifest names paths and nothing else; that must be slow, not broken.

    The recorded byte length is what lets a reader seek to a Parquet footer without being able
    to stat the object. Without it there is nothing to seek against, so the part is read whole
    -- which is what maille did everywhere before locators existed.
    """
    import json

    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")

    manifest = json.loads(store.objects["c/meshed.json"])
    manifest["files"]["levels"] = {
        level: [entry["path"] for entry in entries]
        for level, entries in manifest["files"]["levels"].items()
    }
    manifest["files"]["cells"] = manifest["files"]["cells"]["path"]
    manifest["files"]["objects"] = manifest["files"]["objects"]["path"]
    store.objects["c/meshed.json"] = json.dumps(manifest).encode()

    opened = maille.open_collection(store, "c")
    entry = a_level_zero_cell(opened)
    decoded = opened.read_cell(entry.level, entry.cell)

    assert len(decoded.vertices) == entry.vertex_count


def test_a_store_that_cannot_range_read_still_reads(collection: maille.MeshCollection):
    """``get_range`` is optional, and its absence degrades rather than fails."""

    class NoRanges:
        """A store offering only the three methods maille requires, and not one more."""

        def __init__(self) -> None:
            self.inner = maille.MemoryStore()

        def put(self, path: str, data: object) -> None:
            self.inner.put(path, data)

        def get(self, path: str) -> bytes:
            return self.inner.get(path)

        def list(self, prefix: str | None = None):
            return self.inner.list(prefix)

    store = NoRanges()
    maille.write_collection(collection, store, "c")

    opened = maille.open_collection(store, "c")
    entry = a_level_zero_cell(opened)
    decoded = opened.read_cell(entry.level, entry.cell)

    assert len(decoded.vertices) == entry.vertex_count


def test_releasing_drops_the_geometry_and_keeps_the_catalogs(measured: maille.Collection):
    """A long-lived viewer must be able to let go of levels without re-opening the collection."""
    measured.geometry(0)
    entry = a_level_zero_cell(measured)
    measured.read_cell(entry.level, entry.cell)

    measured.release()

    assert not measured._level_tables and not measured._parquet_files
    assert measured.cells, "the spatial index is small and a planner needs it every frame"
    assert len(measured.read_cell(entry.level, entry.cell).vertices) == entry.vertex_count


def test_the_round_trip_is_unchanged_by_how_finely_it_was_chunked(collection: maille.MeshCollection):
    """Row-group size is a transfer-shaping knob and must not touch the geometry."""
    coarse, fine = maille.MemoryStore(), maille.MemoryStore()
    maille.write_collection(collection, coarse, "c", row_group_bytes=1024 * 1024)
    maille.write_collection(collection, fine, "c", row_group_bytes=1024)

    one, other = maille.open_collection(coarse, "c"), maille.open_collection(fine, "c")

    assert set(one.cells) == set(other.cells)
    for key in sorted(one.cells):
        left, right = one.read_cell(*key), other.read_cell(*key)
        assert np.array_equal(left.vertices, right.vertices)
        assert np.array_equal(left.faces, right.faces)
        assert left.object_ids == right.object_ids


def test_the_axes_and_the_grid_survive_a_chunked_write(chunked: AccountingStore):
    """The declarations are the collection's identity; chunking is about bytes, not meaning."""
    opened = maille.open_collection(chunked, "c")

    assert opened.axes == AXES
    assert opened.grid.cell_size == CELL_SIZE
    assert opened.grid.levels == LEVELS
