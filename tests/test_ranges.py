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

import json
import warnings
from typing import Any

import numpy as np
import pytest

import maille
from tests.conftest import AXES, CELL_SIZE, LEVELS, AccountingStore

#: Small enough that the fixture's level 0 lands in several row groups. The default is sized
#: for real collections, and a fixture big enough to split at the default would make the suite
#: slow to prove a point that the budget itself is what varies.
ROW_GROUP_BYTES = 16 * 1024


@pytest.fixture(scope="session")
def wide_objects() -> dict[int, Any]:
    """Enough geometry that a level part is comfortably larger than a Parquet footer window.

    The shared fixture collection is a few tens of kilobytes per level, which is *smaller than
    :attr:`maille.StoreFile.TAIL_BYTES`* -- so opening a part pulls the whole thing as its tail
    and every subsequent read is free. That is the right behaviour for a small part (one round
    trip beats two) and it makes a byte measurement meaningless: everything reads as zero.

    So these objects exist to put the part above that window, which is where a real collection
    lives and where row-group granularity is the thing actually being measured.
    """
    trimesh = pytest.importorskip("trimesh")
    return {
        1000 + index * 7: trimesh.creation.icosphere(radius=30.0, subdivisions=3).apply_translation(
            [60.0 + index * 70.0, 80.0, 60.0]
        )
        for index in range(24)
    }


@pytest.fixture(scope="session")
def wide(wide_objects: dict[int, Any]) -> maille.MeshCollection:
    """The larger collection, built once for the whole session."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return maille.build_collection(wide_objects, axes=AXES, cell_size=CELL_SIZE, levels=LEVELS)


@pytest.fixture()
def chunked(wide: maille.MeshCollection) -> AccountingStore:
    """The larger collection, written with row groups small enough to be worth locating."""
    store = AccountingStore()
    maille.write_collection(store=store, collection=wide, prefix="c", row_group_bytes=ROW_GROUP_BYTES)
    return store


def level_bytes(store: AccountingStore, level: int = 0) -> int:
    """How large one level is on disk -- the transfer a reader used to pay to draw one cell."""
    return sum(len(body) for path, body in store.objects.items() if f"level={level}" in path)


def row_groups(collection: maille.Collection, level: int = 0) -> int:
    """How many row groups a level was written in."""
    return sum(entry.row_groups or 0 for entry in collection.manifest.level_files(level) or [])


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


def test_reading_a_cell_costs_about_one_row_group(
    chunked: AccountingStore, measured: maille.Collection
):
    """The whole point, stated as the number it has to be.

    ``warm < level`` would be satisfied by a read of nearly the entire level, so it is not the
    assertion this file's premise deserves. The bound is an *even share*: a level in ``n`` row
    groups should cost about a ``1/n`` of it to read one cell out of, and the slack below is
    for row groups that came out uneven because they are budgeted on uncompressed blob bytes
    and measured after compression.

    **The cold number is the load-bearing one.** A reader that fetched the whole level and
    cached it would report a warm read of zero -- perfectly cheap, and completely wrong. Only
    the first read of a fresh collection can tell the two apart, so that is what is bounded
    strictly here and the warm number is checked as a consequence rather than as the proof.
    """
    entry = a_level_zero_cell(measured)
    whole, groups = level_bytes(chunked), row_groups(measured)
    share = whole / groups

    measured.read_cell(entry.level, entry.cell)
    cold = chunked.bytes_read()
    chunked.forget()

    measured.read_cell(entry.level, entry.cell)
    warm = chunked.bytes_read()

    assert cold < whole, (
        f"the first read of a cell moved {cold} bytes against a level of {whole}. Fetching the "
        f"level and caching it would look free on every later read and cost this on the first, "
        f"which is exactly the behaviour the locator replaced."
    )
    assert warm <= cold, "the second read must not re-parse the footer the first one parsed"
    assert warm < 2 * share, (
        f"a warm cell read moved {warm} bytes where an even row-group share of this level is "
        f"{share:.0f} ({whole} bytes in {groups} row groups)"
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
    batched_reads, batched_bytes = len(chunked.reads), chunked.bytes_read()

    one_at_a_time = maille.open_collection(chunked, "c")
    one_at_a_time.cells  # noqa: B018
    chunked.forget()
    serial = [one_at_a_time.read_cell(level, cell) for level, cell in keys]
    serial_reads, serial_bytes = len(chunked.reads), chunked.bytes_read()

    assert batched_reads < serial_reads, (
        f"grouping by row group made {batched_reads} reads where reading one cell at a time made "
        f"{serial_reads}. Equality means the grouping collapsed to a per-cell loop, which is the "
        f"regression this test exists for."
    )
    assert batched_bytes <= serial_bytes
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


def test_every_cell_reads_back_as_the_size_its_catalog_row_claims(measured: maille.Collection):
    """Read every cell through its locator and check it against what the catalog said.

    Note what is *not* asserted: that ``decoded.cell == entry.cell``. The reader sets that from
    the entry it was handed, so it is true by construction and would stay true if the locator
    pointed somewhere else entirely. The counts are the discriminating part -- a locator that
    lands on a neighbouring cell hands back a blob of a different size.
    """
    for entry in sorted(measured.cells.values(), key=lambda e: (e.level, e.cell)):
        decoded = measured.read_cell(entry.level, entry.cell)

        assert len(decoded.vertices) == entry.vertex_count
        assert len(decoded.faces) * 3 == entry.index_count
        assert set(decoded.object_ids) and len(decoded.object_ids) == entry.object_count


def test_a_locator_pointing_at_the_wrong_row_group_is_caught(chunked: AccountingStore):
    """The guard for a catalog that disagrees with the geometry, which nothing else reaches.

    A wrong ``row_group`` is the quiet failure this whole mechanism could have: it does not
    raise on its own, it hands back some other cell's blob, and that blob decodes perfectly
    well against the wrong box. So the reader checks that the row group it fetched actually
    contains the cell it asked for -- and this is the test that the check is reachable, by
    rewriting one catalog row to point at a different group.
    """
    import pyarrow as pa

    from maille.frames import parquet_to_table, table_to_parquet

    catalog = parquet_to_table(chunked.objects["c/catalog/cells.parquet"])
    groups = catalog.column("row_group").to_pylist()
    levels = catalog.column("level").to_pylist()
    row = next(i for i, (level, group) in enumerate(zip(levels, groups)) if level == 0 and group == 0)
    moved = [max(groups) if index == row else group for index, group in enumerate(groups)]
    corrupted = catalog.set_column(
        catalog.schema.get_field_index("row_group"),
        catalog.schema.field("row_group"),
        pa.array(moved, type=pa.int32()),
    )
    chunked.objects["c/catalog/cells.parquet"] = table_to_parquet(corrupted)

    opened = maille.open_collection(chunked, "c")
    entry = opened.cells[(int(levels[row]), int(catalog.column("cell").to_pylist()[row]))]

    with pytest.raises(maille.FormatError, match="catalog and the geometry disagree"):
        opened.read_cell(entry.level, entry.cell)


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
