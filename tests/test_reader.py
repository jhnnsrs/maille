"""Reading a collection back: the manifest, the catalogs, a decoded cell, a whole object."""

from __future__ import annotations

import json

import numpy as np
import pytest

import maille
from tests.conftest import AXES, CELL_SIZE, LEVELS


def test_opening_reads_the_manifest_and_nothing_else(written: maille.MemoryStore):
    """Reading is lazy in the order the format is: declarations, then index, then geometry."""

    class CountingStore(maille.MemoryStore):
        def __init__(self, objects: dict[str, bytes]) -> None:
            super().__init__()
            self.objects = dict(objects)
            self.reads: list[str] = []

        def get(self, path: str) -> bytes:
            self.reads.append(path)
            return super().get(path)

    store = CountingStore(written.objects)

    collection = maille.open_collection(store, "collection")
    assert store.reads == ["collection/meshed.json"]

    collection.cells  # noqa: B018 - the access is the point
    assert store.reads == ["collection/meshed.json", "collection/catalog/cells.parquet"]


def test_the_declarations_survive_the_round_trip(collection: maille.MeshCollection, opened: maille.Collection):
    """What the writer declared is what a renderer configures its decoder from."""
    assert opened.manifest.spec_version == maille.SPEC_VERSION
    assert opened.grid.cell_size == CELL_SIZE
    assert opened.grid.levels == LEVELS
    assert opened.axes == AXES
    assert opened.encoding.to_dict() == collection.manifest.encoding.to_dict()
    assert opened.manifest.counts["objects"] == collection.object_catalog.num_rows


def test_the_cell_catalog_answers_which_cells_without_opening_geometry(opened: maille.Collection):
    """The whole reason the spatial index is a separate file."""
    assert len(opened.cells) > 0
    assert {entry.level for entry in opened.cells.values()} == set(range(LEVELS))

    for entry in opened.cells.values():
        assert entry.index_count % 3 == 0
        assert entry.face_count * 3 == entry.index_count
        assert entry.object_count >= 1
        assert np.all(np.asarray(entry.bbox_min) <= np.asarray(entry.bbox_max))


def test_a_cell_decodes_into_the_geometry_it_declares(opened: maille.Collection):
    """The counts in the catalog are the counts in the blobs."""
    for entry in opened.cells.values():
        cell = opened.read_cell(entry.level, entry.cell)

        assert len(cell.vertices) == entry.vertex_count
        assert len(cell.faces) * 3 == entry.index_count
        assert len(cell.object_ids) == entry.object_count
        assert cell.faces.max() < len(cell.vertices), "an index points past the cell's vertices"


def test_a_cells_bounds_match_the_geometry_it_holds(opened: maille.Collection):
    """The catalog's bbox is what a planner culls on, so it has to be the real extent."""
    for entry in opened.cells.values():
        cell = opened.read_cell(entry.level, entry.cell)
        quantum = max(opened.grid.cell_extent(entry.level)) / maille.QUANT_MAX

        assert cell.vertices.min(axis=0) == pytest.approx(np.asarray(entry.bbox_min), abs=quantum * 2)
        assert cell.vertices.max(axis=0) == pytest.approx(np.asarray(entry.bbox_max), abs=quantum * 2)


def test_one_object_can_be_isolated_out_of_a_shared_cell(opened: maille.Collection):
    """Indices are cell-global, so isolating an object means re-basing them -- easy to get wrong."""
    shared = [entry for entry in opened.cells.values() if entry.object_count > 1]
    if not shared:
        pytest.skip("this fixture has no cell holding more than one object")

    entry = shared[0]
    cell = opened.read_cell(entry.level, entry.cell)

    total = 0
    for object_id in cell.object_ids:
        piece = cell.object_mesh(object_id)
        total += len(piece.faces)
        assert piece.faces.min() >= 0, "an isolated object's indices were not re-based"
        assert piece.faces.max() < len(piece.vertices), "an isolated object's index points outside its vertices"

    assert total == len(cell.faces), "the objects' faces do not add up to the cell's"


def test_asking_a_cell_for_an_object_it_does_not_hold_says_so(opened: maille.Collection):
    """A KeyError naming what is there beats an IndexError from a slice."""
    entry = next(iter(opened.cells.values()))
    cell = opened.read_cell(entry.level, entry.cell)

    with pytest.raises(KeyError, match="not in cell"):
        cell.object_mesh(999999)


def test_the_object_catalog_answers_where_an_object_is(opened: maille.Collection, objects: dict):
    """The inverted index doing its one job: a lookup rather than a scan."""
    assert set(opened.objects) == set(objects)

    for object_id, entry in opened.objects.items():
        assert entry.cells, f"object {object_id} is in the catalog but names no cells"
        for level, cell in entry.cells:
            assert (level, cell) in opened.cells, "the object catalog names a cell the cell catalog does not have"
        assert entry.cells_at(0), f"object {object_id} has no level-0 cells"


def test_an_object_is_reassembled_across_the_cells_that_hold_it(opened: maille.Collection, objects: dict):
    """The end-to-end form: lookup, one fetch per named cell, then weld."""
    for object_id, source in objects.items():
        reassembled = opened.object_mesh(object_id, level=0)
        quantum = max(CELL_SIZE) / maille.QUANT_MAX

        assert np.asarray(reassembled.bounds[0]) == pytest.approx(np.asarray(source.bounds[0]), abs=quantum * 4)
        assert np.asarray(reassembled.bounds[1]) == pytest.approx(np.asarray(source.bounds[1]), abs=quantum * 4)
        assert len(reassembled.faces) > 0


def test_an_object_is_present_at_every_level(opened: maille.Collection):
    """A level is a standalone representation, checked from the reader's side this time."""
    for object_id in opened.objects:
        for level in range(LEVELS):
            assert opened.cells_for_object(object_id, level=level), (
                f"object {object_id} disappears at level {level}, so a zoomed-out view would lose it"
            )


def test_asking_for_an_object_that_is_not_there_says_so(opened: maille.Collection):
    """Naming the collection's contents beats a bare KeyError."""
    with pytest.raises(KeyError, match="no object"):
        opened.cells_for_object(424242)


def test_children_are_read_off_the_mask_without_touching_the_catalog(opened: maille.Collection):
    """Descending has to be free, or the octree costs more than it saves."""
    for entry in opened.cells.values():
        for key in entry.children():
            assert key[0] == entry.level - 1
            assert key in opened.cells


def test_the_roots_are_the_coarsest_level(opened: maille.Collection):
    """Where a descent starts."""
    roots = opened.roots()

    assert roots, "a collection with levels has a coarsest level"
    assert all(entry.level == LEVELS - 1 for entry in roots)


def test_reading_a_cell_that_is_not_there_says_which_level_was_asked(opened: maille.Collection):
    """An error that names the level saves a round of guessing."""
    with pytest.raises(KeyError, match="Level 0 holds no cell"):
        opened.read_cell(0, 999999)


def test_a_collection_can_be_opened_at_the_root_of_a_store(collection: maille.MeshCollection):
    """A granted prefix is usually already rooted at the collection, so the prefix is empty."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store)

    opened = maille.open_collection(store)

    assert "meshed.json" in store.objects
    assert len(opened.cells) == collection.cell_catalog.num_rows


def test_a_prefix_with_no_manifest_at_all_is_refused(collection: maille.MeshCollection):
    """An empty prefix is an unfinished upload, not an empty collection."""
    with pytest.raises(maille.UnfinishedCollectionError):
        maille.open_collection(maille.MemoryStore(), "nothing-here")


def test_a_level_is_found_by_listing_when_the_manifest_does_not_name_its_parts(
    collection: maille.MeshCollection,
):
    """`files` is a claim, not authority -- the prefix listing is what actually holds.

    maille always writes `files.levels`, so this branch never runs on a tree maille produced.
    It is the branch that runs on everything else, which for a wire format is the case that
    matters: another writer may name its levels only by their paths.
    """
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "collection")

    stripped = json.loads(store.objects["collection/meshed.json"])
    stripped["files"] = {}
    store.objects["collection/meshed.json"] = json.dumps(stripped).encode()

    opened = maille.open_collection(store, "collection")

    assert opened.level_paths(0) == ["level=0/part-00000.parquet"], "the listing was not made relative again"
    assert opened.geometry(0).num_rows == collection.shards[0][1].num_rows

    entry = opened.cells_at(0)[0]
    assert len(opened.read_cell(entry.level, entry.cell).vertices) == entry.vertex_count


def test_the_listing_fallback_works_at_the_root_of_a_store(collection: maille.MeshCollection):
    """The empty-prefix case is where the path arithmetic is easiest to get off by one."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store)

    stripped = json.loads(store.objects["meshed.json"])
    stripped["files"] = {}
    store.objects["meshed.json"] = json.dumps(stripped).encode()

    opened = maille.open_collection(store)

    assert opened.level_paths(1) == ["level=1/part-00000.parquet"]
    assert opened.geometry(1).num_rows > 0


def test_a_level_with_nothing_stored_under_it_says_so(collection: maille.MeshCollection):
    """A gap the manifest does not admit to is still a gap, and it is named rather than empty."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "collection")

    stripped = json.loads(store.objects["collection/meshed.json"])
    stripped["files"] = {}
    store.objects["collection/meshed.json"] = json.dumps(stripped).encode()
    del store.objects["collection/level=2/part-00000.parquet"]

    opened = maille.open_collection(store, "collection")

    with pytest.raises(maille.FormatError, match="nothing is stored under"):
        opened.geometry(2)
