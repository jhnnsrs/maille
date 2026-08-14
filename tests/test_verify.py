"""Checking the checker: every corruption must fail its own check and no others.

A verifier that passes everything is worthless and a verifier that fails everything is worse,
so the tests here are mostly *targeted damage*. Break one thing, and assert two properties:
the check that owns that claim fails, and the ones that do not own it still pass. The second
half is the one that catches a verifier quietly reporting noise.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

import maille
from maille.frames import parquet_to_table, table_to_parquet
from maille.verify import TIERS, verify
from tests.conftest import CELL_SIZE, LEVELS


@pytest.fixture()
def sound(collection: maille.MeshCollection) -> maille.MemoryStore:
    """A collection written the way the writer writes it, and therefore expected to pass."""
    store = maille.MemoryStore()
    maille.write_collection(collection, store, "c")
    return store


def opened(store: maille.MemoryStore) -> maille.Collection:
    """The collection under test."""
    return maille.open_collection(store, "c")


def failed(store: maille.MemoryStore, *, tier: str = "blobs") -> set[str]:
    """The names of the checks that did not pass."""
    return {check.name for check in verify(opened(store), tier=tier).failures}


def edit_catalog(store: maille.MemoryStore, column: str, values: list[Any]) -> None:
    """Rewrite one column of the cell catalog in place."""
    table = parquet_to_table(store.objects["c/catalog/cells.parquet"])
    replaced = table.set_column(
        table.schema.get_field_index(column), table.schema.field(column), pa.array(values, type=table.schema.field(column).type)
    )
    store.objects["c/catalog/cells.parquet"] = table_to_parquet(replaced)


def edit_geometry(store: maille.MemoryStore, path: str, column: str, values: list[Any]) -> None:
    """Rewrite one column of a geometry part, keeping the manifest's byte length honest.

    Re-serializing changes the file's length, and the manifest records that length so a reader
    can seek to the footer. Leaving it stale corrupts a *second* thing -- and the test would
    then be checking whichever failure the verifier happened to notice first, rather than the
    one it set out to plant.
    """
    key = f"c/{path}"
    table = parquet_to_table(store.objects[key])
    replaced = table.set_column(
        table.schema.get_field_index(column), table.schema.field(column), pa.array(values, type=table.schema.field(column).type)
    )
    body = table_to_parquet(replaced)
    store.objects[key] = body

    manifest = json.loads(store.objects["c/maille.json"])
    for entries in manifest["files"]["levels"].values():
        for entry in entries:
            if entry["path"] == path:
                entry["bytes"] = len(body)
                entry["rowGroups"] = 1
    store.objects["c/maille.json"] = json.dumps(manifest).encode()


@pytest.mark.parametrize("tier", TIERS)
def test_a_collection_the_writer_wrote_passes_every_tier(sound: maille.MemoryStore, tier: str):
    """The baseline. Without it, every failure below could just be a broken verifier."""
    report = verify(opened(sound), tier=tier)

    assert report.ok, f"a freshly written collection failed its own checks:\n{report}"
    assert report.checks, "a report with no checks in it is not a pass"
    assert bool(report) is True


def test_the_tiers_are_cumulative_and_cost_more_as_they_go(sound: maille.MemoryStore):
    """A caller picks a tier by what it can afford, so each must include the cheaper ones."""
    collection = opened(sound)

    counts = [len(verify(collection, tier=tier).checks) for tier in TIERS]

    assert counts == sorted(counts), f"tiers should only add checks, got {dict(zip(TIERS, counts))}"
    assert counts[0] < counts[-1], "the deepest tier must actually check more than the cheapest"


def test_an_unknown_tier_is_refused(sound: maille.MemoryStore):
    """A typo must not silently run the cheapest tier and report a pass."""
    with pytest.raises(ValueError, match="tier"):
        verify(opened(sound), tier="thorough")


def test_a_missing_file_is_reported_rather_than_raised(sound: maille.MemoryStore):
    """The manifest is a promise about the tree; this is the check that it was kept."""
    del sound.objects["c/level=1/part-00000.parquet"]

    assert "files exist" in failed(sound, tier="structure")


def test_a_cleared_child_mask_bit_is_caught(sound: maille.MemoryStore):
    """A cleared bit hides a child and everything below it, with nothing raised anywhere."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    masks = table.column("child_mask").to_pylist()
    levels = table.column("level").to_pylist()
    target = next(i for i, (level, mask) in enumerate(zip(levels, masks)) if level > 0 and mask)
    # Turn off one set bit, which is the quietest possible corruption of the octree.
    masks[target] &= masks[target] - 1
    edit_catalog(sound, "child_mask", masks)

    assert failed(sound, tier="structure") == {"child masks name children that exist"}


def test_an_inverted_lod_error_is_caught(sound: maille.MemoryStore):
    """A parent more accurate than its child stops a planner descending to the detail it wanted."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    levels = table.column("level").to_pylist()
    errors = table.column("lod_error").to_pylist()
    target = next(i for i, level in enumerate(levels) if level == LEVELS - 1)
    errors[target] = 0.0
    edit_catalog(sound, "lod_error", errors)

    assert "a parent's error dominates its children's" in failed(sound, tier="structure")


def test_a_missing_locator_is_caught(sound: maille.MemoryStore):
    """A catalog row with no locator is a cell a reader has to go looking for."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    parts = table.column("part").to_pylist()
    parts[0] = None
    edit_catalog(sound, "part", parts)

    assert "every cell carries a locator" in failed(sound, tier="structure")


def test_a_locator_pointing_at_a_part_that_does_not_exist_is_caught(sound: maille.MemoryStore):
    """Caught from the manifest alone, so registration rejects it rather than the first viewer."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    parts = table.column("part").to_pylist()
    parts[0] = 99
    edit_catalog(sound, "part", parts)

    assert "locators point inside the files that exist" in failed(sound, tier="structure")


def test_a_row_group_past_the_end_of_its_part_is_caught(sound: maille.MemoryStore):
    """The manifest records how many row groups a part has, so this is answerable for free."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    groups = table.column("row_group").to_pylist()
    groups[0] = 9_999
    edit_catalog(sound, "row_group", groups)

    assert "locators point inside the files that exist" in failed(sound, tier="structure")


def test_an_object_catalog_naming_a_cell_that_does_not_exist_is_caught(sound: maille.MemoryStore):
    """The identity index is only an index if what it points at is there."""
    table = parquet_to_table(sound.objects["c/catalog/objects.parquet"])
    cells = table.column("cells").to_pylist()
    cells[0] = [*cells[0], {"level": 0, "cell": 12_345_678}]
    replaced = table.set_column(
        table.schema.get_field_index("cells"),
        table.schema.field("cells"),
        pa.array(cells, type=table.schema.field("cells").type),
    )
    sound.objects["c/catalog/objects.parquet"] = table_to_parquet(replaced)

    assert failed(sound, tier="structure") == {"the object catalog names cells that exist"}


def test_a_wrong_vertex_count_is_caught_when_the_blob_is_decoded(sound: maille.MemoryStore):
    """Structure cannot see this and blobs must: the count is a claim about bytes."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    counts = table.column("vertex_count").to_pylist()
    counts[0] = counts[0] + 7
    edit_catalog(sound, "vertex_count", counts)

    assert verify(opened(sound), tier="structure").ok, "no structural claim was touched"
    assert "every blob decodes to the counts its row claims" in failed(sound, tier="blobs")


def test_a_truncated_blob_is_caught(sound: maille.MemoryStore):
    """The failure a codec produces is garbage, not an exception, so something must ask."""
    table = parquet_to_table(sound.objects["c/level=0/part-00000.parquet"])
    positions = table.column("positions").to_pylist()
    positions[0] = positions[0][: len(positions[0]) // 2]
    edit_geometry(sound, "level=0/part-00000.parquet", "positions", positions)

    assert "every blob decodes to the counts its row claims" in failed(sound, tier="blobs")


def test_a_geometry_row_holding_an_unknown_object_is_caught(sound: maille.MemoryStore):
    """The forward and inverted identity indexes are written from one source and must agree."""
    table = parquet_to_table(sound.objects["c/level=0/part-00000.parquet"])
    ids = table.column("object_ids").to_pylist()
    ids[0] = [*ids[0][:-1], 4_711]
    edit_geometry(sound, "level=0/part-00000.parquet", "object_ids", ids)

    assert "the two catalogs agree about who is in which cell" in failed(sound, tier="blobs")


def test_the_geometry_checks_actually_compared_something(sound: maille.MemoryStore):
    """A check that compared nothing passes for the wrong reason.

    Both cross-level checks count what they looked at, and both would report a serene pass on a
    collection where they found no pairs to compare. Since these are the checks the module
    exists for, the counters are asserted rather than trusted.
    """
    report = verify(opened(sound), tier="geometry")
    detail = {check.name: check.detail for check in report.checks}

    pinned = detail["on-plane vertices are held fixed across levels (boundary: LOCKED)"]
    bounded = detail["lod_error bounds how far a vertex moved from level 0"]

    assert not pinned.startswith("0 "), f"the LOCKED check compared nothing: {pinned}"
    assert not bounded.startswith("0 "), f"the lod_error check compared nothing: {bounded}"


def test_a_coarse_level_larger_than_the_one_below_is_caught(sound: maille.MemoryStore):
    """The inversion the writer warns about: zooming out fetches more and draws worse."""
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    levels = table.column("level").to_pylist()
    counts = table.column("index_count").to_pylist()
    counts = [count * 100 if level == LEVELS - 1 else count for level, count in zip(levels, counts)]
    edit_catalog(sound, "index_count", counts)

    assert "a coarse level holds less than the level it summarises" in failed(sound, tier="geometry")


def test_an_lod_error_too_small_to_be_true_is_caught(sound: maille.MemoryStore):
    """``lod_error`` is what a planner spends, so a collection understating it draws blurred.

    Only the finer levels are shrunk, so the parent-dominance check stays satisfied and this
    failure lands on the check that owns it rather than on its neighbour.
    """
    table = parquet_to_table(sound.objects["c/catalog/cells.parquet"])
    levels = table.column("level").to_pylist()
    errors = table.column("lod_error").to_pylist()
    errors = [1e-9 if 0 < level < LEVELS - 1 else error for level, error in zip(levels, errors)]
    edit_catalog(sound, "lod_error", errors)

    assert "lod_error bounds how far a vertex moved from level 0" in failed(sound, tier="geometry")


def test_a_boundary_vertex_that_moved_is_caught(sound: maille.MemoryStore):
    """The claim the whole format rests on: a vertex on a cell face plane did not move.

    Nothing downstream can see this -- a renderer looks at one cell and a crack only appears
    where two levels meet -- so it is checked by displacing one pinned vertex and requiring the
    verifier to notice.
    """
    from maille.codecs import decode_positions, encode_positions
    from maille.geometry import on_planes

    collection = opened(sound)

    # Search from the coarsest level down for one that has a pinned vertex at all. The very
    # coarsest often does not: when the whole collection fits in one cell there is nothing
    # lying on that cell's faces, and the claim has no content there.
    for level in range(LEVELS - 1, 0, -1):
        path = f"level={level}/part-00000.parquet"
        table = parquet_to_table(sound.objects[f"c/{path}"])
        cells = table.column("cell").to_pylist()
        counts = table.column("vertex_count").to_pylist()
        positions = table.column("positions").to_pylist()
        for row, cell in enumerate(cells):
            origin, extent = maille.cell_box(cell, level, collection.grid.cell_size)
            vertices = decode_positions(
                positions[row],
                cell=cell,
                level=level,
                cell_size=collection.grid.cell_size,
                codec=collection.encoding.codec,
                vertex_count=counts[row],
            )
            pinned = on_planes(vertices, extent, tolerance=float(extent.max()) / maille.QUANT_MAX)
            if not pinned.any():
                continue
            # Several quanta, so the displacement cannot be mistaken for the documented
            # residual between two levels' quantization grids -- and *inward*, because a
            # vertex on a face pushed outward leaves the cell entirely and the encoder
            # rejects it as a partitioning bug before the verifier ever sees it.
            target = pinned.argmax()
            centre = origin + extent / 2.0
            vertices[target] += np.sign(centre - vertices[target]) * (extent / 64.0)
            positions[row] = encode_positions(
                vertices, cell=cell, level=level, cell_size=collection.grid.cell_size, codec=collection.encoding.codec
            )
            edit_geometry(sound, path, "positions", positions)
            break
        else:
            continue
        break
    else:
        pytest.fail("the fixture has no pinned vertex at any level to displace")

    assert "on-plane vertices are held fixed across levels (boundary: LOCKED)" in failed(sound, tier="geometry")


def test_the_report_reads_as_something_a_person_would_read(sound: maille.MemoryStore):
    """A report nobody can read is a report nobody reads."""
    del sound.objects["c/level=1/part-00000.parquet"]

    report = verify(opened(sound), tier="structure")

    assert not report
    assert "FAIL" in str(report)
    assert "files exist" in str(report)
    assert any(check.examples for check in report.failures), "a failure should name what failed"


def test_a_single_level_collection_verifies(objects: dict[int, Any]):
    """One level means no cross-level claims to check, which must be a pass and not a crash."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = maille.build_collection(objects, cell_size=CELL_SIZE, levels=1)
    store = maille.MemoryStore()
    maille.write_collection(built, store, "c")

    report = verify(maille.open_collection(store, "c"), tier="geometry")

    assert report.ok, str(report)


def test_the_manifest_and_the_report_agree_on_which_files_exist(sound: maille.MemoryStore):
    """``verify_paths`` is what a caller iterates to copy or check a collection by hand."""
    from maille.verify import verify_paths

    paths = verify_paths(opened(sound))

    assert set(paths) == {path.removeprefix("c/") for path in sound.objects if path != "c/maille.json"}
    assert json.loads(sound.objects["c/maille.json"])["files"]["cells"]["path"] in paths
