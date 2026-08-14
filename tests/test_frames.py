"""The Parquet layer: the schemas, the required columns, and the checks before a write."""

from __future__ import annotations

import pytest

import maille
from maille.frames import arrow_schemas, parquet_to_table, table_to_parquet


def test_every_role_declares_the_columns_the_format_requires():
    """The column layer is the contract: nothing downstream opens a blob."""
    schemas = arrow_schemas()

    for role, required in maille.REQUIRED_COLUMNS.items():
        assert set(required) <= set(schemas[role].names), f"the {role} schema omits a required column"


def test_the_column_types_are_the_ones_a_describe_would_print():
    """Name *and* type are compared by a server, so a Morton code stored as text is refused."""
    pa = pytest.importorskip("pyarrow")
    schemas = arrow_schemas()

    cells = schemas["cell_catalog"]
    assert cells.field("level").type == pa.int32()
    assert cells.field("cell").type == pa.int64(), "a Morton code needs 53 bits"
    assert cells.field("lod_error").type == pa.float64()
    assert cells.field("child_mask").type == pa.uint8(), "eight children, eight bits"

    geometry = schemas["geometry"]
    assert geometry.field("positions").type == pa.binary()
    assert geometry.field("indices").type == pa.binary()
    assert geometry.field("object_ids").type == pa.list_(pa.int64())

    objects = schemas["object_catalog"]
    assert objects.field("cells").type == pa.list_(
        pa.struct([pa.field("level", pa.int32()), pa.field("cell", pa.int64())])
    )


def test_the_built_frames_match_the_declared_schemas(collection: maille.MeshCollection):
    """What the builder produces is what the schemas promise."""
    schemas = arrow_schemas()

    assert collection.cell_catalog.schema.equals(schemas["cell_catalog"])
    assert collection.object_catalog.schema.equals(schemas["object_catalog"])
    for _, shard in collection.shards:
        assert shard.schema.equals(schemas["geometry"])


def test_extra_columns_are_allowed(collection: maille.MeshCollection):
    """A writer may carry a denormalized attribute copy alongside; the check is presence-only."""
    pa = pytest.importorskip("pyarrow")
    widened = collection.cell_catalog.append_column(
        "colour", pa.array(["red"] * collection.cell_catalog.num_rows)
    )

    maille.validate_columns(widened, "cell_catalog")  # must not raise


def test_a_missing_column_is_named_in_the_error(collection: maille.MeshCollection):
    """The message has to say which column and which role, or it costs a debugging round."""
    narrowed = collection.cell_catalog.drop_columns(["child_mask"])

    with pytest.raises(maille.FormatError) as failure:
        maille.validate_columns(narrowed, "cell_catalog")

    assert "child_mask" in str(failure.value)
    assert "cell catalog" in str(failure.value)


def test_an_unknown_role_is_refused():
    """Three roles, and a typo in one is caught rather than silently checking nothing."""
    with pytest.raises(maille.FormatError, match="not a role"):
        maille.validate_columns(None, "geometries")


def test_parquet_round_trips_the_frames(collection: maille.MeshCollection):
    """Including the binary blob columns, which are the ones with anything to lose."""
    restored = parquet_to_table(table_to_parquet(collection.shards[0][1]))

    assert restored.schema.equals(collection.shards[0][1].schema)
    assert restored.to_pylist() == collection.shards[0][1].to_pylist()


def test_an_empty_level_still_produces_a_valid_frame():
    """A genuinely empty level is written out rather than skipped: a gap is never asked for."""
    schemas = arrow_schemas()
    from maille.frames import build_table

    empty = build_table([], schemas["geometry"])

    assert empty.num_rows == 0
    maille.validate_columns(empty, "geometry")
    assert parquet_to_table(table_to_parquet(empty)).num_rows == 0
