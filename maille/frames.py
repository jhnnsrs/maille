"""The three Parquet schemas, and the columns each role must carry.

Nothing that reads a collection renders these files itself: a server checks the columns and
the declarations and never opens a blob. So the column layer is the contract, and everything
below it is defined by :mod:`maille.codec` and nowhere else.

Extra columns are allowed on purpose -- a writer may carry a denormalized attribute copy
alongside -- so a check tests that the required columns are present, never that no others are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from maille.errors import FormatError, MissingExtraError

#: The columns each role must carry, used to check a frame before an upload is spent on it.
#: A server checks the same names (plus their types) with a DuckDB ``DESCRIBE``; this is the
#: writer's half, so a malformed frame fails before it is written to a store.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "cell_catalog": (
        "level", "cell", "vertex_count", "index_count",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "lod_error", "object_count", "child_mask",
        "part", "row_group", "blob_bytes",
    ),
    "object_catalog": (
        "object_id", "ordinal",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "vertex_count", "index_count", "cells",
    ),
    "geometry": (
        "level", "cell", "positions", "indices", "vertex_count", "index_count",
        "object_ids", "object_ordinals", "object_vertex_offsets", "object_index_offsets",
    ),
}


def require_pyarrow() -> Any:  # noqa: ANN401
    """Import pyarrow, which the format needs to exist at all."""
    try:
        import pyarrow  # type: ignore
    except ImportError as error:  # pragma: no cover - pyarrow is a hard dependency
        raise MissingExtraError(
            "pyarrow is required: a mesh collection is Parquet. Install it with `pip install maille`."
        ) from error
    return pyarrow


def arrow_schemas() -> dict[str, Any]:
    """The three Arrow schemas, spelled so a DuckDB ``DESCRIBE`` prints what a server accepts."""
    pa = require_pyarrow()
    bbox = [
        pa.field(f"bbox_{corner}_{axis}", pa.float64())
        for corner in ("min", "max")
        for axis in ("x", "y", "z")
    ]
    return {
        "cell_catalog": pa.schema([
            pa.field("level", pa.int32()),
            pa.field("cell", pa.int64()),
            pa.field("vertex_count", pa.int32()),
            pa.field("index_count", pa.int32()),
            *bbox,
            pa.field("lod_error", pa.float64()),
            pa.field("object_count", pa.int32()),
            pa.field("child_mask", pa.uint8()),
            # The locator: which part of the level holds this cell, and which row group of it.
            # Null on a built-but-unwritten collection -- part assignment and row-group
            # boundaries are only known once the geometry has actually been serialized, so the
            # writer fills these in and nothing before it can.
            pa.field("part", pa.int32(), nullable=True),
            pa.field("row_group", pa.int32(), nullable=True),
            pa.field("blob_bytes", pa.int64(), nullable=True),
        ]),
        "object_catalog": pa.schema([
            pa.field("object_id", pa.int64()),
            pa.field("ordinal", pa.int32()),
            *bbox,
            pa.field("vertex_count", pa.int32()),
            pa.field("index_count", pa.int32()),
            pa.field("cells", pa.list_(pa.struct([pa.field("level", pa.int32()), pa.field("cell", pa.int64())]))),
        ]),
        "geometry": pa.schema([
            pa.field("level", pa.int32()),
            pa.field("cell", pa.int64()),
            pa.field("positions", pa.binary()),
            pa.field("indices", pa.binary()),
            pa.field("vertex_count", pa.int32()),
            pa.field("index_count", pa.int32()),
            pa.field("object_ids", pa.list_(pa.int64())),
            pa.field("object_ordinals", pa.list_(pa.int32())),
            pa.field("object_vertex_offsets", pa.list_(pa.int32())),
            pa.field("object_index_offsets", pa.list_(pa.int32())),
        ]),
    }


def build_table(rows: Sequence[Mapping[str, Any]], schema: Any) -> Any:  # noqa: ANN401
    """Build an Arrow table from row dicts under a fixed schema, empty rows included."""
    pa = require_pyarrow()
    columns = {field.name: [row[field.name] for row in rows] for field in schema}
    return pa.table(columns, schema=schema)


def validate_columns(table: Any, role: str) -> None:  # noqa: ANN401
    """Refuse a frame missing a column its role requires, before an upload is spent on it.

    The earliest point the mistake is catchable and the only point it is cheap.
    """
    try:
        required = REQUIRED_COLUMNS[role]
    except KeyError as error:
        raise FormatError(f"{role!r} is not a role of a mesh collection; try {', '.join(REQUIRED_COLUMNS)}.") from error

    names = set(_column_names(table))
    missing = [column for column in required if column not in names]
    if missing:
        raise FormatError(
            f"This frame is being written as the {role.replace('_', ' ')} of a mesh collection, and the format "
            f"requires it to carry {', '.join(required)}. It is missing {', '.join(missing)} (it has "
            f"{', '.join(sorted(names))}). Build it with `maille.build_collection`, which produces all three frames "
            f"with the right columns."
        )


def _column_names(table: Any) -> Iterable[str]:  # noqa: ANN401
    """The column names of an Arrow table, a pandas frame, or anything with a schema."""
    names = getattr(table, "column_names", None)
    if names is not None:
        return list(names)
    schema = getattr(table, "schema", None)
    if schema is not None and hasattr(schema, "names"):
        return list(schema.names)
    columns = getattr(table, "columns", None)
    if columns is not None:
        return [str(column) for column in columns]
    raise FormatError(f"maille cannot read column names off a {type(table).__name__}.")


def table_to_parquet(table: Any, *, compression: str = "zstd") -> bytes:  # noqa: ANN401
    """Serialize an Arrow table to Parquet bytes.

    The Parquet-level compression is the *file's*, not the format's: ``encoding.compression``
    describes the geometry blobs inside a row, and those are already meshopt-coded and left
    alone here. Compressing the file around them still pays on the catalogs, which are plain
    numeric columns.
    """
    pa = require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression)
    return bytes(sink.getvalue().to_pybytes())


#: How many bytes of geometry a row group aims to hold. **This is the knob the whole
#: range-reading story turns on**, and it is a two-sided one: a row group is the smallest thing
#: a reader can fetch, so a large one means fetching a cell drags its neighbours along -- but a
#: Parquet footer grows with row-group count, and the footer is read on every part a reader
#: opens. Chunk too finely and a viewer trades "download the level" for "download a big footer",
#: which is a worse deal on the first cell and no better on the rest.
#:
#: 512 KiB sits where a row group holds a handful of cells rather than one or hundreds. The
#: footer is cached per part for the life of a reader, so its cost is paid once per session
#: while the row-group cost is paid per fetch -- which is the asymmetry this number is picked
#: against.
DEFAULT_ROW_GROUP_BYTES = 512 * 1024


def blob_sizes(table: Any) -> list[int]:  # noqa: ANN401
    """How many bytes of geometry each row of a shard carries.

    Budgeting on the blobs rather than on a row count is what keeps chunks even: a cell holding
    one small object and a cell holding two hundred differ by orders of magnitude in bytes and
    not at all in rows.
    """
    positions = table.column("positions").to_pylist()
    indices = table.column("indices").to_pylist()
    return [len(p or b"") + len(i or b"") for p, i in zip(positions, indices)]


def plan_byte_chunks(sizes: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Group consecutive rows into ``(start, count)`` runs that each fit ``budget`` bytes.

    Used twice, at two scales: to split a level into parts, and to split a part into row
    groups. A single row larger than the budget still goes in a run of its own rather than
    being split -- a cell is the smallest thing a reader fetches.
    """
    if not sizes:
        return [(0, 0)]
    chunks: list[tuple[int, int]] = []
    start = 0
    running = 0
    for row, size in enumerate(sizes):
        if running and running + size > budget:
            chunks.append((start, row - start))
            start, running = row, 0
        running += size
    chunks.append((start, len(sizes) - start))
    return chunks


def table_to_chunked_parquet(
    table: Any,  # noqa: ANN401
    *,
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    compression: str = "zstd",
) -> tuple[bytes, list[tuple[int, int]]]:
    """Serialize a geometry shard with one row group per byte-budgeted run of cells.

    Returns the Parquet bytes and the ``(start_row, row_count)`` of each row group, in order --
    which is what lets the writer record, per cell, the row group a reader must fetch to get
    it. Separate from :func:`table_to_parquet` rather than an argument to it: the catalogs are
    read whole and have no use for either the chunking or the second return value.
    """
    pa = require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    chunks = plan_byte_chunks(blob_sizes(table), row_group_bytes)
    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(sink, table.schema, compression=compression) as writer:
        for start, count in chunks:
            writer.write_table(table.slice(start, count))
    return bytes(sink.getvalue().to_pybytes()), chunks


def parquet_to_table(body: bytes) -> Any:  # noqa: ANN401
    """Read Parquet bytes back into an Arrow table."""
    pa = require_pyarrow()
    import pyarrow.parquet as pq  # type: ignore

    return pq.read_table(pa.BufferReader(body))


__all__ = [
    "DEFAULT_ROW_GROUP_BYTES",
    "REQUIRED_COLUMNS",
    "arrow_schemas",
    "blob_sizes",
    "build_table",
    "parquet_to_table",
    "plan_byte_chunks",
    "require_pyarrow",
    "table_to_chunked_parquet",
    "table_to_parquet",
    "validate_columns",
]
