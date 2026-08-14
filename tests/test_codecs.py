"""The byte format: per-cell quantization and the two blob codecs.

These are the tests a decoder in another language should be able to read as a specification --
the byte layout half of it, with the addressing in ``test_octree.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import maille
from maille.codecs import QUANT_MAX
from maille.octree import cell_box


@pytest.mark.parametrize("compression", [maille.COMPRESSION_NONE, maille.COMPRESSION_ZSTD])
def test_positions_round_trip_within_a_quantum(compression: str):
    """The quantization error is bounded by the cell's own quantum, however the blob is stored."""
    cell_size = (128, 128, 64)
    cell = maille.morton_encode_one((1, 2, 3))
    origin, extent = cell_box(cell, 0, cell_size)

    rng = np.random.default_rng(0)
    vertices = origin + rng.random((64, 3)) * extent

    blob = maille.encode_positions(vertices, cell=cell, level=0, cell_size=cell_size, compression=compression)
    decoded = maille.decode_positions(
        blob, cell=cell, level=0, cell_size=cell_size, compression=compression, vertex_count=len(vertices)
    )

    quantum = extent / QUANT_MAX
    assert np.all(np.abs(decoded - vertices) <= quantum), "a vertex moved by more than one quantum"


def test_an_uncompressed_positions_blob_is_six_bytes_per_vertex():
    """Under `codec: NONE` the blob is exactly three interleaved little-endian uint16."""
    cell_size = (128, 128, 64)
    origin, _ = cell_box(0, 0, cell_size)
    vertices = origin + np.array([[0.0, 0.0, 0.0], [128.0, 128.0, 64.0]])

    blob = maille.encode_positions(vertices, cell=0, level=0, cell_size=cell_size, compression=maille.COMPRESSION_NONE)

    assert len(blob) == 6 * len(vertices)
    assert np.frombuffer(blob, dtype="<u2").tolist() == [0, 0, 0, QUANT_MAX, QUANT_MAX, QUANT_MAX]


def test_a_vertex_outside_its_cell_is_refused_rather_than_clamped():
    """The one check that cannot be recovered downstream.

    Clamping would produce a blob of the right length with the right column types, and the only
    symptom would be geometry welded to a cell wall -- undetectable by any later layer.
    """
    with pytest.raises(maille.PartitioningError, match="partitioning bug, not a"):
        maille.encode_positions(
            np.array([[200.0, 10.0, 10.0]]), cell=0, level=0, cell_size=(128, 128, 64), compression=maille.COMPRESSION_NONE
        )


def test_a_boundary_vertex_is_not_mistaken_for_a_stray_one():
    """A vertex exactly on the far face quantizes to 65535, and must not trip the check."""
    blob = maille.encode_positions(
        np.array([[128.0, 128.0, 64.0]]), cell=0, level=0, cell_size=(128, 128, 64), compression=maille.COMPRESSION_NONE
    )
    assert np.frombuffer(blob, dtype="<u2").tolist() == [QUANT_MAX, QUANT_MAX, QUANT_MAX]


@pytest.mark.parametrize("compression", [maille.COMPRESSION_NONE, maille.COMPRESSION_ZSTD])
def test_indices_round_trip_as_the_same_triangles(compression: str):
    """MESHOPT preserves triangle order but may rotate a triangle's starting vertex.

    Same three corners, same winding, identical surface -- so the check is on the corner sets,
    which is what a renderer sees, rather than on the bytes.
    """
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 4, 0], [4, 5, 1]], dtype=np.int64)

    blob = maille.encode_indices(faces, compression=compression, vertex_count=6)
    decoded = maille.decode_indices(blob, compression=compression, index_count=faces.size)

    assert decoded.shape == faces.shape
    assert [set(triangle) for triangle in decoded] == [set(triangle) for triangle in faces]


def test_decoding_meshopt_without_a_count_is_refused():
    """The encoded buffer carries no length, which is why the geometry row has the column."""
    with pytest.raises(ValueError, match="`vertex_count` is required"):
        maille.decode_positions(b"", cell=0, level=0, cell_size=(128, 128, 64), compression=maille.COMPRESSION_ZSTD)
    with pytest.raises(ValueError, match="`index_count` is required"):
        maille.decode_indices(b"", compression=maille.COMPRESSION_ZSTD)


def test_a_compressed_blob_needs_its_row_to_know_its_length():
    """The ZSTD framing carries no content size, so the row's counts *are* the length.

    Stated as a test because it is the one thing a decoder in another language could get wrong
    silently: it must size the output buffer from `vertex_count` / `index_count`, not from
    anything in the frame.
    """
    vertices = np.array([[0.0, 0, 0], [64, 64, 32]])
    blob = maille.encode_positions(
        vertices, cell=0, level=0, cell_size=(128, 128, 64), compression=maille.COMPRESSION_ZSTD
    )

    with pytest.raises(maille.FormatError, match="required to decode a compressed"):
        maille.decode_positions(
            blob, cell=0, level=0, cell_size=(128, 128, 64), compression=maille.COMPRESSION_ZSTD
        )

    decoded = maille.decode_positions(
        blob,
        cell=0,
        level=0,
        cell_size=(128, 128, 64),
        compression=maille.COMPRESSION_ZSTD,
        vertex_count=len(vertices),
    )
    assert decoded.shape == vertices.shape


def test_an_unknown_codec_or_compression_is_refused_by_name():
    """A stale constant should be told, not silently handed the raw layout."""
    with pytest.raises(maille.FormatError, match="`codec` is"):
        maille.encode_positions(
            np.zeros((1, 3)), cell=0, level=0, cell_size=(128, 128, 64), codec="DRACO"
        )
    with pytest.raises(maille.FormatError, match="`compression` is"):
        maille.encode_positions(
            np.zeros((1, 3)), cell=0, level=0, cell_size=(128, 128, 64), compression="BROTLI"
        )
