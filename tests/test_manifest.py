"""The manifest: what it must carry, what it may not guess, and what it refuses."""

from __future__ import annotations

import json

import pytest

import maille
from maille.manifest import Encoding, Grid, Manifest, level_part_path, level_prefix, validate_axes


def a_manifest(**overrides: object) -> dict:
    """A conforming manifest as a dict, with fields overridden for the refusal tests."""
    base = {
        # The version this build emits, rather than a literal: these tests are about what a
        # manifest must carry, and pinning a number here would fail the format's next version
        # for reasons that have nothing to do with what each test checks.
        "specVersion": maille.SPEC_VERSION,
        "grid": {"cellSize": [128, 128, 64], "levels": 3, "sortKey": "MORTON"},
        "encoding": {
            "positions": "UINT16_QUANTIZED_PER_CELL",
            "indices": "UINT32",
            "codec": "MESHOPT",
            "compression": "NONE",
            "boundary": "LOCKED",
            "decimation": "QUARTER",
        },
        "axes": ["z", "y", "x"],
        "counts": {"objects": 3, "cellsPerLevel": [34, 8, 1]},
        "files": {"cells": "catalog/cells.parquet", "objects": "catalog/objects.parquet"},
    }
    base.update(overrides)
    return base


def test_the_layout_is_the_one_the_format_fixes():
    """These paths are the contract; a reader that cannot list a prefix still finds them."""
    assert maille.MANIFEST_NAME == "maille.json"
    assert maille.CELL_CATALOG_PATH == "catalog/cells.parquet"
    assert maille.OBJECT_CATALOG_PATH == "catalog/objects.parquet"
    assert level_prefix(2) == "level=2"
    assert level_part_path(2) == "level=2/part-00000.parquet"
    assert level_part_path(0, 13) == "level=0/part-00013.parquet"


def test_a_manifest_round_trips_through_its_own_bytes():
    """What is written is what is read; nothing is resolved on only one side."""
    manifest = Manifest.from_dict(a_manifest())

    assert Manifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


def test_the_stored_encoding_is_resolved_rather_than_sparse():
    """A renderer configures its decoder from what it reads back.

    Defaulting a key inside the writer without persisting it would hand every reader an
    encoding that says nothing -- which is the one implementation trap in this design.
    """
    written = json.loads(Manifest(grid=Grid((128, 128, 64), 3), encoding=Encoding(), axes=("z", "y", "x")).to_json())

    assert set(written["encoding"]) == {
        "positions", "indices", "codec", "compression", "boundary", "decimation",
    }


def test_an_unreadable_version_is_refused_rather_than_read_anyway():
    """The version selects how every byte in the prefix is read."""
    with pytest.raises(maille.FormatError, match="cannot read"):
        Manifest.from_dict(a_manifest(specVersion=str(int(maille.SPEC_VERSION) + 1)))
    with pytest.raises(maille.FormatError, match="cannot read"):
        Manifest.from_dict(a_manifest(specVersion=str(int(maille.SPEC_VERSION) - 1)))
    with pytest.raises(maille.FormatError, match="cannot read"):
        Manifest.from_dict(a_manifest(specVersion=""))


def test_a_manifest_without_a_grid_or_an_encoding_is_refused():
    """Nothing else in the store states them, so a manifest without them is undecodable."""
    for dropped in ("grid", "encoding"):
        raw = a_manifest()
        del raw[dropped]
        with pytest.raises(maille.FormatError, match="must carry a `grid` and an `encoding`"):
            Manifest.from_dict(raw)


def test_an_encoding_missing_a_key_a_decoder_needs_is_refused():
    """`codec` and `compression` are exactly the keys that must never be guessed."""
    for dropped in ("codec", "compression", "indices"):
        encoding = dict(a_manifest()["encoding"])  # type: ignore[arg-type]
        del encoding[dropped]
        with pytest.raises(maille.FormatError, match=dropped):
            Manifest.from_dict(a_manifest(encoding=encoding))


def test_a_value_outside_the_vocabulary_is_refused():
    """An undefined value would be recorded and acted on by a decoder that cannot honour it."""
    with pytest.raises(maille.FormatError, match="`encoding.codec`"):
        Encoding(codec="DRACO")
    with pytest.raises(maille.FormatError, match="`encoding.compression`"):
        Encoding(compression="BROTLI")


def test_a_truncated_manifest_is_named_as_an_interrupted_write():
    """The other shape a killed writer leaves behind, and it should not read as corruption."""
    body = Manifest.from_dict(a_manifest()).to_json()[:40]

    with pytest.raises(maille.FormatError, match="not valid JSON"):
        Manifest.from_json(body)


def test_a_grid_that_cannot_address_geometry_is_refused():
    """Three components, at least one voxel each, at least one level."""
    with pytest.raises(maille.FormatError, match="three whole numbers"):
        Grid(cell_size=(128, 128, 0), levels=3)
    with pytest.raises(maille.FormatError, match="at least one level"):
        Grid(cell_size=(128, 128, 64), levels=0)
    with pytest.raises(maille.FormatError, match="three-dimensional"):
        Grid.from_dict({"cellSize": [128, 128], "levels": 2})


def test_the_cell_size_is_read_in_x_y_z_and_not_reversed():
    """An anisotropic size read backwards is a legal grid describing a different octree.

    Asymmetric on purpose: a cubic fixture passes a reversed implementation.
    """
    grid = Grid.from_dict({"cellSize": [128, 128, 64], "levels": 3, "sortKey": "MORTON"})

    assert grid.cell_size == (128, 128, 64)
    assert grid.cell_extent(0) == (128.0, 128.0, 64.0)
    assert grid.cell_extent(2) == (512.0, 512.0, 256.0)


def test_axes_is_checked_but_never_invented():
    """Any order is accepted -- a collection owns its coordinate system -- but not a broken one."""
    assert validate_axes(["z", "y", "x"]) == ("z", "y", "x")
    assert validate_axes(["x", "y", "z"]) == ("x", "y", "z")

    with pytest.raises(maille.FormatError, match="names 3 axes"):
        validate_axes(["z", "y"])
    with pytest.raises(maille.FormatError, match="names each axis once"):
        validate_axes(["y", "y", "x"])


def test_a_manifest_without_axes_is_read_in_full():
    """Nothing in the format decodes through `axes`, so a manifest without it is complete.

    The grid and the encoding are what a reader cannot work without, and they are refused when
    missing. An axis order is a claim by the layer above about how this collection relates to
    something else -- so its absence means that layer made no claim, not that the collection is
    unreadable.
    """
    raw = a_manifest()
    del raw["axes"]

    manifest = Manifest.from_dict(raw)

    assert manifest.axes is None
    assert manifest.grid.cell_size == (128, 128, 64)
    assert "axes" not in manifest.to_dict(), "an absent claim must not become a written one"
    assert Manifest.from_json(manifest.to_json()).axes is None


def test_a_malformed_axes_claim_is_still_refused():
    """Optional does not mean unchecked: a present-but-wrong claim is a mistake to name."""
    with pytest.raises(maille.FormatError, match="names 3 axes"):
        Manifest.from_dict(a_manifest(axes=["z", "y"]))
    with pytest.raises(maille.FormatError, match="names each axis once"):
        Manifest.from_dict(a_manifest(axes=["y", "y", "x"]))


def test_a_declared_axis_order_is_carried_through_untouched():
    """maille never reorders or normalises it -- any order is the caller's to state."""
    for order in (["z", "y", "x"], ["x", "y", "z"], ["c", "a", "b"]):
        assert list(Manifest.from_dict(a_manifest(axes=order)).to_dict()["axes"]) == order


def test_both_blob_knobs_default_to_nothing():
    """A collection you get without asking needs no decoder on the reading side.

    That is the whole reason `NONE`/`NONE` is the default: the blob a consumer pulls out of the
    Parquet column *is* the buffer it uploads. Size is available in exchange for a decoder, and
    both ways of buying it are opt-in.
    """
    encoding = Encoding()

    assert encoding.codec == maille.CODEC_NONE
    assert encoding.compression == maille.COMPRESSION_NONE


@pytest.mark.parametrize(
    ("codec", "compression"),
    [
        (maille.CODEC_NONE, maille.COMPRESSION_NONE),
        (maille.CODEC_NONE, maille.COMPRESSION_ZSTD),
        (maille.CODEC_MESHOPT, maille.COMPRESSION_NONE),
    ],
)
def test_the_pairs_the_format_allows(codec: str, compression: str):
    """Three of the four combinations are legal, and each is a real choice."""
    written = Encoding(codec=codec, compression=compression).to_dict()

    assert written["codec"] == codec
    assert written["compression"] == compression


def test_meshopt_with_zstd_is_refused_as_undecodable():
    """The one illegal pair, and it is a decodability problem rather than a taste one.

    ZSTD framing here carries no content size, so a reader recovers the uncompressed length
    from the row: 6 bytes a vertex, 4 bytes an index. A meshopt blob has no such relation to
    its counts, so the length is unknowable and the pair cannot be read back at all.
    """
    with pytest.raises(maille.FormatError, match="cannot be decoded"):
        Encoding(codec=maille.CODEC_MESHOPT, compression=maille.COMPRESSION_ZSTD)


def test_the_manifest_is_named_after_the_format():
    """`maille.json`, at the root of the prefix -- the one file a reader must find by name."""
    assert maille.MANIFEST_NAME == "maille.json"
