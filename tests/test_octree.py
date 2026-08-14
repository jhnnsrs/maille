"""How space is divided and addressed: the Morton code a cell is, and the box it stands for.

These are the tests a decoder in another language should be able to read as a specification --
the addressing half of it, with the byte layout in ``test_codecs.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import maille
from maille.octree import MORTON_BITS, cell_box


def test_morton_interleaves_with_x_least_significant():
    """The bit order is the format, not an implementation detail.

    Spelled out rather than round-tripped: a writer and a reader that agree on the *wrong*
    interleave round-trip perfectly and address every cell wrongly.
    """
    assert maille.morton_encode_one((1, 0, 0)) == 0b001
    assert maille.morton_encode_one((0, 1, 0)) == 0b010
    assert maille.morton_encode_one((0, 0, 1)) == 0b100
    assert maille.morton_encode_one((1, 1, 1)) == 0b111
    assert maille.morton_encode_one((2, 0, 0)) == 0b001000
    assert maille.morton_encode_one((0, 0, 2)) == 0b100000


def test_morton_round_trips_over_asymmetric_triples():
    """Asymmetric on purpose: (3, 5, 9) survives a swapped decoder, (3, 3, 3) does not."""
    for triple in [(0, 0, 0), (3, 5, 9), (17, 2, 300), (1, 0, 65536)]:
        assert maille.morton_decode(maille.morton_encode_one(triple)) == triple


def test_a_cell_index_past_seventeen_bits_is_refused():
    """The cap is what keeps a Morton code under the format's 2**53 limit."""
    with pytest.raises(ValueError, match="17-bit limit"):
        maille.morton_encode_one((1 << MORTON_BITS, 0, 0))


def test_a_negative_cell_index_is_refused():
    """Geometry is shifted into the positive octant before it is addressed."""
    with pytest.raises(ValueError, match="non-negative"):
        maille.morton_encode(np.array([[-1, 0, 0]]))


def test_the_cell_box_is_the_morton_triple_scaled_by_the_level():
    """A decoder needs only `level` and `cell` to find the box, which is what per-cell means."""
    origin, extent = cell_box(maille.morton_encode_one((2, 1, 3)), 1, (128, 128, 64))
    assert extent.tolist() == [256.0, 256.0, 128.0]
    assert origin.tolist() == [512.0, 256.0, 384.0]
