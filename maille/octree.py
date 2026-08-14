"""How space is divided and addressed: a cell index triple as one integer, and its box.

This is the spatial organization rather than the byte format -- what a cell *is*, not how its
geometry is packed. :mod:`maille.codecs` is the latter, and it depends on this module for
:func:`cell_box`; nothing here depends on it.

Level 0 is the finest. A cell at level ``L`` spans ``cell_size * 2**L`` voxels per axis, so
level ``L`` has one cell for every ``2**3`` cells at level ``L-1``. A point lands in the cell
whose index triple is ``floor(p / (cell_size * 2**L))``.

``cell`` is the Morton code of that triple with **component 0 least significant**: bit ``3*n``
is bit ``n`` of ``i``, bit ``3*n+1`` is bit ``n`` of ``j``, bit ``3*n+2`` is bit ``n`` of
``k``. Each index is capped at 17 bits so the code stays under the format's ``2**53`` limit.

The interleave is what makes one integer a usable sort key: cells near each other in space land
near each other in the catalog, so a plan's rows are a handful of runs rather than a scatter.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

#: The largest Morton code the format allows, and the per-axis index width that keeps us under
#: it: three axes at 17 bits interleave to 51.
MORTON_BITS = 17
MAX_MORTON = 1 << 53


def _spread_bits(value: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Insert two zero bits after each of the low 17 bits, for a 3D Morton interleave."""
    result = np.zeros_like(value, dtype=np.int64)
    for bit in range(MORTON_BITS):
        result |= ((value >> bit) & 1) << (3 * bit)
    return result


def morton_encode(triples: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Morton codes for an ``(n, 3)`` array of ``(i, j, k)`` cell indices, x least significant."""
    triples = np.asarray(triples, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"Cell indices come as an (n, 3) array, got {triples.shape}.")
    if triples.min(initial=0) < 0:
        raise ValueError("Cell indices are non-negative; shift the geometry into the positive octant first.")
    if triples.max(initial=0) >= (1 << MORTON_BITS):
        raise ValueError(
            f"A cell index reached {int(triples.max())}, past the {MORTON_BITS}-bit limit that keeps "
            f"a Morton code under 2**53. Use a larger cell_size or fewer levels."
        )
    codes = _spread_bits(triples[:, 0]) | (_spread_bits(triples[:, 1]) << 1) | (_spread_bits(triples[:, 2]) << 2)
    if codes.size and int(codes.max()) >= MAX_MORTON:
        raise ValueError("A Morton code exceeded the format's 2**53 limit.")
    return codes


def morton_encode_one(triple: Sequence[int]) -> int:
    """The Morton code of a single ``(i, j, k)`` cell index triple."""
    return int(morton_encode(np.asarray([triple], dtype=np.int64))[0])


def morton_decode(code: int) -> tuple[int, int, int]:
    """The ``(i, j, k)`` cell index triple behind a Morton code. The inverse of :func:`morton_encode`."""
    i = j = k = 0
    for bit in range(MORTON_BITS):
        i |= ((code >> (3 * bit)) & 1) << bit
        j |= ((code >> (3 * bit + 1)) & 1) << bit
        k |= ((code >> (3 * bit + 2)) & 1) << bit
    return i, j, k


def cell_box(
    cell: int, level: int, cell_size: Sequence[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The grid box of a cell: its origin and its extent, in voxels, ``(x, y, z)``.

    This is what makes the quantization invertible from the row alone: a decoder that has
    ``level``, ``cell`` and the manifest's ``cell_size`` has the box, and needs nothing else.
    """
    sizes = np.asarray(cell_size, dtype=np.int64)
    extent = sizes.astype(np.float64) * (2 ** int(level))
    origin = np.asarray(morton_decode(int(cell)), dtype=np.float64) * extent
    return origin, extent


__all__ = [
    "MAX_MORTON",
    "MORTON_BITS",
    "cell_box",
    "morton_decode",
    "morton_encode",
    "morton_encode_one",
]
