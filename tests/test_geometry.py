"""The two promises nothing downstream can check: ``boundary: LOCKED`` and ``decimation: QUARTER``.

Both are declarations a server records and cannot verify, and no reader will notice a writer
that lied about either. So they are verified here or nowhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import maille
from maille.codec import QUANT_MAX, morton_encode_one
from maille.geometry import decimate_fixed, on_planes, snap_boundary

CELL_SIZE = np.array([128, 128, 64], dtype=np.int64)
LEVELS = 3
COARSE_EXTENT = CELL_SIZE.astype(np.float64) * (2 ** (LEVELS - 1))


def round_trip_at(point: np.ndarray, level: int) -> np.ndarray:
    """Quantize a point against whichever cell holds it at ``level``, then decode it back."""
    extent = CELL_SIZE.astype(np.float64) * (2**level)
    triple = np.floor(point / extent).astype(np.int64)
    cell = morton_encode_one(tuple(int(component) for component in triple))
    blob = maille.encode_positions(
        point.reshape(1, 3), cell=cell, level=level, cell_size=CELL_SIZE, codec=maille.CODEC_NONE
    )
    return maille.decode_positions(blob, cell=cell, level=level, cell_size=CELL_SIZE, codec=maille.CODEC_NONE)[0]


def test_a_plane_shared_by_every_level_decodes_identically_at_every_level():
    """This is ``LOCKED``: the two sides of a shared face agree, whatever level draws them.

    The plane chosen is a face at every level -- a multiple of the *coarsest* cell extent --
    which is exactly the case where a disagreement would be a visible crack between a fine cell
    and the coarse neighbour it abuts.
    """
    point = np.array([512.0, 512.0, 256.0])  # 128*4, 128*4, 64*4: a face plane at levels 0..2
    snapped, boundary = snap_boundary(point.reshape(1, 3), CELL_SIZE, COARSE_EXTENT)
    assert boundary[0], "a point on a cell face plane is a boundary vertex"

    decoded = [round_trip_at(snapped[0], level) for level in range(LEVELS)]
    for level, position in enumerate(decoded):
        assert position == pytest.approx(point, abs=0.0), f"level {level} moved a locked vertex"


def test_a_tangential_coordinate_is_snapped_onto_the_coarsest_lattice():
    """The second half of the snap, and the one that is easy to leave out.

    A coordinate *along* a shared face still has to land somewhere every level can name, or the
    two sides of the face disagree along it even though the plane itself is exact.
    """
    point = np.array([512.0, 300.123456, 33.98765])
    snapped, _ = snap_boundary(point.reshape(1, 3), CELL_SIZE, COARSE_EXTENT)

    step = COARSE_EXTENT / QUANT_MAX
    residual = snapped[0, 1:] / step[1:]
    assert residual == pytest.approx(np.rint(residual), abs=1e-9), "a tangential coordinate is off the lattice"

    decoded = [round_trip_at(snapped[0], level) for level in range(LEVELS)]
    for level, position in enumerate(decoded[1:], start=1):
        assert position == pytest.approx(decoded[0], abs=1e-9), f"level {level} disagrees with level 0 along the face"


def test_an_interior_vertex_is_left_alone():
    """Snapping an interior vertex would spend the precision the finest level exists to carry."""
    point = np.array([60.123456, 70.654321, 33.111111])
    snapped, boundary = snap_boundary(point.reshape(1, 3), CELL_SIZE, COARSE_EXTENT)

    assert not boundary[0]
    assert snapped[0] == pytest.approx(point, abs=0.0)


def test_the_on_plane_window_is_a_full_coarse_quantum_wide():
    """The width is load-bearing, not a tolerance.

    A vertex nearer to a plane than half a coarse quantum would be snapped *across* it if it
    were treated as interior, landing outside the cell that holds it -- where quantization
    cannot represent it at all. So anything within a whole quantum is pinned instead.
    """
    quantum = float(COARSE_EXTENT[0]) / QUANT_MAX
    just_inside = np.array([[512.0 + 0.9 * quantum, 40.0, 20.0]])

    snapped, boundary = snap_boundary(just_inside, CELL_SIZE, COARSE_EXTENT)

    assert boundary[0], "a vertex within a coarse quantum of a plane counts as on it"
    assert snapped[0, 0] == 512.0, "and is pinned to the plane rather than snapped across it"


def test_decimation_never_moves_a_fixed_vertex():
    """``LOCKED`` is boundary immobility, and this is that property directly."""
    rng = np.random.default_rng(7)
    grid = np.stack(np.meshgrid(np.arange(9.0), np.arange(9.0), [0.0]), axis=-1).reshape(-1, 3)
    vertices = grid + rng.random(grid.shape) * 0.01
    faces = []
    for i in range(8):
        for j in range(8):
            a, b, c, d = i * 9 + j, i * 9 + j + 1, (i + 1) * 9 + j, (i + 1) * 9 + j + 1
            faces.extend([[a, b, c], [b, d, c]])
    faces = np.array(faces, dtype=np.int64)

    fixed = vertices[:, 0] <= 0.5  # one whole edge of the sheet
    kept, _, _ = decimate_fixed(vertices, faces, fixed=fixed, target_faces=16)

    for locked in vertices[fixed]:
        assert np.isclose(kept, locked).all(axis=1).any(), "a fixed vertex was moved or collapsed away"


def test_decimation_reaches_its_target_when_nothing_is_pinned():
    """``QUARTER`` is a face-count ratio, so hitting the count is the whole obligation."""
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(subdivisions=3)
    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    faces = np.asarray(sphere.faces, dtype=np.int64)

    _, kept, moved = decimate_fixed(
        vertices, faces, fixed=np.zeros(len(vertices), dtype=bool), target_faces=len(faces) // 4
    )

    assert len(kept) <= len(faces) // 4
    assert moved > 0.0, "the displacement is the decimation half of lod_error"


def test_a_small_object_is_not_decimated_out_of_existence():
    """A level is a standalone representation, so an object may not vanish from a coarse one.

    The failure this guards is silent in the worst way: a level that lost a row looks exactly
    like a level that never had one, so the object simply disappears when a viewer zooms out.
    """
    trimesh = pytest.importorskip("trimesh")
    tiny = trimesh.creation.box(extents=[6.0, 4.0, 3.0]).apply_translation([40.0, 40.0, 20.0])

    collection = maille.build_collection({5: tiny}, axes=("z", "y", "x"), cell_size=(128, 128, 64), levels=3)

    for level, shard in collection.shards:
        assert shard.num_rows > 0, f"level {level} lost the only object in the collection"
        assert 5 in shard.column("object_ids")[0].as_py()


def test_on_planes_finds_the_faces_of_the_level_it_is_asked_about():
    """Which vertices a level may not move is a per-level question, not a global one."""
    vertices = np.array([[128.0, 10.0, 10.0], [60.0, 10.0, 10.0]])

    assert on_planes(vertices, np.array([128.0, 128.0, 64.0])).tolist() == [True, False]
    # At level 1 the cell is 256 wide, so x=128 is interior -- free for the decimator to spend.
    assert on_planes(vertices, np.array([256.0, 256.0, 128.0])).tolist() == [False, False]
