"""Checking a written collection against the claims its manifest makes.

The README says two of those claims are "true by construction and unverifiable by anything
downstream". That is half right: they are unverifiable *from a single cell*, which is all a
renderer ever looks at. Given the whole collection they are perfectly checkable, and a format
whose trust story is "the writer keeps this or nobody does" should ship the thing that checks.

Three tiers, because they cost wildly different amounts and a caller should get to choose:

``structure``
    The catalogs against each other and against the manifest. Reads two small files and no
    geometry at all -- cheap enough for a server to run at registration time, which is the
    point in the pipeline where rejecting a bad collection is free.

``blobs``
    Decodes every cell. Catches a truncated or mis-encoded blob, an index pointing past the
    vertex array, an offset run that does not ascend -- the failures that produce garbage
    rather than an exception.

``geometry``
    The two cross-level claims: ``boundary: LOCKED`` and ``decimation``, plus ``lod_error`` as
    a real bound. Reads every level of every cell and compares them, so it is the expensive
    one and the only one that can actually falsify the format's headline promise.

Nothing here raises. A verifier that stops at the first problem tells you about one thing when
you wanted to know about all of them, so every check runs and the report carries the lot.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from maille.codecs import QUANT_MAX
from maille.geometry import on_planes
from maille.octree import cell_box
from maille.stores import get_range_bytes, join

if TYPE_CHECKING:  # pragma: no cover - typing only
    from maille.reader import CellEntry, Collection

#: The tiers, cheapest first. Each includes the ones before it.
TIERS = ("structure", "blobs", "geometry")


@dataclass(frozen=True)
class Check:
    """One question asked of a collection, and what the answer was."""

    name: str
    tier: str
    ok: bool
    detail: str = ""
    #: A few offending keys, when there are any. Truncated on purpose: a broken collection
    #: usually breaks in thousands of places and a report nobody can read is a report nobody
    #: reads.
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        """A single line, so a report prints as a list."""
        mark = "ok  " if self.ok else "FAIL"
        shown = f" [{', '.join(self.examples)}]" if self.examples else ""
        return f"{mark} {self.name}: {self.detail}{shown}"


@dataclass(frozen=True)
class VerifyReport:
    """Every check that ran, and whether the collection survived all of them."""

    checks: tuple[Check, ...] = ()
    tier: str = "blobs"
    #: What was not checked and why -- an optional dependency missing, a tier not requested.
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        """The checks that did not pass."""
        return tuple(check for check in self.checks if not check.ok)

    def __bool__(self) -> bool:
        """A report is truthy when the collection is sound."""
        return self.ok

    def __str__(self) -> str:
        """The whole report, one check per line."""
        head = f"{'PASS' if self.ok else 'FAIL'}: {len(self.checks) - len(self.failures)}/{len(self.checks)} checks at tier {self.tier!r}"
        lines = [head, *(str(check) for check in self.checks)]
        lines += [f"skipped {note}" for note in self.skipped]
        return "\n".join(lines)


def verify(collection: Collection, *, tier: str = "blobs") -> VerifyReport:
    """Check a collection against what its manifest claims, and report everything found.

    Args:
        collection: An opened collection. Only read from.
        tier: How much to check -- ``structure``, ``blobs`` or ``geometry``. Each includes the
            cheaper ones.

    Returns:
        A :class:`VerifyReport`. Falsy when anything failed, so ``if not verify(c): ...`` reads
        the way it should.
    """
    if tier not in TIERS:
        raise ValueError(f"`tier` is one of {', '.join(TIERS)}, got {tier!r}.")

    checks: list[Check] = []
    skipped: list[str] = []
    wanted = TIERS[: TIERS.index(tier) + 1]

    checks.extend(_structure(collection))
    if "blobs" in wanted:
        checks.extend(_blobs(collection))
    if "geometry" in wanted:
        checks.extend(_geometry(collection))
    else:
        skipped.append("the `geometry` tier, which is what checks `boundary` and `decimation`")

    return VerifyReport(checks=tuple(checks), tier=tier, skipped=tuple(skipped))


# --------------------------------------------------------------------------- #
# structure: the catalogs against each other, with no geometry read
# --------------------------------------------------------------------------- #


def _structure(collection: Collection) -> Iterator[Check]:
    """The checks that need only the manifest and the two catalogs."""
    yield _files_exist(collection)
    yield _recorded_lengths_are_right(collection)
    yield _levels_are_contiguous(collection)
    yield _child_masks_name_what_exists(collection)
    yield _parents_dominate_their_children(collection)
    yield _every_cell_is_located(collection)
    yield _locators_are_in_range(collection)
    yield _ordinals_are_dense(collection)
    yield _objects_name_cells_that_exist(collection)


def _files_exist(collection: Collection) -> Check:
    """Every file the manifest names is actually there.

    A one-byte range read rather than a fetch: the question is whether the object exists, and
    on a large collection the difference between asking that and answering it by download is
    the difference between a check and an outage.
    """
    missing: list[str] = []
    paths = [
        collection.manifest.catalog_file("cells", "catalog/cells.parquet").path,
        collection.manifest.catalog_file("objects", "catalog/objects.parquet").path,
    ]
    for level in range(collection.grid.levels):
        paths.extend(entry.path for entry in collection.manifest.level_files(level) or [])
    for path in paths:
        try:
            get_range_bytes(collection.store, join(collection.prefix, path), 0, 1)
        except Exception:  # noqa: BLE001 - any failure means "not readable"
            missing.append(path)
    return Check(
        name="files exist",
        tier="structure",
        ok=not missing,
        detail=f"{len(paths) - len(missing)}/{len(paths)} files the manifest names are readable",
        examples=tuple(missing[:5]),
    )


def _recorded_lengths_are_right(collection: Collection) -> Check:
    """The byte length in the manifest is where the file actually ends.

    This is the one claim in the manifest that a reader *acts on blindly*: it seeks there to
    find the Parquet footer. Get it wrong and pyarrow reports "magic bytes not found", which
    names the symptom in a file that is perfectly intact and says nothing about the manifest
    that lied about it -- so it is worth four bytes to ask directly.

    Four bytes is all it costs: a Parquet file ends with ``PAR1``, so reading the last four
    bytes of the recorded length answers "is this where the file ends" without downloading it.
    """
    wrong: list[str] = []
    checked = 0
    for level in range(collection.grid.levels):
        for entry in collection.manifest.level_files(level) or []:
            if entry.size is None:
                continue
            checked += 1
            path = join(collection.prefix, entry.path)
            try:
                tail = get_range_bytes(collection.store, path, max(0, entry.size - 4), 4)
            except Exception as error:  # noqa: BLE001 - unreadable is its own answer
                wrong.append(f"{entry.path} could not be read at its recorded end ({error})")
                continue
            if tail != b"PAR1":
                wrong.append(
                    f"{entry.path} records {entry.size} bytes, where the last four are {tail!r} rather than b'PAR1'"
                )
    return Check(
        name="recorded file lengths land on the end of the file",
        tier="structure",
        ok=not wrong,
        detail=f"{checked - len(wrong)}/{checked} recorded lengths end where a Parquet file ends",
        examples=tuple(wrong[:5]),
    )


def _levels_are_contiguous(collection: Collection) -> Check:
    """A gap in the levels is geometry a planner never asks for.

    A planner descends from the coarsest level it can see, so everything under a missing level
    is absent from every render with nothing raised anywhere.
    """
    declared = collection.grid.levels
    present = {level for level, _ in collection.cells}
    missing = sorted(set(range(declared)) - present)
    return Check(
        name="levels are contiguous",
        tier="structure",
        ok=not missing,
        detail=f"the grid declares {declared} levels and the catalog holds {len(present)}",
        examples=tuple(f"level {level} is empty" for level in missing[:5]),
    )


def _child_masks_name_what_exists(collection: Collection) -> Check:
    """``child_mask`` is how a planner descends, so a wrong bit is a hole or a dead fetch.

    A bit set where no child exists sends a planner after geometry that is not there; a bit
    clear where one does hides that child and everything below it.
    """
    from maille.octree import morton_encode_one

    wrong: list[str] = []
    for (level, cell), entry in collection.cells.items():
        if level == 0:
            if entry.child_mask:
                wrong.append(f"level 0 cell {cell} claims children")
            continue
        # The children that exist are worked out from the octree, **not from the mask** -- a
        # mask compared only against itself can never reveal a bit that was cleared, and a
        # cleared bit is the worse direction: it hides a child and everything under it, where
        # a spurious one merely wastes a fetch.
        i, j, k = entry.triple
        octants = (
            morton_encode_one((2 * i + (octant & 1), 2 * j + ((octant >> 1) & 1), 2 * k + ((octant >> 2) & 1)))
            for octant in range(8)
        )
        real = {(level - 1, child) for child in octants if (level - 1, child) in collection.cells}
        claimed = set(entry.children())
        if claimed != real:
            wrong.append(
                f"level {level} cell {cell} claims {sorted(c for _, c in claimed)} as children, "
                f"{sorted(c for _, c in real)} exist"
            )
    return Check(
        name="child masks name children that exist",
        tier="structure",
        ok=not wrong,
        detail=f"{len(collection.cells) - len(wrong)}/{len(collection.cells)} cells agree with their mask",
        examples=tuple(wrong[:5]),
    )


def _parents_dominate_their_children(collection: Collection) -> Check:
    """A coarse cell is never more accurate than the cells it summarises.

    If it were, a planner would stop descending at a parent whose error looked acceptable while
    its children were the ones that actually met the budget -- the octree would silently invert.
    """
    wrong: list[str] = []
    for (level, cell), entry in collection.cells.items():
        for key in entry.children():
            child = collection.cells.get(key)
            if child is not None and child.lod_error > entry.lod_error + 1e-9:
                wrong.append(f"level {level} cell {cell} ({entry.lod_error:.4g}) < child {key[1]} ({child.lod_error:.4g})")
    return Check(
        name="a parent's error dominates its children's",
        tier="structure",
        ok=not wrong,
        detail=f"{len(wrong)} parent/child pairs inverted",
        examples=tuple(wrong[:5]),
    )


def _every_cell_is_located(collection: Collection) -> Check:
    """A catalog row without a locator is a cell a reader has to go looking for."""
    unlocated = [
        f"level {entry.level} cell {entry.cell}"
        for entry in collection.cells.values()
        if entry.part is None or entry.row_group is None
    ]
    return Check(
        name="every cell carries a locator",
        tier="structure",
        ok=not unlocated,
        detail=f"{len(collection.cells) - len(unlocated)}/{len(collection.cells)} cells name a part and row group",
        examples=tuple(unlocated[:5]),
    )


def _locators_are_in_range(collection: Collection) -> Check:
    """A locator naming a part or row group that does not exist fails at read time.

    Checked here, from the manifest alone, so it is caught at registration rather than by the
    first viewer to look at that corner of the volume.
    """
    wrong: list[str] = []
    groups: dict[tuple[int, int], int | None] = {}
    for level in range(collection.grid.levels):
        for part, entry in enumerate(collection.manifest.level_files(level) or []):
            groups[(level, part)] = entry.row_groups
    for entry in collection.cells.values():
        if entry.part is None or entry.row_group is None:
            continue
        key = (entry.level, entry.part)
        if key not in groups:
            wrong.append(f"level {entry.level} cell {entry.cell} names part {entry.part}, which does not exist")
        elif groups[key] is not None and entry.row_group >= (groups[key] or 0):
            wrong.append(
                f"level {entry.level} cell {entry.cell} names row group {entry.row_group} of {groups[key]}"
            )
    return Check(
        name="locators point inside the files that exist",
        tier="structure",
        ok=not wrong,
        detail=f"{len(wrong)} locators out of range",
        examples=tuple(wrong[:5]),
    )


def _ordinals_are_dense(collection: Collection) -> Check:
    """Ordinals are a dense index into the objects, which is what makes them worth storing."""
    ordinals = sorted(entry.ordinal for entry in collection.objects.values())
    expected = list(range(len(ordinals)))
    return Check(
        name="object ordinals are dense",
        tier="structure",
        ok=ordinals == expected,
        detail=f"{len(ordinals)} objects, ordinals {'are' if ordinals == expected else 'are not'} 0..n-1 exactly once",
    )


def _objects_name_cells_that_exist(collection: Collection) -> Check:
    """The identity index is only an index if what it points at is there.

    ``objects.parquet`` exists so that isolating one object is a lookup rather than a scan; a
    cell key it names that the spatial index does not hold turns that lookup into a KeyError in
    front of a user.
    """
    wrong: list[str] = []
    for object_id, entry in collection.objects.items():
        for key in entry.cells:
            if key not in collection.cells:
                wrong.append(f"object {object_id} names level {key[0]} cell {key[1]}, which is not in the catalog")
    return Check(
        name="the object catalog names cells that exist",
        tier="structure",
        ok=not wrong,
        detail=f"{len(wrong)} dangling cell references",
        examples=tuple(wrong[:5]),
    )


# --------------------------------------------------------------------------- #
# blobs: decode everything
# --------------------------------------------------------------------------- #


def _blobs(collection: Collection) -> Iterator[Check]:
    """Decode every cell once and ask everything that needs the geometry in hand."""
    counts: list[str] = []
    bounds: list[str] = []
    offsets: list[str] = []
    boxes: list[str] = []
    identity: list[str] = []
    decoded_cells = 0

    for entry in sorted(collection.cells.values(), key=lambda item: (item.level, item.cell)):
        where = f"level {entry.level} cell {entry.cell}"
        try:
            decoded = collection.read_cell(entry.level, entry.cell)
        except Exception as error:  # noqa: BLE001 - a blob that will not decode is the finding
            counts.append(f"{where} did not decode ({type(error).__name__}: {error})")
            continue
        decoded_cells += 1

        if len(decoded.vertices) != entry.vertex_count or len(decoded.faces) * 3 != entry.index_count:
            counts.append(
                f"{where} decoded {len(decoded.vertices)}v/{len(decoded.faces) * 3}i against a catalog "
                f"claiming {entry.vertex_count}v/{entry.index_count}i"
            )
        if len(decoded.faces) and (decoded.faces.max() >= len(decoded.vertices) or decoded.faces.min() < 0):
            bounds.append(f"{where} has an index outside its {len(decoded.vertices)} vertices")

        offsets.extend(_offset_problems(decoded, where))
        boxes.extend(_box_problems(collection, entry, decoded, where))
        identity.extend(_identity_problems(collection, entry, decoded, where))

    total = len(collection.cells)
    yield Check(
        name="every blob decodes to the counts its row claims",
        tier="blobs",
        ok=not counts,
        detail=f"{decoded_cells}/{total} cells decoded, {len(counts)} disagreed with the catalog",
        examples=tuple(counts[:5]),
    )
    yield Check(
        name="indices stay inside their vertex array",
        tier="blobs",
        ok=not bounds,
        detail=f"{len(bounds)} cells index past their own vertices",
        examples=tuple(bounds[:5]),
    )
    yield Check(
        name="object offset runs ascend and stay in bounds",
        tier="blobs",
        ok=not offsets,
        detail=f"{len(offsets)} cells carry an unusable offset run",
        examples=tuple(offsets[:5]),
    )
    yield Check(
        name="a cell's vertices lie inside the box it addresses",
        tier="blobs",
        ok=not boxes,
        detail=f"{len(boxes)} cells hold geometry outside their own cell box",
        examples=tuple(boxes[:5]),
    )
    yield Check(
        name="the two catalogs agree about who is in which cell",
        tier="blobs",
        ok=not identity,
        detail=f"{len(identity)} disagreements between the geometry and the object catalog",
        examples=tuple(identity[:5]),
    )


def _offset_problems(decoded: Any, where: str) -> list[str]:  # noqa: ANN401
    """Whether the per-object offset runs can actually be used to slice the cell."""
    problems: list[str] = []
    ids = decoded.object_ids
    for name, offsets, limit in (
        ("vertex", decoded.object_vertex_offsets, len(decoded.vertices)),
        ("index", decoded.object_index_offsets, len(decoded.faces) * 3),
    ):
        if len(offsets) != len(ids):
            problems.append(f"{where} has {len(offsets)} {name} offsets for {len(ids)} objects")
            continue
        if list(offsets) != sorted(offsets):
            problems.append(f"{where} has {name} offsets that do not ascend")
        if offsets and (offsets[0] != 0 or offsets[-1] > limit):
            problems.append(f"{where} has {name} offsets running 0..{limit} outside that range")
    return problems


def _box_problems(collection: Collection, entry: CellEntry, decoded: Any, where: str) -> list[str]:  # noqa: ANN401
    """Whether a cell's geometry is inside the box its Morton code addresses.

    This is what makes ``cell_size`` checkable at all: a vertex outside its own cell is either
    a wrong ``cell_size`` or a wrong quantization, and both decode without complaint.
    """
    if not len(decoded.vertices):
        return []
    origin, extent = cell_box(entry.cell, entry.level, collection.grid.cell_size)
    slack = float(np.max(extent)) / QUANT_MAX
    low, high = decoded.vertices.min(axis=0), decoded.vertices.max(axis=0)
    if (low < origin - slack).any() or (high > origin + extent + slack).any():
        return [f"{where} spans {low}..{high} outside {origin}..{origin + extent}"]
    return []


def _identity_problems(collection: Collection, entry: CellEntry, decoded: Any, where: str) -> list[str]:  # noqa: ANN401
    """Whether the forward and inverted identity indexes say the same thing.

    ``object_ids`` inside the geometry row is the forward direction; ``objects.parquet.cells``
    is the inverse. They are written from the same data and can only disagree if something has
    gone wrong, which is exactly why the disagreement is worth asking about.
    """
    problems: list[str] = []
    if len(decoded.object_ids) != entry.object_count:
        problems.append(f"{where} holds {len(decoded.object_ids)} objects against a catalog claiming {entry.object_count}")
    if list(decoded.object_ids) != sorted(decoded.object_ids):
        problems.append(f"{where} lists its object ids out of order")
    for object_id in decoded.object_ids:
        record = collection.objects.get(int(object_id))
        if record is None:
            problems.append(f"{where} holds object {object_id}, which the object catalog does not list")
        elif (entry.level, entry.cell) not in record.cells:
            problems.append(f"{where} holds object {object_id}, which does not name this cell")
    return problems


# --------------------------------------------------------------------------- #
# geometry: the cross-level claims
# --------------------------------------------------------------------------- #


def _geometry(collection: Collection) -> Iterator[Check]:
    """The two claims the README calls unverifiable, plus ``lod_error`` as a real bound."""
    yield _coarse_levels_are_smaller(collection)
    yield _boundary_vertices_survive_decimation(collection)
    yield _lod_error_is_a_real_bound(collection)


def _coarse_levels_are_smaller(collection: Collection) -> Check:
    """A coarse level that is not smaller is an octree that costs a fetch and saves nothing.

    The declared ratio is what ``encoding.decimation`` promises. The tolerance is deliberately
    loose -- twice the target -- because the writer already warns loudly when it cannot reach
    it, and the failure this check is for is the *inversion*, where zooming out fetches more.
    """
    faces = {
        level: sum(entry.index_count for entry in collection.cells.values() if entry.level == level) // 3
        for level in range(collection.grid.levels)
    }
    inverted = [level for level in range(1, collection.grid.levels) if faces[level] > faces[level - 1]]
    ratios = ", ".join(
        f"L{level}/L{level - 1}={faces[level] / faces[level - 1]:.2f}"
        for level in range(1, collection.grid.levels)
        if faces[level - 1]
    )
    return Check(
        name="a coarse level holds less than the level it summarises",
        tier="geometry",
        ok=not inverted,
        detail=f"face counts {faces}; {ratios or 'nothing to compare'}",
        examples=tuple(f"level {level} is larger than level {level - 1}" for level in inverted),
    )


def _boundary_vertices_survive_decimation(collection: Collection) -> Check:
    """``boundary: LOCKED``: a vertex on a cell face plane did not move when the level coarsened.

    This is the claim that makes the whole format viable -- it is why a fine cell drawn beside
    a coarse one meets it without a crack -- and it is the one nothing downstream can see. The
    check: take a coarse cell's vertices that lie on its own face planes, and require each to
    be present among its children's vertices on those same planes.

    **Checked in both directions, because one is not enough.** Asking only "is every coarse
    on-plane vertex still there among the children" silently ignores the failure it most needs
    to catch: a vertex that was *moved off* the plane is no longer on-plane, so it drops out of
    the coarse set and the check never looks at it. The other direction closes that -- every
    child vertex lying on one of the coarse cell's own planes must still be present at the
    coarse level, because a coarser cell's planes are a subset of a finer cell's, so such a
    vertex is pinned at every level in between and cannot legitimately go anywhere.

    The tolerance is the coarse cell's quantization step. Two levels quantize against different
    extents, so an exactly-locked vertex still lands on two slightly different reconstructions
    -- ``geometry.py`` works the residual out exactly, and it is smaller than one step.

    **It is one step in two different units, and they must not be swapped.** Deciding whether a
    vertex is *on* a plane is a question in cell-relative coordinates, where 1.0 is a whole cell
    and one quantum is ``1 / QUANT_MAX``; deciding whether two vertices are *the same point* is
    a question in voxels, where that same quantum is ``max(extent) / QUANT_MAX``. Passing the
    voxel figure to :func:`~maille.geometry.on_planes` multiplies the tolerance by the extent --
    half a voxel at level 1, two voxels at level 2 -- and the check then collects vertices that
    merely pass near a plane, which nothing pinned and which are free to move. Every one of them
    is then reported as drift. A closed surface tangent to a cell face is enough to trigger it.
    """
    drifted: list[str] = []
    compared = 0
    for entry in sorted(collection.cells.values(), key=lambda item: (item.level, item.cell)):
        if entry.level == 0:
            continue
        children = [collection.cells[key] for key in entry.children() if key in collection.cells]
        if not children:
            continue
        _, extent = cell_box(entry.cell, entry.level, collection.grid.cell_size)
        planes = np.asarray(extent, dtype=np.float64)
        # The same quantum, in the two units the checks below need it in.
        step = float(np.max(extent)) / QUANT_MAX  # voxels: is this the same point?
        on_plane_tolerance = 1.0 / QUANT_MAX  # cell-relative: is this on a face?
        try:
            coarse = collection.read_cell(entry.level, entry.cell).vertices
            fine = np.vstack([collection.read_cell(c.level, c.cell).vertices for c in children])
        except Exception:  # noqa: BLE001, S112 - the blob tier already reported this
            continue
        if not len(coarse) or not len(fine):
            continue

        pinned = coarse[on_planes(coarse, planes, tolerance=on_plane_tolerance)]
        held = fine[on_planes(fine, planes, tolerance=on_plane_tolerance)]
        compared += len(pinned) + len(held)
        appeared = _count_unmatched(pinned, held, tolerance=2.0 * step) if len(pinned) else 0
        vanished = _count_unmatched(held, pinned, tolerance=2.0 * step) if len(held) else 0
        if appeared:
            drifted.append(
                f"level {entry.level} cell {entry.cell}: {appeared}/{len(pinned)} on-plane vertices "
                f"are not on its children's planes"
            )
        if vanished:
            drifted.append(
                f"level {entry.level} cell {entry.cell}: {vanished}/{len(held)} of its children's "
                f"on-plane vertices did not survive to this level"
            )
    return Check(
        name="on-plane vertices are held fixed across levels (boundary: LOCKED)",
        tier="geometry",
        ok=not drifted,
        detail=f"{compared} pinned vertices compared, {len(drifted)} cells drifted",
        examples=tuple(drifted[:5]),
    )


def _lod_error_is_a_real_bound(collection: Collection) -> Check:
    """``lod_error`` is what a planner spends its budget against, so it has to be true.

    **Measured against the level-0 surface, not its vertices.** The tempting check is the
    distance from each coarse vertex to the nearest fine vertex, and it is wrong: decimating a
    large flat face deletes the vertices in the middle of it, so a surviving vertex can sit
    exactly on the original surface and still be a long way from any surviving point of it. On
    a box 300 voxels across that check reports a 51-voxel error against a true one of 0.002 --
    it fails a collection that is perfect. What a renderer sees, and what ``lod_error`` is
    spent as, is deviation from the surface.
    """
    worst: list[str] = []
    checked = 0
    for object_id, record in collection.objects.items():
        try:
            base = collection.object_mesh(object_id, level=0)
        except Exception:  # noqa: BLE001, S112 - nothing to compare against
            continue
        for level in range(1, collection.grid.levels):
            cells = [collection.cells[key] for key in record.cells if key[0] == level and key in collection.cells]
            if not cells:
                continue
            try:
                coarse = collection.object_mesh(object_id, level=level).vertices
            except Exception:  # noqa: BLE001, S112 - absent at this level is a different question
                continue
            budget = max(cell.lod_error for cell in cells)
            distance = _max_surface_distance(coarse, base.vertices, base.faces)
            checked += 1
            if distance > budget:
                worst.append(f"object {object_id} at level {level} deviates {distance:.4g} against a declared {budget:.4g}")
    return Check(
        name="lod_error bounds how far a vertex moved from level 0",
        tier="geometry",
        ok=not worst,
        detail=f"{checked} object/level pairs compared, {len(worst)} exceeded their declared error",
        examples=tuple(worst[:5]),
    )


# --------------------------------------------------------------------------- #
# the two numeric helpers
# --------------------------------------------------------------------------- #


def _tree(points: npt.NDArray[np.float64]) -> Any | None:  # noqa: ANN401
    """A KD-tree over these points, or ``None`` when scipy is not installed.

    scipy arrives with trimesh, so it is normally there. The brute-force fallback exists so
    that verifying does not become the one thing that needs a dependency the format does not.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:  # pragma: no cover - scipy ships with the mesh dependencies
        return None
    return cKDTree(np.asarray(points, dtype=np.float64))


#: How many candidate triangles each query point is measured against. Chosen by nearest
#: centroid, so this trades exactness for not being O(points x faces): a point whose true
#: nearest triangle is not among the 32 whose centroids are closest to it would have to sit
#: beside a face far larger than its neighbours, which decimation does not produce.
_CANDIDATE_FACES = 32


def _max_surface_distance(
    query: npt.NDArray[np.float64], vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> float:
    """The furthest any query point sits from the surface those triangles describe.

    Point-to-triangle rather than point-to-vertex, for the reason spelled out in
    :func:`_lod_error_is_a_real_bound`. Candidates come from a KD-tree over face centroids so
    the cost is linear in the query rather than quadratic in the mesh.
    """
    query = np.asarray(query, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(query) or not len(faces):
        return 0.0

    corners = (vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]])
    centroids: npt.NDArray[np.float64] = (corners[0] + corners[1] + corners[2]) / 3.0
    tree = _tree(centroids)
    if tree is None:
        return float(max(_point_to_faces(point, corners, None) for point in query))
    wanted = min(_CANDIDATE_FACES, len(faces))
    _, candidates = tree.query(query, k=wanted)
    candidates = np.atleast_2d(candidates.reshape(len(query), -1))
    return float(max(_point_to_faces(point, corners, row) for point, row in zip(query, candidates)))


def _point_to_faces(
    point: npt.NDArray[np.float64],
    corners: tuple[npt.NDArray[np.float64], ...],
    which: npt.NDArray[np.int64] | None,
) -> float:
    """The distance from one point to the nearest of some triangles.

    The closest point of a triangle is found by clamping the barycentric coordinates of the
    perpendicular projection into the simplex -- which is exact on the face and an
    approximation on an edge or a corner, always erring *large*. That is the safe direction:
    this feeds a bound, so overstating a distance can only fail a collection loudly rather than
    pass a bad one quietly.
    """
    a, b, c = (corner if which is None else corner[which] for corner in corners)
    ab, ac, ap = b - a, c - a, point - a
    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)
    d20 = (ap * ab).sum(-1)
    d21 = (ap * ac).sum(-1)
    denominator = d00 * d11 - d01 * d01
    denominator = np.where(denominator == 0.0, 1e-20, denominator)
    v = np.clip((d11 * d20 - d01 * d21) / denominator, 0.0, 1.0)
    w = np.clip((d00 * d21 - d01 * d20) / denominator, 0.0, 1.0)
    u = np.clip(1.0 - v - w, 0.0, 1.0)
    total = u + v + w
    total = np.where(total == 0.0, 1.0, total)
    u, v, w = u / total, v / total, w / total
    closest = a * u[..., None] + b * v[..., None] + c * w[..., None]
    return float(np.linalg.norm(closest - point, axis=-1).min())


def _count_unmatched(
    query: npt.NDArray[np.float64], reference: npt.NDArray[np.float64], *, tolerance: float
) -> int:
    """How many query points have no reference point within ``tolerance``."""
    if not len(query):
        return 0
    tree = _tree(reference)
    if tree is not None:
        return int((tree.query(np.asarray(query, dtype=np.float64))[0] > tolerance).sum())
    return sum(1 for point in query if np.linalg.norm(reference - point, axis=1).min() > tolerance)


def verify_paths(collection: Collection) -> Sequence[str]:
    """Every file the collection's manifest names, in the order the writer landed them."""
    paths = [
        collection.manifest.catalog_file("cells", "catalog/cells.parquet").path,
        collection.manifest.catalog_file("objects", "catalog/objects.parquet").path,
    ]
    for level in range(collection.grid.levels):
        paths.extend(entry.path for entry in collection.manifest.level_files(level) or [])
    return paths


__all__ = ["TIERS", "Check", "VerifyReport", "verify", "verify_paths"]
