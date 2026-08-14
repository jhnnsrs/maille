"""What may be handed to the builder as an object, and how it becomes vertices and faces.

Two shapes are accepted, and they are the same shape underneath:

- anything carrying ``.vertices`` and ``.faces`` -- a ``trimesh.Trimesh`` is what a mesh
  extractor usually hands you, and :class:`HasVerticesAndFaces` is that requirement written
  down rather than the concrete class;
- a plain ``(vertices, faces)`` pair of numpy arrays, for a caller who already has them and
  should not need trimesh in *their* code to pass them.

trimesh is a dependency of maille rather than an extra -- ``slice_mesh_plane`` is how a mesh is
cut at the cell planes, and it is not a function worth reimplementing -- so it is imported
plainly here. What the second shape buys is that the dependency stays *maille's*: a caller with
vertices and faces in hand never has to build a ``Trimesh`` to hand them over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
import trimesh


@runtime_checkable
class HasVerticesAndFaces(Protocol):
    """Anything carrying ``.vertices`` and ``.faces``, which is what a mesh is here.

    Stated as a protocol rather than as ``trimesh.Trimesh`` because that is what
    :func:`coerce_mesh` actually asks for: a ``Trimesh`` satisfies it, and so does a scene
    object, a subclass, or whatever else a caller's own pipeline hands around. Narrowing this
    to the concrete class would reject calls the code deliberately supports.
    """

    @property
    def vertices(self) -> Any:  # noqa: ANN401 - array-like, and every library spells it differently
        """The ``(n, 3)`` vertex positions."""
        ...

    @property
    def faces(self) -> Any:  # noqa: ANN401 - array-like, and every library spells it differently
        """The ``(m, 3)`` triangle indices."""
        ...


#: Anything the builder accepts as one object's geometry.
MeshSource = Union["Mesh", HasVerticesAndFaces, tuple[npt.ArrayLike, npt.ArrayLike]]


@dataclass(frozen=True)
class Mesh:
    """One object's surface: ``(n, 3)`` float vertices in voxels and ``(m, 3)`` int faces."""

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]

    @property
    def bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The axis-aligned bounds, as ``(low, high)``, matching trimesh's attribute."""
        if not len(self.vertices):
            zeros = np.zeros(3, dtype=np.float64)
            return zeros, zeros.copy()
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def as_trimesh(self) -> trimesh.Trimesh:
        """Wrap as a ``trimesh.Trimesh`` for the operations that need one.

        ``process=False``: merging vertices here would move them, and the boundary argument
        rests on the writer controlling exactly when a vertex moves.
        """
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)


def coerce_mesh(source: MeshSource) -> Mesh:
    """Turn whatever was passed for one object into vertices and faces.

    Accepts a :class:`Mesh`, anything satisfying :class:`HasVerticesAndFaces` (a
    ``trimesh.Trimesh``, most often), or a ``(vertices, faces)`` pair.
    """
    if isinstance(source, Mesh):
        return source

    vertices = getattr(source, "vertices", None)
    faces = getattr(source, "faces", None)
    if vertices is None or faces is None:
        if isinstance(source, Sequence) and len(source) == 2:
            vertices, faces = source[0], source[1]
        else:
            raise TypeError(
                f"An object is a trimesh.Trimesh, a maille.Mesh, or a (vertices, faces) pair; got "
                f"{type(source).__name__}."
            )

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError(f"Vertices come as an (n, 3) array of voxel coordinates, got {vertex_array.shape}.")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"Faces come as an (m, 3) array of triangle indices, got {face_array.shape}.")
    if face_array.size and (face_array.max() >= len(vertex_array) or face_array.min() < 0):
        raise ValueError(
            f"A face indexes vertex {int(face_array.max())} of {len(vertex_array)}, so these faces do not belong to "
            f"these vertices."
        )
    return Mesh(vertices=vertex_array, faces=face_array)


def coerce_objects(objects: Mapping[int, MeshSource]) -> dict[int, Mesh]:
    """Coerce a whole ``{object_id: source}`` mapping, naming the object that failed.

    The ids are the ones the objects carry in whatever they were extracted from -- a label
    volume's instance ids, say -- and they are written through to ``object_ids`` unchanged.
    """
    coerced: dict[int, Mesh] = {}
    for object_id, source in objects.items():
        try:
            coerced[int(object_id)] = coerce_mesh(source)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Object {object_id!r}: {error}") from error
    return coerced


__all__ = ["HasVerticesAndFaces", "Mesh", "MeshSource", "coerce_mesh", "coerce_objects"]
