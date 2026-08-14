"""The errors maille raises, named so a caller can tell a bug from a missing dependency."""

from __future__ import annotations


class MailleError(Exception):
    """Base class for every error maille raises."""


class MissingExtraError(MailleError, ImportError):
    """An optional dependency is needed for what was asked and is not installed."""


class FormatError(MailleError, ValueError):
    """The bytes or the declarations do not describe a readable collection."""


class PartitioningError(FormatError):
    """Geometry does not fit the cell it was assigned to.

    Its own class because it is never a rounding problem. Quantization is per cell, so a vertex
    outside its cell cannot be represented at all -- and the tempting repair, clamping it onto
    the cell face, is what makes the bug invisible downstream.
    """


class UnfinishedCollectionError(FormatError, FileNotFoundError):
    """A prefix carries no manifest, which is what an interrupted write leaves behind.

    The manifest is written last precisely so this is distinguishable: a tree has no atomic
    "upload finished" flag, so the completion marker has to be a file that only exists once
    everything it points at does.
    """
