"""What ``import maille`` puts in front of a caller.

The package root is the API. Everything below it is a module a caller may reach into, but
``__all__`` is the part that is promised, and a name that drifts out of it is a public symbol
nobody can find or a documented one that no longer exists. Neither shows up in any other test,
because every other test imports what it needs directly.
"""

from __future__ import annotations

import maille

#: Submodules are importable and not part of the surface `__all__` describes.
_SUBMODULES = {
    "build", "codec", "errors", "frames", "geometry", "manifest",
    "planner", "reader", "simplify", "sources", "store", "writer",
}


def test_everything_promised_is_actually_there():
    """A name in ``__all__`` that does not resolve is a documented symbol that is not real."""
    missing = [name for name in maille.__all__ if not hasattr(maille, name)]

    assert not missing, f"`__all__` promises {missing}, which `import maille` does not provide"


def test_nothing_public_is_left_out_of_the_promise():
    """A public name outside ``__all__`` is reachable, undocumented, and accidentally supported.

    ``from maille import *`` will not hand it over and the docs will not list it, yet it works
    -- so callers find it, depend on it, and break when it moves. Either it belongs in
    ``__all__`` or it belongs behind an underscore.
    """
    public = {
        name
        for name in dir(maille)
        if not name.startswith("_") and name not in _SUBMODULES
    }

    assert not public - set(maille.__all__), (
        f"these are importable from `maille` but not in `__all__`: {sorted(public - set(maille.__all__))}"
    )


def test_the_promise_lists_each_name_once():
    """A duplicate is a merge that went unnoticed.

    Ordering is not checked here: ruff's ``RUF022`` already enforces it as a lint gate, and it
    sorts the isort way (constants, then classes, then functions) rather than the way
    ``sorted()`` does. Asserting a second, different order here would just make the two fight.
    """
    duplicated = sorted({name for name in maille.__all__ if maille.__all__.count(name) > 1})

    assert not duplicated, f"`__all__` lists {duplicated} more than once"
