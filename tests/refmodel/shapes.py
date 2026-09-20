"""A fixed, finite library of class shapes for the stateful differential tests.

Every shape is a declarative description understood by both the real
``traitlets`` builder (:mod:`tests.refmodel.real_adapter`) and the independent
reference model (:mod:`tests.refmodel.model`).  Shapes are created exactly once
at import time, so random generation never causes unbounded class creation or
``__mro_entries__`` / metaclass cache growth.

Coverage across the library:

* ``SHAPE_BASIC``       -- flat class, static defaults, a validator
* ``SHAPE_DYNAMIC``     -- dynamic defaults and a transforming validator
* ``SHAPE_INHERIT``     -- single-level inheritance with a trait *override*
                           (base dynamic default must no longer apply; base
                           cross-validator still does) and metadata
* ``SHAPE_MIXIN``       -- mixin composition and per-instance counter isolation
* ``SHAPE_SAME_METHOD`` -- same-named ``_<trait>_default`` /
                           ``_<trait>_validate`` methods at different MRO levels

Each shape exposes the trait names in ``TRAIT_NAMES``; generated operations may
only refer to those names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALL_NAMES = ("i", "u", "l", "z")


@dataclass(frozen=True)
class TraitSpec:
    kind: str
    default: Any = ...  # ellipsis == "use the trait kind default"
    inner_kind: str | tuple[str, ...] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Shape:
    name: str
    # layer -> {trait name: spec}; layer 0 is the most-base class
    layers: list[dict[str, TraitSpec]]
    # (layer, trait, hook) -> hook name; hooks are looked up in HOOKS
    defaults: dict[tuple[int, str], str] = field(default_factory=dict)
    validators: dict[tuple[int, str], str] = field(default_factory=dict)
    # explicit base layer indices per layer; None -> single-inheritance chain
    layer_bases: list[tuple[int, ...]] | None = None
    traits: tuple[str, ...] = ALL_NAMES


def _list_spec() -> TraitSpec:
    return TraitSpec("list", ..., "int", {"role": "list"})


def _union_spec() -> TraitSpec:
    return TraitSpec("union", ..., ("int", "unicode"), {"role": "union"})


# Hook functions are shared plain functions.  Both builders bind them as
# methods on the generated classes.  The counters they increment live in a
# per-run "counter box" so that instances are fully isolated.
def hook_default_i_100(self, box):  # pragma: no cover - replaced at build time
    raise NotImplementedError


def make_hooks():
    """Return fresh hook callables plus the counter box they share.

    Real and reference instances of one run share one counter box so call
    counts can be compared directly after each operation.
    """

    class Box:
        def __init__(self):
            self.counts: dict[str, int] = {}

        def bump(self, key: str) -> None:
            self.counts[key] = self.counts.get(key, 0) + 1

    box = Box()

    def default_i(self):
        box.bump("default:i")
        return 100

    def default_u_dyn(self):
        box.bump("default:u")
        return "dyn"

    def default_u_child(self):
        box.bump("default:u_child")
        return "dyn-child"

    def default_l(self):
        box.bump("default:l")
        return [1, 2]

    def default_z(self):
        box.bump("default:z")
        return 0

    def validate_i_reject_negative(self, proposal):
        value = proposal["value"] if isinstance(proposal, dict) else proposal
        box.bump("validate:i")
        if value == -7:
            raise TraitErrorHook("negative i")
        return value

    def validate_u_transform(self, proposal):
        value = proposal["value"] if isinstance(proposal, dict) else proposal
        box.bump("validate:u")
        if value == "BAD":
            raise TraitErrorHook("bad u")
        return value + "!"

    def validate_z_reject_99(self, proposal):
        value = proposal["value"] if isinstance(proposal, dict) else proposal
        box.bump("validate:z")
        if value == 99:
            raise TraitErrorHook("bad z")
        return value

    def validate_l_even(self, proposal):
        value = proposal["value"] if isinstance(proposal, dict) else proposal
        box.bump("validate:l")
        if any(v == 66 for v in value):
            raise TraitErrorHook("bad list")
        return value

    hooks = {
        "default:i": default_i,
        "default:u": default_u_dyn,
        "default:u_child": default_u_child,
        "default:l": default_l,
        "default:z": default_z,
        "validate:i": validate_i_reject_negative,
        "validate:u": validate_u_transform,
        "validate:z": validate_z_reject_99,
        "validate:l": validate_l_even,
    }
    return hooks, box


class TraitErrorHook(Exception):
    """Raised by hooks; adapters map this to the framework's TraitError."""


# ---------------------------------------------------------------------------
# The finite shape library
# ---------------------------------------------------------------------------

SHAPE_BASIC = Shape(
    name="Basic",
    layers=[
        {
            "i": TraitSpec("int", 0, metadata={"layer": "base"}),
            "u": TraitSpec("unicode", "", metadata={"layer": "base"}),
            "l": _list_spec(),
            "z": _union_spec(),
        }
    ],
    validators={(0, "i"): "validate:i", (0, "u"): "validate:u"},
)

SHAPE_DYNAMIC = Shape(
    name="Dynamic",
    layers=[
        {
            "i": TraitSpec("int", metadata={"layer": "base"}),
            "u": TraitSpec("unicode", metadata={"layer": "base"}),
            "l": TraitSpec("list", ..., "int"),
            "z": TraitSpec("union", ..., ("int", "unicode")),
        }
    ],
    defaults={(0, "i"): "default:i", (0, "u"): "default:u", (0, "l"): "default:l"},
    validators={(0, "z"): "validate:z", (0, "l"): "validate:l"},
)

SHAPE_INHERIT = Shape(
    name="Inherit",
    layers=[
        {
            "i": TraitSpec("int", 1, metadata={"layer": "base", "tag": "base"}),
            "u": TraitSpec("unicode", "base"),
            "l": TraitSpec("list", ..., "int"),
            "z": TraitSpec("union", ..., ("int", "unicode")),
        },
        {
            # Override: brand-new trait instance.  The base dynamic default
            # for "i" (below) must NOT apply, but the base validator does.
            "i": TraitSpec("int", 2, metadata={"layer": "child", "tag": "child"}),
        },
    ],
    defaults={(0, "i"): "default:i"},
    validators={(0, "i"): "validate:i", (1, "z"): "validate:z"},
)

SHAPE_MIXIN = Shape(
    name="MixinShape",
    layers=[
        {
            "i": TraitSpec("int", 0),
            "u": TraitSpec("unicode", ""),
            "l": TraitSpec("list", ..., "int"),
            "z": TraitSpec("union", ..., ("int", "unicode")),
        },
        {
            "i": TraitSpec("int", 7, metadata={"origin": "mixin"}),
        },
        {
            "u": TraitSpec("unicode", "joined", metadata={"origin": "joined"}),
        },
    ],
    layer_bases=[(), (), (1, 0)],
    validators={(2, "u"): "validate:u"},
)

SHAPE_SAME_METHOD = Shape(
    name="SameMethod",
    layers=[
        {
            "i": TraitSpec("int"),
            "u": TraitSpec("unicode"),
            "l": TraitSpec("list", ..., "int"),
            "z": TraitSpec("union", ..., ("int", "unicode")),
        },
        {
            "u": TraitSpec("unicode", "child-default"),
        },
    ],
    # Same-named default/validate methods exist at both levels; the child
    # trait instance for "u" is introduced in layer 1 so only layer>=1
    # generators are in scope.
    defaults={(0, "u"): "default:u", (1, "u"): "default:u_child", (0, "i"): "default:i"},
    validators={(0, "z"): "validate:z"},
)

ALL_SHAPES: tuple[Shape, ...] = (
    SHAPE_BASIC,
    SHAPE_DYNAMIC,
    SHAPE_INHERIT,
    SHAPE_MIXIN,
    SHAPE_SAME_METHOD,
)
SHAPES_BY_NAME = {shape.name: shape for shape in ALL_SHAPES}
