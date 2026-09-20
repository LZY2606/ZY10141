"""Bounded set of class hierarchies used by the stateful generator.

There are exactly ``FAMILIES`` hierarchies, reused across every generated run,
so creating instances never grows a dynamic-type cache.  Each family exercises
one documented dimension:

* ``flat``      -- unrelated traits on one class;
* ``inherit``   -- single inheritance, inherited dynamic defaults;
* ``override``  -- same-kind trait override with metadata replacement;
* ``cross``     -- different-kind trait override (Int -> Unicode/Union);
* ``mixin``     -- cooperative mixin contributing a field and an observer;
* ``validdyn``  -- same-name ``@validate`` override and ``List`` factory.
"""

from __future__ import annotations

from .model import ClassSpec, FieldSpec, TraitSpec

INT = TraitSpec("int")
STR = TraitSpec("unicode")
LIST_INT = TraitSpec("list", element=TraitSpec("int"))
LIST_STR = TraitSpec("list", element=TraitSpec("unicode"))
UNION = TraitSpec("union", default=0, members=(TraitSpec("int"), TraitSpec("unicode")))


def _dyn(name: str) -> FieldSpec:
    return FieldSpec(name, TraitSpec("int"), dynamic_default=True)


FAMILIES: dict[str, ClassSpec] = {}


def _register(spec: ClassSpec) -> ClassSpec:
    FAMILIES[spec.name] = spec
    return spec


# 1. Flat hierarchy ----------------------------------------------------------
_register(
    ClassSpec(
        "Flat",
        fields=(
            FieldSpec("a", INT.with_(tags=frozenset([("group", "scalar")]))),
            FieldSpec("s", STR.with_(tags=frozenset([("group", "text")]))),
            FieldSpec("items", LIST_INT),
            FieldSpec("u", UNION),
            FieldSpec("ro", TraitSpec("int", default=7, read_only=True)),
        ),
        observer_fields=("a",),
    )
)

# 2. Single inheritance with inherited dynamic defaults ----------------------
_base = ClassSpec(
    "InhBase",
    fields=(
        FieldSpec("a", INT, dynamic_default=True),
        FieldSpec("s", STR),
    ),
    observer_fields=("a",),
)
_register(_base)
_register(
    ClassSpec(
        "InhLeaf",
        bases=(_base,),
        fields=(
            FieldSpec("items", LIST_INT),
            FieldSpec("u", UNION),
        ),
    )
)

# 3. Same-kind override: metadata is replaced, not merged --------------------
_ov_base = ClassSpec(
    "OvBase",
    fields=(
        FieldSpec(
            "n",
            INT.with_(default=5, tags=frozenset([("layer", "base"), ("shared", "base")])),
            dynamic_default=False,
        ),
    ),
)
_register(_ov_base)
_register(
    ClassSpec(
        "OvLeaf",
        bases=(_ov_base,),
        fields=(
            FieldSpec(
                "n",
                INT.with_(default=9, tags=frozenset([("layer", "leaf")])),
            ),
        ),
        observer_fields=("n",),
    )
)

# 4. Different-kind override: Int -> Union -----------------------------------
_cr_base = ClassSpec("CrBase", fields=(FieldSpec("v", INT),))
_register(_cr_base)
_register(
    ClassSpec(
        "CrMid",
        bases=(_cr_base,),
        fields=(FieldSpec("v", STR),),
    )
)
_cr_mid = FAMILIES["CrMid"]
_register(
    ClassSpec(
        "CrLeaf",
        bases=(_cr_mid,),
        fields=(FieldSpec("v", UNION),),
    )
)

# 5. Cooperative mixin --------------------------------------------------------
_mx_base = ClassSpec("MxBase", fields=(FieldSpec("a", INT),))
_register(_mx_base)
_register(
    ClassSpec(
        "MxMixin",
        fields=(
            FieldSpec("m", STR.with_(tags=frozenset([("origin", "mixin")]))),
        ),
        observer_fields=("m",),
    )
)
_register(
    ClassSpec(
        "MxCombined",
        bases=(_mx_base, FAMILIES["MxMixin"]),
        fields=(FieldSpec("items", LIST_STR),),
        observer_fields=("a",),
    )
)

# 6. Validator override + List factory default -------------------------------
_vd_base = ClassSpec(
    "VdBase",
    fields=(
        FieldSpec("n", INT, validator="base_n_validate"),
        FieldSpec("items", TraitSpec("list", default_kind="list_factory",
                                     element=TraitSpec("int"))),
    ),
)
_register(_vd_base)
_register(
    ClassSpec(
        "VdLeaf",
        bases=(_vd_base,),
        fields=(FieldSpec("n", INT, validator="leaf_n_validate"),),
    )
)


# Static defaults for dynamic-default fields --------------------------------
DYNAMIC_DEFAULTS = {
    "InhBase.a.default": 11,
}

# Validator hook ids -> accepted/rejected values.  Hooks reject the universal
# sentinel bad value and otherwise return the proposal unchanged (a few add a
# deterministic offset so coercion is observable).
BAD_VALUE = -99

VALIDATORS = {
    "base_n_validate": {"add": 1000},
    "leaf_n_validate": {"add": 1},
}
