"""Class-level dimensions: overrides, metadata, inheritance and isolation."""

from __future__ import annotations

from . import factories as F
from .families import FAMILIES


def _real(name):
    return F.build_real_class(FAMILIES[name])


def test_override_replaces_metadata_rather_than_merging():
    assert _real("OvBase").class_traits()["n"].metadata == {
        "layer": "base",
        "shared": "base",
    }
    # The overriding trait wins wholesale; parent metadata is not inherited.
    assert _real("OvLeaf").class_traits()["n"].metadata == {"layer": "leaf"}
    model = F.build_model(FAMILIES["OvLeaf"])[0]
    assert dict(model.fields["n"].spec.metadata()) == {"layer": "leaf"}


def test_dynamic_default_scope_stops_at_overriding_class():
    # OvLeaf overrides the trait, so the base dynamic default is out of scope.
    assert _real("OvLeaf")().n == 9
    assert _real("InhBase")().a == 11


def test_cross_kind_override_uses_leaf_trait():
    assert _real("CrBase")().v == 0
    assert _real("CrMid")().v == ""
    leaf = _real("CrLeaf")()
    assert leaf.v == 0
    leaf.v = "text"
    assert leaf.v == "text"


def test_mixin_contributes_field_and_observer():
    combined = _real("MxCombined")()
    assert combined.m == ""
    assert combined.a == 0
    before = F.real_observer_counts(combined).get("m", 0)
    combined.m = "x"
    assert F.real_observer_counts(combined)["m"] == before + 1
    resolved = F.build_model(FAMILIES["MxCombined"])[0]
    assert set(resolved.fields) == {"a", "m", "items"}


def test_validator_override_uses_nearest_handler():
    leaf = _real("VdLeaf")()
    leaf.n = 0
    # Leaf validator adds 1; the base validator (+1000) must not run as well.
    assert leaf.n == 1


def test_instances_are_isolated_for_values_and_observers():
    cls = _real("VdBase")
    first, second = cls(), cls()
    first.items.append(1)
    assert first.items == [1]
    assert second.items == []

    seen = []
    first.observe(lambda change: seen.append(change.new), names="n")
    second.n = 4
    first.n = 2
    assert seen == [2 + 1000]


def test_mutable_default_is_distinct_per_real_instance():
    cls = _real("VdBase")
    one, two = cls(), cls()
    one.items.append(99)
    assert one.items == [99]
    assert two.items == []
    assert one.items is not two.items
