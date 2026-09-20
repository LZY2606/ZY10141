"""Hand-written fixed sequences.

These are the "saved reduced failures" / mutation-killing sequences: short,
deterministic traces targeting each contract branch. They run as part of the
default suite.
"""

from __future__ import annotations

import pytest

from .generator import generate
from .harness import Operation as Op
from .harness import compare_sequences, execute_sequence


def test_default_then_set_failure_keeps_old_value():
    # Read a static default, assign valid then invalid values; the rejected
    # assignment must not leave an unvalidated value and must notify once.
    ops = [
        Op("get", "a"),
        Op("set", "a", value=42),
        Op("set", "a", value="bad"),
        Op("get", "a"),
    ]
    compare_sequences("Flat", ops)


def test_delete_then_inherit_override_and_read():
    ops = [
        Op("set", "n", value=9),
        Op("delete", "n"),
        Op("get", "n"),
    ]
    compare_sequences("OvLeaf", ops)


def test_held_batches_collapse_same_name_notifications():
    ops = [
        Op("observe", "a", observer_id=0),
        Op("hold_enter"),
        Op("set", "a", value=1),
        Op("set", "a", value=2),
        Op("set", "a", value=3),
        Op("hold_exit"),
    ]
    trace = execute_sequence("Flat", ops, side="model")
    events = trace.outcomes[-1].events
    assert events == [("a", "0", "3")]
    compare_sequences("Flat", ops, unordered_steps={5})


def test_failed_held_batch_rolls_back_everything():
    ops = [
        Op("observe", "n", observer_id=0),
        Op("set", "n", value=5),
        Op("hold_enter"),
        Op("set", "n", value=6),
        Op("set", "items", value=[1, "x"]),  # element validation fails
        Op("hold_exit"),
        Op("get", "n"),
    ]
    compare_sequences("VdBase", ops, unordered_steps={6})


def test_cross_validator_rejects_in_held_batch():
    ops = [
        Op("observe", "n", observer_id=0),
        Op("set", "n", value=5),
        Op("hold_enter"),
        Op("set", "n", value=6),
        Op("set", "n", value=-99),  # rejected by the cross validator on exit
        Op("hold_exit"),
        Op("get", "n"),
    ]
    compare_sequences("VdBase", ops, unordered_steps={6})


def test_dynamic_default_cached_and_counted_once():
    ops = [
        Op("get", "a"),
        Op("get", "a"),
        Op("get", "a"),
    ]
    real = execute_sequence("InhBase", ops, side="real").outcomes[-1]
    model = execute_sequence("InhBase", ops, side="model").outcomes[-1]
    assert real.counts["default"] == {"a": 1}
    assert model.counts["default"] == {"a": 1}
    compare_sequences("InhBase", ops)


def test_mutable_list_default_not_shared_across_instances():
    first = execute_sequence("VdBase", [Op("set", "items", value=[1])], side="model")
    fresh = execute_sequence("VdBase", [Op("get", "items")], side="model")
    assert first.outcomes[-1].values["items"] == [1]
    # A second instance materialises its own empty list default.
    assert fresh.outcomes[0].values["items"] == []
    first_obj = first.outcomes[-1]
    assert first_obj.values["items"] == [1]


def test_list_element_error_keeps_container_path():
    ops = [Op("set", "items", value=[1, "x", 3])]
    for side in ("real", "model"):
        outcome = execute_sequence("Flat", ops, side=side).outcomes[0]
        assert outcome.error is not None
        kind, field, inner, path, tail = outcome.error
        assert kind == "type_error"
        assert field == "items"
        assert inner == "Int"
        assert path == ("List",)
        assert tail == "str 'x'"


def test_set_trait_bypasses_read_only_but_setattr_does_not():
    ops = [
        Op("set", "ro", value=1),
        Op("set_trait", "ro", value=2),
        Op("get", "ro"),
    ]
    compare_sequences("Flat", ops)


def test_override_replaces_metadata_and_default_scope():
    ops = [Op("get", "n")]
    compare_sequences("OvBase", ops)
    compare_sequences("OvLeaf", ops)


def test_removed_observer_receives_nothing():
    ops = [
        Op("observe", "a", observer_id=0),
        Op("set", "a", value=1),
        Op("unobserve", "a", observer_id=0),
        Op("set", "a", value=2),
    ]
    compare_sequences("Flat", ops)


def test_cross_validation_lock_skips_cross_validator():
    ops = [
        Op("lock_enter"),
        Op("set", "n", value=-99),
        Op("lock_exit"),
        Op("get", "n"),
    ]
    compare_sequences("VdBase", ops)


@pytest.mark.parametrize("family", ["Flat", "InhLeaf", "OvLeaf", "CrLeaf", "VdLeaf"])
def test_reduced_fixed_family_smoke(family):
    generated = generate(family, seed=0, length=12)
    compare_sequences(
        family, generated.operations, unordered_steps=generated.unordered_steps
    )
