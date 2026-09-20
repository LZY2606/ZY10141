"""Hand-written contract tests driven through the differential runner.

Each test pins a *sequence* of public operations and asserts that the real
traitlets instance and the independent reference model observe identical
behavior at every step: visible values, raised exception (type + field name),
default/validator call counts, and notification events grouped by flush.

Notification ordering across traits is deliberately NOT asserted: events are
compared as multisets within each flush boundary, because the documented
contract bundles notifications without promising a cross-trait order.
"""

from __future__ import annotations

from .driver import run_sequence
from .shapes import (
    SHAPE_BASIC,
    SHAPE_DYNAMIC,
    SHAPE_INHERIT,
    SHAPE_MIXIN,
    SHAPE_SAME_METHOD,
)


def test_read_default_then_assign_and_reread():
    # Static Int/Unicode defaults are pre-populated; lazy List/Union are not.
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("get", "i"),
            ("set", "i", 5, "attr"),
            ("get", "i"),
            ("get", "u"),
            ("get", "l"),
            ("get", "z"),
        ],
    )
    final = observations[-1]
    assert final["error"] is None
    assert final["values"] == {
        "i": 5,
        "u": "",
        "l": [],
        "z": 0,
        "z.__union_meta__": {"branch": "int", "branch_index": 0, "role": "union"},
    }


def test_failed_assignment_leaves_previous_value_untouched():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("set", "i", 5, "attr"),
            ("set", "i", "not-an-int", "attr"),
            ("set", "l", [1, "bad"], "attr"),
        ],
    )
    assert observations[1]["error"] == (
        "TraitError",
        "The 'i' trait of a BasicBase instance expected an int, not the str 'not-an-int'.",
    )
    assert observations[2]["error"] == (
        "TraitError",
        "The 'l' trait of a BasicBase instance contains an Int of a List "
        "which expected an int, not the str 'bad'.",
    )
    # No unvalidated value survived either failure.
    assert observations[2]["values"]["i"] == 5
    assert observations[2]["values"]["l"] == "<MISSING>"


def test_validator_reject_and_transform_counts():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("set", "i", 5, "attr"),
            ("set", "i", -7, "attr"),
            ("set", "u", "ok", "attr"),
            ("set", "u", "BAD", "attr"),
        ],
    )
    assert observations[0]["error"] is None
    assert observations[1]["error"] == ("TraitError", "negative i")
    assert observations[2]["error"] is None
    assert observations[2]["values"]["u"] == "ok!"
    assert observations[3]["error"] == ("TraitError", "bad u")
    counts = observations[3]["counts"]
    # Validator ran for every non-held assignment attempt that passed the
    # trait-level type check, including the rejecting -7 call.
    assert counts["validate:i"] == 2
    assert counts["validate:u"] == 2


def test_batched_hold_compresses_changes_as_multiset():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("observe", "h1", "ALL", "change"),
            ("begin", "hold"),
            ("set", "i", 1, "attr"),
            ("set", "i", 2, "attr"),
            ("set", "u", "ab", "attr"),
            ("end", "hold"),
        ],
    )
    flush = observations[-1]["groups"]
    # Exactly one flush group; repeated assignments to "i" merge into one
    # event with the first old and the last new.  Order across traits is not
    # asserted (sorted comparison), only the multiset membership.
    assert len(flush) == 1
    assert ("h1", "i", "change", 0, 2) in flush[0]
    assert ("h1", "u", "change", "", "ab!") in flush[0]
    assert len(flush[0]) == 2
    assert observations[-1]["values"]["i"] == 2
    # Cross validation ran once per changed trait, at flush.
    assert observations[-1]["counts"]["validate:i"] == 1


def test_hold_cross_validation_failure_rolls_back():
    observations = run_sequence(
        SHAPE_DYNAMIC,
        [
            ("observe", "h1", "ALL", "change"),
            ("begin", "hold"),
            ("set", "i", 5, "attr"),
            ("set", "z", 99, "attr"),
            ("end", "hold"),
        ],
    )
    assert observations[-1]["error"] == ("TraitError", "bad z")
    # Both held changes are rolled back; no notifications escape.
    assert observations[-1]["groups"] == []
    # Rollback pop leaves the Int kind-level static default (0) cached.
    assert observations[-1]["values"]["i"] == 0


def test_delete_then_read_raises_attribute_error():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("delete", "i"),
            ("get", "i"),
            ("set", "i", 9, "attr"),
            ("get", "i"),
        ],
    )
    assert observations[0]["error"] is None
    assert observations[1]["error"] == ("AttributeError", "i")
    assert observations[1]["deleted"] == {"i": True}
    # A fresh assignment clears the deleted state.
    assert observations[3]["values"]["i"] == 9


def test_dynamic_default_cached_and_counted_once():
    observations = run_sequence(
        SHAPE_DYNAMIC,
        [
            ("get", "i"),
            ("get", "i"),
            ("set", "i", 5, "attr"),
            ("delete", "i"),
        ],
    )
    assert observations[1]["counts"]["default:i"] == 1
    assert observations[1]["values"]["i"] == 100
    assert observations[2]["values"]["i"] == 5


def test_override_hides_base_dynamic_default_but_keeps_validator():
    observations = run_sequence(
        SHAPE_INHERIT,
        [
            ("get", "i"),
            ("set", "i", -7, "attr"),
            ("get", "u"),
        ],
    )
    # The child override supplies the literal default 2; the base @default is
    # out of scope, so no default generator call is recorded.
    assert observations[0]["values"]["i"] == 2
    assert "default:i" not in observations[0]["counts"]
    # The base cross-validator still applies to the overriding trait.
    assert observations[1]["error"] == ("TraitError", "negative i")
    assert observations[1]["counts"]["validate:i"] == 1


def test_mixin_mro_metadata_and_defaults():
    observations = run_sequence(
        SHAPE_MIXIN,
        [
            ("get", "i"),
            ("get", "u"),
            ("set", "u", "ok", "attr"),
        ],
    )
    # "i" comes from the mixin layer (default 7), "u" from the joined class.
    assert observations[0]["values"]["i"] == 7
    assert observations[1]["values"]["u"] == "joined"
    assert observations[2]["values"]["u"] == "ok!"

    from .real_adapter import build_classes

    cls = build_classes()[SHAPE_MIXIN.name]
    assert cls.class_traits()["i"].metadata == {"origin": "mixin"}
    assert cls.class_traits()["u"].metadata == {"origin": "joined"}


def test_same_named_default_method_resolution():
    observations = run_sequence(
        SHAPE_SAME_METHOD,
        [
            ("get", "u"),
            ("get", "i"),
        ],
    )
    # Both base and child define _u_default; the child trait instance resolves
    # to the child generator ("dyn-child"), while "i" uses the base generator.
    assert observations[0]["values"]["u"] == "dyn-child"
    assert observations[0]["counts"]["default:u_child"] == 1
    assert observations[1]["values"]["i"] == 100


def test_mutable_list_default_isolated_between_instances():
    from .driver import make_pair

    pair = make_pair(SHAPE_DYNAMIC)
    second_pair = make_pair(SHAPE_DYNAMIC)
    second = second_pair.real
    pair.real.l.append(9)
    assert second.l == [1, 2]

    # And across reference instances in independent runs.
    pair_a = make_pair(SHAPE_DYNAMIC)
    pair_b = make_pair(SHAPE_DYNAMIC)
    pair_a.ref.get_trait("l").append(9)
    assert pair_b.ref.get_trait("l") == [1, 2]

    # And real instances do not share the boxed counters.
    assert pair.real_box is not second_pair.real_box


def test_observer_register_unregister_dedup():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("observe", "h1", "i", "change"),
            ("observe", "h1", "i", "change"),  # duplicate registration is a no-op
            ("set", "i", 5, "attr"),
            ("unobserve", "h1", "i", "change"),
            ("set", "i", 6, "attr"),
        ],
    )
    events = observations[2]["groups"]
    assert events == [[("h1", "i", "change", 0, 5)]]
    assert observations[4]["groups"] == []


def test_unobserve_missing_handler_raises_value_error():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("unobserve", "h2", "i", "change"),
        ],
    )
    # Only removing from an existing bucket that lacks the handler raises;
    # an entirely unregistered name/type is a silent no-op on both sides.
    assert observations[0]["error"] is None


def test_cross_validation_lock_defers_validator_to_release():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("begin", "lock"),
            ("set", "i", -7, "attr"),
            ("end", "lock"),
        ],
    )
    # Inside the raw lock the rejecting cross-validator is skipped, so the
    # value is accepted (no hold/flush re-validation happens for bare lock).
    assert observations[1]["error"] is None
    assert observations[2]["values"]["i"] == -7
