"""Fixed, reduced sequences saved as ordinary tests.

These are deliberately tiny state sequences (of the kind produced by
:func:`refmodel.driver.shrink`) that pin the combinations the task calls out:
read defaults -> batch assign with one validation failure -> delete/inherit,
failed element validation, and notification batching.
"""

from __future__ import annotations

from .driver import run_sequence
from .shapes import SHAPE_BASIC, SHAPE_DYNAMIC, SHAPE_INHERIT, SHAPE_SAME_METHOD


def test_read_defaults_batch_one_failure_then_delete_inherit():
    # Single-inheritance shape: read everything, hold two changes one invalid,
    # then delete a trait and verify the state after rollback.
    observations = run_sequence(
        SHAPE_INHERIT,
        [
            ("get", "i"),
            ("get", "u"),
            ("observe", "h1", "ALL", "change"),
            ("begin", "hold"),
            ("set", "i", 1, "attr"),
            ("set", "i", -7, "attr"),
            ("end", "hold"),
            ("delete", "u"),
            ("get", "u"),
        ],
    )
    # The held cross-validation failure rolls "i" back to its static default.
    assert observations[6]["error"] == ("TraitError", "negative i")
    assert observations[6]["values"]["i"] == 2
    assert observations[7]["deleted"] == {"u": True}
    assert observations[8]["error"] == ("AttributeError", "u")


def test_failed_list_element_keeps_old_list_and_preserves_path():
    observations = run_sequence(
        SHAPE_DYNAMIC,
        [
            ("set", "l", [1, 2], "attr"),
            ("set", "l", [1, "bad"], "attr"),
            ("get", "l"),
        ],
    )
    assert observations[1]["error"] == (
        "TraitError",
        "The 'l' trait of a DynamicBase instance contains an Int of a List "
        "which expected an int, not the str 'bad'.",
    )
    # The bad whole-list assignment did not replace the validated old value.
    assert observations[2]["values"]["l"] == [1, 2]


def test_batched_notifications_merge_first_old_last_new():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("observe", "h1", "i", "change"),
            ("observe", "h2", "u", "change"),
            ("begin", "hold"),
            ("set", "i", 1, "attr"),
            ("set", "i", 2, "attr"),
            ("set", "u", "ok", "attr"),
            ("end", "hold"),
        ],
    )
    group = observations[-1]["groups"][0]
    # Two handlers/traits collapse to two merged events regardless of order.
    assert ("h1", "i", "change", 0, 2) in group
    assert ("h2", "u", "change", "", "ok!") in group
    assert len(group) == 2


def test_set_trait_and_attr_agree_for_valid_and_invalid_values():
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("set", "i", 5, "set_trait"),
            ("set", "z", "x", "set_trait"),
            ("set", "z", [1], "set_trait"),
        ],
    )
    assert observations[0]["error"] is None
    assert observations[1]["error"] is None
    assert observations[1]["values"]["z"] == "x"
    assert observations[2]["error"][0] == "TraitError"
    assert observations[2]["values"]["z"] == "x"


def test_delete_in_hold_then_later_assignment_recovers():
    observations = run_sequence(
        SHAPE_SAME_METHOD,
        [
            ("observe", "h1", "i", "change"),
            ("begin", "hold"),
            ("set", "i", 3, "attr"),
            ("delete", "i"),
            ("set", "i", 6, "attr"),
            ("end", "hold"),
            ("get", "i"),
        ],
    )
    # delete followed by a fresh assignment within the block clears the
    # deleted state, so flush succeeds and reports the final value.
    assert observations[5]["error"] is None
    assert observations[6]["values"]["i"] == 6
