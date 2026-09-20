"""Minimal reproductions for framework quirks the reference model mirrors.

These pin *observed* traitlets 5.16 behavior that is under-specified or
surprising relative to the documented contract.  They are kept as ordinary
fixed tests (rather than treated as model bugs) so that a future traitlets
release changing any of them produces an explicit, localized failure with the
smallest possible repro attached.
"""

from __future__ import annotations

from .driver import run_sequence
from .shapes import SHAPE_BASIC, SHAPE_DYNAMIC, SHAPE_SAME_METHOD


def test_set_then_delete_inside_hold_raises_attribute_error_at_flush():
    # A cached assignment followed by delete within one held block makes the
    # flush ``getattr`` hit the DELETED sentinel -> AttributeError, while the
    # cached change notification is still delivered in ``finally``.
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("observe", "h1", "i", "change"),
            ("begin", "hold"),
            ("set", "i", 6, "attr"),
            ("delete", "i"),
            ("end", "hold"),
        ],
    )
    assert observations[-1]["error"] == ("AttributeError", "i")
    assert observations[-1]["deleted"] == {"i": True}
    assert observations[-1]["groups"] == [[("h1", "i", "change", 0, 6)]]


def test_delete_then_set_reports_deleted_sentinel_as_old():
    # After delete, the cached value is the DELETED sentinel; a successful
    # later assignment reports that sentinel (normalized here) as ``old``.
    observations = run_sequence(
        SHAPE_BASIC,
        [
            ("observe", "h1", "i", "change"),
            ("delete", "i"),
            ("set", "i", 4, "attr"),
        ],
    )
    assert observations[-1]["groups"] == [
        [("h1", "i", "change", "<DELETED_SENTINEL>", 4)]
    ]


def test_rollback_restores_kind_default_for_never_read_dynamic_trait():
    # Pop-on-rollback of an unread Int leaves the kind-level static default 0
    # cached (rather than re-running the dynamic default generator).
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
    assert observations[-1]["values"]["i"] == 0
    assert "default:i" not in observations[-1]["counts"]


def test_rollback_of_deleted_sentinel_raises_type_error_and_still_notifies():
    # delete + set inside a held block, with a later failing validation:
    # rollback replays the DELETED sentinel via set_trait, which fails type
    # validation (new TraitError replaces the original), and the cached
    # notifications are still delivered from ``finally``.
    observations = run_sequence(
        SHAPE_SAME_METHOD,
        [
            ("set", "z", "hi", "set_trait"),
            ("delete", "z"),
            ("observe", "h1", "ALL", "change"),
            ("set", "l", [66], "set_trait"),
            ("begin", "hold"),
            ("set", "z", 5, "attr"),
            ("set", "z", 99, "set_trait"),
            ("end", "hold"),
        ],
    )
    error = observations[-1]["error"]
    assert error[0] == "TraitError"
    assert "not the object at '<SENTINEL>'" in error[1]
    assert observations[-1]["values"]["z"] == 99
    groups = observations[-1]["groups"]
    assert groups and ("h1", "z", "change", "<DELETED_SENTINEL>", 99) in groups[0]
