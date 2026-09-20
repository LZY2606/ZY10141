"""Tests for the greedy sequence shrinker used when a divergence is found.

The shrinker is a development-time tool; these tests exercise it directly by
monkeypatching the failure predicate so no mutated model is required.
"""

from __future__ import annotations

from . import driver
from .shapes import SHAPE_BASIC


def test_shrink_removes_ops_unrelated_to_the_failing_one(monkeypatch):
    ops = [
        ("get", "u"),
        ("set", "i", 5, "attr"),
        ("delete", "i"),
        ("get", "l"),
        ("observe", "h1", "u", "change"),
    ]

    def fake_fails(shape, candidate):
        # Pretend only sequences that delete "i" reproduce the divergence.
        return any(op == ("delete", "i") for op in candidate)

    monkeypatch.setattr(driver, "_fails", fake_fails)
    reduced = driver.shrink(SHAPE_BASIC, ops)
    assert reduced == [("delete", "i")]


def test_shrink_keeps_prefix_needed_to_reach_state(monkeypatch):
    ops = [
        ("set", "i", 5, "attr"),
        ("delete", "i"),
        ("get", "u"),
        ("get", "l"),
    ]

    def fake_fails(shape, candidate):
        # Failure needs the set *and* the delete in that order (state sequence).
        return ("set", "i", 5, "attr") in candidate and ("delete", "i") in candidate

    monkeypatch.setattr(driver, "_fails", fake_fails)
    reduced = driver.shrink(SHAPE_BASIC, ops)
    assert reduced == [("set", "i", 5, "attr"), ("delete", "i")]
