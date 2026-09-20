"""Mutation analysis proving the fixed sequences can kill real bugs.

For every point mutation in :mod:`tests.statemodel.mutants` at least one fixed
sequence must fail *on the model side* while the unmutated model agrees with
real traitlets. This guards against assertions that would stay green if the
contract branches were broken.
"""

from __future__ import annotations

import pytest

from .harness import Operation as Op
from .harness import compare_sequences, execute_sequence
from .mutants import MUTANT_NAMES, mutate


SEQUENCES = {
    "descriptor_set": (
        "Flat",
        [
            Op("set", "a", value=5),
            Op("set", "a", value="bad"),
            Op("get", "a"),
        ],
        set(),
    ),
    "descriptor_delete": (
        "Flat",
        [
            Op("set", "a", value=5),
            Op("delete", "a"),
        ],
        set(),
    ),
    "default_cache": (
        "InhBase",
        [Op("get", "a"), Op("get", "a"), Op("get", "a")],
        set(),
    ),
    "validation_rollback": (
        "VdBase",
        [
            Op("set", "n", value=5),
            Op("hold_enter"),
            Op("set", "n", value=6),
            Op("set", "items", value=[7, 8]),
            Op("set", "n", value=-99),  # cross validator fails on batch exit
            Op("hold_exit"),
            Op("get", "n"),
            Op("get", "items"),
        ],
        {5, 6, 7},
    ),
    "notification_batching": (
        "Flat",
        [
            Op("observe", "a", observer_id=0),
            Op("hold_enter"),
            Op("set", "a", value=1),
            Op("set", "a", value=2),
            Op("set", "a", value=3),
            Op("hold_exit"),
        ],
        {5},
    ),
}


@pytest.mark.parametrize("mutant", MUTANT_NAMES)
def test_unmutated_sequences_agree_with_traitlets(mutant):
    family, ops, unordered = SEQUENCES[mutant]
    compare_sequences(family, ops, unordered_steps=unordered)


@pytest.mark.parametrize("mutant", MUTANT_NAMES)
def test_each_mutant_is_killed(mutant):
    family, ops, unordered = SEQUENCES[mutant]
    real_trace = execute_sequence(family, ops, side="real")
    with mutate(mutant):
        mutated_trace = execute_sequence(family, ops, side="model")

    differences = []
    for index, (real, mutated) in enumerate(
        zip(real_trace.outcomes, mutated_trace.outcomes)
    ):
        if (
            real.values != mutated.values
            or real.counts != mutated.counts
            or real.error != mutated.error
            or real.static_observer_hits != mutated.static_observer_hits
        ):
            differences.append((index, "state"))
        elif index in unordered:
            if _by_name(real.events) != _by_name(mutated.events):
                differences.append((index, "events"))
        elif real.events != mutated.events:
            differences.append((index, "events"))

    assert differences, f"mutation {mutant!r} was not detected by any fixed step"


def _by_name(events):
    grouped = {}
    for name, old, new in events:
        grouped.setdefault(name, []).append((old, new))
    return grouped
