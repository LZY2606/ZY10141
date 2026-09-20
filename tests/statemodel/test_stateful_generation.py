"""Stateful differential tests with a bounded, seeded generator."""

from __future__ import annotations

import os

import pytest

from .families import FAMILIES
from .generator import MAX_LENGTH, generate
from .harness import compare_sequences


# Fixed small sample for the default suite: every family, a handful of seeds.
DEFAULT_SEEDS = range(12)
DEFAULT_LENGTH = 24

# Long runs are opt-in so the default suite stays fast and deterministic.
LONG_SEEDS = range(12, 212)
LONG_LENGTHS = (24, MAX_LENGTH)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_generated_default_sample(family, seed):
    generated = generate(family, seed=seed, length=DEFAULT_LENGTH)
    compare_sequences(
        family, generated.operations, unordered_steps=generated.unordered_steps
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("seed", LONG_SEEDS)
@pytest.mark.parametrize("length", LONG_LENGTHS)
@pytest.mark.skipif(
    os.environ.get("STATEMODEL_LONG_RUNS") != "1",
    reason="set STATEMODEL_LONG_RUNS=1 for the long differential campaign",
)
def test_generated_long_runs(family, seed, length):
    generated = generate(family, seed=seed, length=length)
    compare_sequences(
        family, generated.operations, unordered_steps=generated.unordered_steps
    )


def test_generator_is_bounded():
    generated = generate("Flat", seed=1, length=MAX_LENGTH)
    assert len(generated.operations) <= MAX_LENGTH + 1  # + trailing context exit
    with pytest.raises(ValueError):
        generate("Flat", seed=1, length=MAX_LENGTH + 1)
    # The same seed must produce the same sequence (no global RNG drift).
    again = generate("Flat", seed=1, length=MAX_LENGTH)
    assert [tuple(op.__dict__.items()) for op in again.operations] == [
        tuple(op.__dict__.items()) for op in generated.operations
    ]


def test_no_unbounded_class_creation():
    from . import factories

    # Ensure the fixed family classes have been materialised once.
    for family in FAMILIES:
        factories.pair_for(family)
    before = dict(factories._REAL_CACHE)
    # Replaying many sequences only creates instances, never new classes.
    for seed in range(20):
        generated = generate("Flat", seed=seed, length=10)
        compare_sequences(
            "Flat", generated.operations, unordered_steps=generated.unordered_steps
        )
    assert factories._REAL_CACHE == before
    assert set(before) >= set(FAMILIES)
