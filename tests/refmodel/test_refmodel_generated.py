"""Stateful differential tests: real traitlets vs the reference model.

The default suite runs a fixed, small sample of deterministic sequences (the
seeds and counts are pinned, so the run is reproducible and cheap).  Longer
rounds are opt-in only::

    REFMODEL_LONG=1 .venv/bin/python -m pytest tests/refmodel -q

with ``REFMODEL_SEED`` / ``REFMODEL_COUNT`` / ``REFMODEL_LENGTH`` optionally
overriding the random seed, sequence count and maximum per-sequence length.

If a sequence ever diverges it can be reduced with
:func:`refmodel.driver.shrink` and the reduced form copied into
``test_refmodel_regressions.py``; generation itself never creates classes (the
finite shape library is reused), so repeated runs cannot grow the class or
metaclass caches.
"""

from __future__ import annotations

import os

import pytest

from .driver import Mismatch, run_sequence
from .generator import generate_many
from .shapes import ALL_SHAPES

DEFAULT_SEED = 1729
DEFAULT_COUNT = 60
DEFAULT_MAX_LENGTH = 24

LONG_COUNT = 2000
LONG_MAX_LENGTH = 40


def _settings():
    long_run = os.environ.get("REFMODEL_LONG") == "1"
    seed = int(os.environ.get("REFMODEL_SEED", DEFAULT_SEED))
    if long_run:
        count = int(os.environ.get("REFMODEL_COUNT", LONG_COUNT))
        max_length = int(os.environ.get("REFMODEL_LENGTH", LONG_MAX_LENGTH))
    else:
        count = int(os.environ.get("REFMODEL_COUNT", DEFAULT_COUNT))
        max_length = int(os.environ.get("REFMODEL_LENGTH", DEFAULT_MAX_LENGTH))
    # Hard caps keep the differential suite bounded regardless of env input.
    return seed, min(count, 6000), min(max_length, 80)


SEED, COUNT, MAX_LENGTH = _settings()
SEQUENCES = generate_many(SEED, ALL_SHAPES, COUNT, MAX_LENGTH)


def _case_id(case):
    shape, _ops = case
    return shape.name


@pytest.mark.parametrize("shape,ops", SEQUENCES, ids=[f"{s.name}-{i}" for i, (s, _o) in enumerate(SEQUENCES)])
def test_generated_matches_reference(shape, ops):
    try:
        run_sequence(shape, list(ops))
    except Mismatch as mismatch:  # pragma: no cover - only on real divergence
        pytest.fail(
            "Reference model diverged from traitlets:\n"
            + str(mismatch)
            + "\n\nReduce with refmodel.driver.shrink and add the reduced "
            "sequence to test_refmodel_regressions.py.",
            pytrace=False,
        )


def test_long_mode_is_opt_in_and_capped():
    # Guards against accidentally widening the default sample.
    if os.environ.get("REFMODEL_LONG") != "1":
        assert COUNT == DEFAULT_COUNT
        assert MAX_LENGTH == DEFAULT_MAX_LENGTH
    assert COUNT <= 6000
    assert MAX_LENGTH <= 80
