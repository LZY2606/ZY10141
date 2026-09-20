"""Opt-in test wrapper around the mutation analysis.

Not run by the default suite (it execs mutated model copies).  Run with::

    REFMODEL_MUTATIONS=1 .venv/bin/python -m pytest tests/refmodel/test_refmodel_mutation.py -q
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("REFMODEL_MUTATIONS") != "1",
    reason="set REFMODEL_MUTATIONS=1 to run mutation analysis",
)


def test_all_mutations_are_killed():
    from . import mutation_analysis

    assert mutation_analysis.main() == 0
