"""Pinned characterisation of the quirk described in DISCREPANCIES.md.

The reference model follows the documented guarantee (no unvalidated value
survives a failed assignment); real traitlets 5.16 leaks the rejected value in
one narrow rollback path. This test records both sides explicitly instead of
loosening the differential oracle.
"""

from __future__ import annotations

import traitlets

from .harness import Operation as Op
from .harness import execute_sequence


def test_framework_leaks_value_after_delete_and_failed_hold():
    from traitlets import HasTraits, Int, TraitError, validate

    class Model(HasTraits):
        n = Int(5)

        @validate("n")
        def _validate_n(self, proposal):
            if proposal["value"] == -9:
                raise TraitError("no -9")
            return proposal["value"]

    obj = Model()
    del obj.n
    try:
        with obj.hold_trait_notifications():
            obj.n = 1
            obj.n = -9
    except TraitError:
        pass
    # Documented implementation quirk: the rejected value is observable.
    assert obj.n == -9


def test_reference_model_preserves_deleted_state_on_same_sequence():
    ops = [
        Op("set", "n", value=5),
        Op("delete", "n"),
        Op("hold_enter"),
        Op("set", "n", value=1),
        Op("set", "n", value=-99),
        Op("hold_exit"),
    ]
    trace = execute_sequence("VdBase", ops, side="model")
    failed_exit = trace.outcomes[5]
    assert failed_exit.error == ("validator_error", "n")
    # The model restores the deleted state rather than leaking -99 + 1.
    assert failed_exit.values["n"] == "<DELETED>"


def test_real_side_quirk_is_distinct_from_model_contract():
    # Guard ensuring the two sides genuinely diverge on this exact trace;
    # if upstream fixes the quirk, update DISCREPANCIES.md and lift the
    # generator restriction.
    from traitlets import HasTraits, Int, TraitError, validate

    class Model(HasTraits):
        n = Int(5)

        @validate("n")
        def _validate_n(self, proposal):
            if proposal["value"] == -9:
                raise TraitError("no -9")
            return proposal["value"]

    obj = Model()
    del obj.n
    try:
        with obj.hold_trait_notifications():
            obj.n = 1
            obj.n = -9
    except TraitError:
        pass
    assert traitlets.__version__
    assert obj.trait_has_value("n")
