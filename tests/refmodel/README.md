# Reference model + stateful differential tests for traitlets

This package contains a small, **independent reference model** of the publicly
observable behavior of a subset of `traitlets`, plus stateful (sequence-based)
differential tests that replay the same operations against the real
`traitlets.HasTraits` and the model and compare every observable effect.

The model is written from the documented contract and black-box probing of
traitlets 5.16; it does **not** import, subclass from, or copy branches of
`HasTraits` / `TraitType`.

## Files

- `model.py` — the reference model: `RefClass` metadata, `RefInstance`
  descriptor semantics (`get`/set/delete), dynamic defaults, per-trait
  cross-validators, change/default observers, and the
  `hold_trait_notifications` / `cross_validation_lock` contexts.
- `shapes.py` — a **fixed, finite** class-shape library (flat, single-level
  inheritance with override, mixin, same-named hook methods). Generating tests
  never creates classes; the library is built once and reused, so class and
  metaclass caches cannot grow unboundedly.
- `real_adapter.py` — builds real `HasTraits` classes (one per shape, cached).
- `ref_builder.py` — builds the corresponding reference-model classes.
- `generator.py` — bounded deterministic op-sequence generator (matched
  begin/end stack, closed JSON-serializable op set, capped length/depth).
- `driver.py` — replays a sequence on both sides and normalizes observations.
- Tests:
  - `test_refmodel_reference.py` — hand-written contract sequences.
  - `test_refmodel_generated.py` — fixed small sample by default; long rounds
    via `REFMODEL_LONG=1`.
  - `test_refmodel_regressions.py` — reduced sequences stored as plain tests.
  - `test_refmodel_quirks.py` — minimal repros of framework quirks.
  - `test_refmodel_shrinking.py` — shrinker tests.
  - `test_refmodel_mutation.py` — opt-in mutation analysis wrapper.
- `mutation_analysis.py` — standalone dev tool (see below).

## What is compared after every operation

- visible per-instance values (including the deleted-sentinel state and the
  `_<trait>_metadata` branch selected by `Union`),
- the raised public exception: type name and field name / message,
- dynamic-default and cross-validator invocation counts,
- observer events, grouped by the flush boundary at which they fire.

Events within one flush group are compared as a **multiset** (sorted), because
the documented contract bundles held notifications without promising an
ordering across traits; we do not pin incidental cross-trait order.

## Running

```bash
.venv/bin/python -m pytest tests/refmodel -q          # default fixed sample
REFMODEL_LONG=1 .venv/bin/python -m pytest tests/refmodel -q   # long rounds
.venv/bin/python -m tests.refmodel.mutation_analysis  # mutation analysis
REFMODEL_MUTATIONS=1 .venv/bin/python -m pytest tests/refmodel/test_refmodel_mutation.py -q
```

Long-round knobs: `REFMODEL_SEED`, `REFMODEL_COUNT` (hard-capped at 6000),
`REFMODEL_LENGTH` (hard-capped at 80). The suite does not disable warnings,
widen exception assertions, or read wall-clock time.

## Quirks mirrored on purpose (see test_refmodel_quirks.py)

- `del trait` stores a DELETED sentinel; reading afterwards raises
  `AttributeError`, and a later successful assignment reports that sentinel as
  the change event's `old`.
- Inside a held block, set+delete on a trait makes the flush `getattr` raise
  `AttributeError` while the cached notification still drains.
- A failed flush rolls each change back in reverse; restoring the DELETED
  sentinel via `set_trait` re-runs type validation and raises a new
  `TraitError` that replaces the original, after which cached notifications
  still fire.
- Rollback of a never-materialized static-kind trait leaves the kind-level
  default cached rather than re-running a dynamic-default generator.

If a future traitlets release changes any of these, the corresponding quirk
test fails first with its minimal reproduction attached.
