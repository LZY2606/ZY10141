# Reference model vs. traitlets 5.16 contract notes

The reference model in `model.py` follows the publicly observable contract.
During construction one framework quirk was found where the implementation
diverges from the promised "a rejected assignment never leaves an
unvalidated value" guarantee. It is excluded from differential generation and
pinned by an explicit regression test instead.

## Deleting a trait, then failing cross validation inside a held batch

Minimal reproduction:

```python
from traitlets import HasTraits, Int, validate, TraitError

class Model(HasTraits):
    n = Int(5)

    @validate("n")
    def _validate_n(self, proposal):
        if proposal["value"] == -9:
            raise TraitError("no -9")
        return proposal["value"]

m = Model()
del m.n                      # stores the internal _DELETED sentinel
try:
    with m.hold_trait_notifications():
        m.n = 1
        m.n = -9             # rejected when the held batch is cross-validated
except TraitError:
    pass

assert m.trait_has_value("n") is True
m.n  # -> -9 (an unvalidated value) instead of raising AttributeError
```

On rollback `hold_trait_notifications` replays `set_trait(name, change.old)`,
and for a previously deleted trait `change.old` is the internal `_DELETED`
sentinel. `TraitType.set` then validates that sentinel, raises a fresh
`TraitError` during rollback, and the rejected value `-9` is left visible.
The reference model instead restores the deleted state without re-validating
it, so `getattr` keeps raising `AttributeError` after the failed batch.
