"""Point mutations of the reference model, used to prove the tests are sharp.

Each mutation flips exactly one contract branch:

* ``descriptor_set``       -- a rejected assignment still stores its value;
* ``descriptor_delete``    -- delete behaves like an ordinary reset instead
                              of installing the deleted sentinel;
* ``default_cache``        -- dynamic defaults are recomputed on every read;
* ``validation_rollback``  -- a failed held batch keeps the proposed values;
* ``notification_batching``-- rapid changes are no longer collapsed into one
                              notification per name.
"""

from __future__ import annotations

from contextlib import contextmanager

from . import model


MUTANT_NAMES = (
    "descriptor_set",
    "descriptor_delete",
    "default_cache",
    "validation_rollback",
    "notification_batching",
)


@contextmanager
def mutate(name: str):
    if name not in MUTANT_NAMES:
        raise KeyError(name)
    if name == "descriptor_set":
        original = model.ModelInstance._set_validated

        def buggy(self, field, value):
            spec = self.model.fields[field].spec
            try:
                new_value = model.coerce(spec, value)
                if not self.locked:
                    new_value = self._cross_validate(field, new_value)
            except BaseException:
                self.values[field] = value  # BUG: leak the rejected value
                raise
            original(self, field, value)

        model.ModelInstance._set_validated = buggy
        try:
            yield
        finally:
            model.ModelInstance._set_validated = original

    elif name == "descriptor_delete":
        original = model.ModelInstance.delete

        def buggy(self, field):
            self.values.pop(field, None)  # BUG: reads silently regenerate

        model.ModelInstance.delete = buggy
        try:
            yield
        finally:
            model.ModelInstance.delete = original

    elif name == "default_cache":
        original = model.ModelInstance.get

        def buggy(self, field):
            if field not in self.model.fields:
                raise AttributeError(field)
            if self.values.get(field, model._UNSET) is model._DELETED:
                raise AttributeError(field)
            value = self._default_for(field)  # BUG: never caches the default
            return value

        model.ModelInstance.get = buggy
        try:
            yield
        finally:
            model.ModelInstance.get = original

    elif name == "validation_rollback":
        original_rollback = model.ModelInstance._rollback

        def buggy(self, hold):
            pass  # BUG: keep the values written inside the failed batch

        model.ModelInstance._rollback = buggy
        try:
            yield
        finally:
            model.ModelInstance._rollback = original_rollback

    else:  # notification_batching
        original_record = model._HoldCache.record

        def buggy(self, event):
            if event.name not in self.per_name:
                self.order.append(event.name)
                self.per_name[event.name] = [event]
            else:
                self.per_name[event.name].append(event)  # BUG: no compression

        model._HoldCache.record = buggy
        try:
            yield
        finally:
            model._HoldCache.record = original_record
