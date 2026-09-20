"""Independent reference model for the publicly observable behavior of a small
subset of :mod:`traitlets`.

This module is deliberately written from the *documented contract* plus
black-box probing of traitlets 5.16; it does not import or subclass any
traitlets internals.  The model supports four trait kinds (``Int``,
``Unicode``, ``List`` and ``Union``), class metadata, dynamic defaults,
cross-validators, change/default observers and the
``hold_trait_notifications`` / ``cross_validation_lock`` contexts.

Only behavior that a user of the public traitlets API can observe is modeled:

* per-instance visible values (including the "deleted" sentinel state),
* exception type, trait name and the public ``TraitError`` message,
* dynamic-default / cross-validator invocation counts,
* observer event streams, grouped by the flush boundary at which they fire.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable

MISSING = object()
ALL = object()
DELETED_SENTINEL = object()


class TraitRefError(Exception):
    """Reference-model counterpart of ``traitlets.TraitError``.

    Named ``TraitError`` on purpose: differential comparison is on the public
    exception *type name*, which must match the framework contract.
    """

    def __init__(self, *args):
        super().__init__(*args)


# Public type name deliberately mirrors the framework ("AttributeError").
class RefAttributeError(AttributeError):
    """Raised on attribute access of a deleted trait."""


@dataclass
class RefTrait:
    kind: str
    default: Any = MISSING
    inner: "RefTrait | list[RefTrait] | None" = None
    metadata: dict = field(default_factory=dict)
    dynamic_default: Callable | None = None
    validator: Callable | None = None


def _type_name(value: Any) -> str:
    return type(value).__name__


def _add_article(name: str) -> str:
    letters = "".join(ch for ch in name if ch.isalnum() or ch.isalpha())
    if letters[:1].lower() in "aeiou":
        return "an " + name
    return "a " + name


def describe_the(value: Any) -> str:
    """Reference counterpart of traitlets.utils.descriptions.describe('the', v)."""
    if isinstance(value, type):
        return "the " + value.__name__ + " " + value.__name__
    if type(value).__repr__ is object.__repr__:
        return f"the {_type_name(value)} at '{hex(id(value))}'"
    return "the " + _type_name(value) + " " + repr(value)


class RefClass:
    """A frozen per-shape class description (the model's metadata)."""

    def __init__(
        self,
        name: str,
        traits: dict[str, RefTrait],
        default_generators: dict[str, Callable],
        validators: dict[str, Callable],
        bases: tuple["RefClass", ...] = (),
    ):
        self.name = name
        self.bases = bases
        # Per-layer namespaces, so MRO-scoped merge/lookup can tell what each
        # class introduced itself.
        self.__dict__["_own_traits"] = dict(traits)
        self.__dict__["_own_default_generators"] = dict(default_generators)
        self.__dict__["_own_validators"] = dict(validators)

        mro = self._mro()
        merged: dict[str, RefTrait] = {}
        gen: dict[str, Callable] = {}
        val: dict[str, Callable] = {}
        own_trait_defining: dict[str, RefClass] = {}
        # MRO runs most-derived first; iterate reversed so more-derived
        # definitions override inherited ones (Python attribute resolution).
        for cls in reversed(mro):
            for tname, trait in cls.__dict__["_own_traits"].items():
                merged[tname] = trait
                own_trait_defining[tname] = cls
            for tname, hook in cls.__dict__["_own_default_generators"].items():
                gen[tname] = hook
            for tname, hook in cls.__dict__["_own_validators"].items():
                val[tname] = hook
        self.traits = merged
        self.default_generators = gen
        self.validators = val
        self.own_defining = own_trait_defining

    def resolve_default_generator(self, name: str) -> Callable | None:
        """MRO-scoped lookup: a dynamic default only applies when it was
        defined on a (base) class of the class that introduced the trait
        instance."""
        defining = self.own_defining[name]
        mro = self._mro()
        # Mirrors the metaclass logic: the generator must be defined on the
        # trait's defining class or a class *more derived* than it (searching
        # from the concrete instance downward).  A trait override moves the
        # defining class to the subclass, hiding generators registered only
        # against the older trait instance in ancestor classes.
        index = mro.index(defining)
        for cls in mro[: index + 1]:
            gen = cls.__dict__.get("_own_default_generators", {}).get(name)
            if gen is not None:
                return gen
        return None

    def _mro(self) -> list["RefClass"]:
        # C3 linearization matching Python's MRO (most derived first) for the
        # finite shape library (single inheritance and one mixin combination).
        if not self.bases:
            return [self]
        linearizations = [list(base._mro()) for base in self.bases] + [list(self.bases)]
        return [self] + self._c3_merge(linearizations)

    @staticmethod
    def _c3_merge(sequences: list[list["RefClass"]]) -> list["RefClass"]:
        sequences = [list(seq) for seq in sequences]
        result: list[RefClass] = []
        while True:
            non_empty = [seq for seq in sequences if seq]
            if not non_empty:
                return result
            for seq in non_empty:
                candidate = seq[0]
                if not any(candidate in tail[1:] for tail in non_empty):
                    break
            result.append(candidate)
            for seq in non_empty:
                if seq[0] is candidate:
                    seq.pop(0)

    def static_initial_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, trait in self.traits.items():
            if trait.dynamic_default is not None:
                continue
            gen = self.resolve_default_generator(name)
            if gen is not None:
                continue
            default = trait_default_value(trait)
            if trait.kind == "union":
                # Union defaults are materialized lazily on first read.
                continue
            if isinstance(default, (int, str)) and default is not MISSING:
                values[name] = default
        return values


def trait_default_value(trait: RefTrait) -> Any:
    if trait.default is not MISSING:
        return trait.default
    if trait.kind == "int":
        return 0
    if trait.kind == "unicode":
        return ""
    if trait.kind == "list":
        return []
    if trait.kind == "union":
        assert isinstance(trait.inner, list)
        for sub in trait.inner:
            dv = trait_default_value(sub)
            if dv is not MISSING:
                return dv
        return MISSING
    return MISSING


@dataclass
class Event:
    name: str
    type: str
    old: Any = MISSING
    new: Any = MISSING
    # Internal bookkeeping (excluded from observable comparison): whether the
    # trait had a cached value before the assignment.  Mirrors the framework's
    # ``old_value is Undefined`` rollback test.
    had_cached_value: bool = True

    def key(self) -> tuple:
        return (self.name, self.type, self.old, self.new)


def trait_info(trait: RefTrait) -> str:
    if trait.kind == "int":
        return "an int"
    if trait.kind == "unicode":
        return "a unicode string"
    if trait.kind == "list":
        return "a list"
    return " or ".join(trait_info(sub) for sub in trait.inner)  # type: ignore[union-attr]


def chain_name(trait: RefTrait) -> str:
    return {"int": "Int", "unicode": "Unicode", "list": "List", "union": "Union"}[trait.kind]


def raise_validation_error(
    cls_name: str,
    trait: RefTrait,
    value: Any,
    obj: "RefInstance | None",
    chain: tuple[RefTrait, ...] = (),
    info: str | None = None,
) -> None:
    if chain:
        deepest, *rest = chain
        chain_parts = []
        for trait_in_chain in (deepest, *rest):
            bare = chain_name(trait_in_chain)
            chain_parts.append("an " + bare if bare[0] in "IOU" else "a " + bare)
        chain_text = " of ".join(chain_parts)
        msg = (
            f"The '{trait.name_attr}' trait of {_article(cls_name)} {cls_name} instance contains {chain_text} "
            f"which expected {info or trait_info(deepest)}, not {describe_the(value)}."
        )
    else:
        msg = (
            f"The '{trait.name_attr}' trait of {_article(cls_name)} {cls_name} instance expected "
            f"{info or trait_info(trait)}, not {describe_the(value)}."
        )
    raise TraitRefError(msg)


def _article(word: str) -> str:
    letters = "".join(ch for ch in word if ch.isalpha())
    return "an" if letters[:1].lower() in "aeiou" else "a"


def validate_trait(
    trait: RefTrait,
    obj: "RefInstance | None",
    value: Any,
    cls_name: str,
    chain: tuple[RefTrait, ...] = (),
    run_cross: bool = True,
) -> Any:
    if trait.kind == "int":
        # Int accepts bool (bool is an int subclass); matches traitlets.
        if not isinstance(value, int):
            raise_validation_error(cls_name, trait, value, obj, chain)
        return value
    if trait.kind == "unicode":
        if not isinstance(value, str):
            raise_validation_error(cls_name, trait, value, obj, chain)
        return value
    if trait.kind == "list":
        if isinstance(value, str):
            raise_validation_error(cls_name, trait, value, obj, chain)
        try:
            items = list(value)
        except TypeError:
            raise_validation_error(cls_name, trait, value, obj, chain)
        inner = trait.inner
        assert isinstance(inner, RefTrait)
        validated: list[Any] = []
        for item in items:
            try:
                validated.append(
                    validate_trait(
                        inner, obj, item, cls_name, (inner,) + chain + (trait,), run_cross=False
                    )
                )
            except TraitRefError:
                raise_validation_error(
                    cls_name,
                    trait,
                    item,
                    obj,
                    (inner,) + chain + (trait,),
                    info=trait_info(inner),
                )
        return validated
    if trait.kind == "union":
        assert isinstance(trait.inner, list)
        for sub in trait.inner:
            try:
                result = validate_trait(sub, obj, value, cls_name, chain, run_cross=False)
            except TraitRefError:
                continue
            if obj is not None and trait.name_attr is not None:
                obj.union_metadata[trait.name_attr] = dict(sub.metadata)
            return result
        raise_validation_error(cls_name, trait, value, obj, chain)
    raise AssertionError(trait.kind)


class RefInstance:
    """A live reference-model instance."""

    def __init__(self, cls: RefClass):
        self._cls = cls
        self._values: dict[str, Any] = {}
        self._deleted: set[str] = set()
        for name, value in cls.static_initial_values().items():
            self._values[name] = value
        self._notifiers: dict[Any, dict[Any, list[Callable]]] = {}
        self._cross_validation_lock = False
        self.union_metadata: dict[str, dict] = {}
        self.events: list[Event] = []
        self._hold_cache: dict[str, list[Event]] | None = None
        self._holding = False
        self.counters: dict[str, int] = {}

    # -- descriptor protocol -------------------------------------------------

    def get_trait(self, name: str) -> Any:
        if name in self._deleted:
            raise RefAttributeError(name)
        if name not in self._values:
            value = self._compute_default(name)
            self._values[name] = value
            # Default events bypass the hold buffer (the framework dispatches
            # them via _notify_observers directly), so they fire immediately.
            self._dispatch(Event(name, "default", new=value))
            return value
        return self._values[name]

    def _compute_default(self, name: str) -> Any:
        cls = self._cls
        generator = cls.resolve_default_generator(name)
        if generator is not None:
            return generator(self)
        trait = cls.traits[name]
        value = trait_default_value(trait)
        if value is MISSING:
            raise RefAttributeError(name)
        value = _fresh_copy(value)
        if trait.kind == "union":
            # The real implementation runs the union's own ``validate`` while
            # materializing the default, which records the matched branch.
            for sub in trait.inner:  # type: ignore[union-attr]
                try:
                    validate_trait(sub, self, value, self._cls.name)
                except TraitRefError:
                    continue
                self.union_metadata[name] = dict(sub.metadata)
                break
        return value

    def set_trait_value(self, name: str, value: Any) -> None:
        cls = self._cls
        if name not in cls.traits:
            raise TraitRefError(f"Class {cls.name} does not have a trait named {name}")
        trait = cls.traits[name]
        new_value = self._validate_trait_value(trait, value)
        had_cached = name in self._values
        try:
            old_value = self._values[name]
        except KeyError:
            if trait.kind in ("union", "list"):
                # Lazy-materialized traits (containers and unions): the
                # framework's descriptor default_value is Undefined here, so
                # the change event reports no usable old.
                old_value = MISSING
            else:
                # Matches TraitType.set: ``old`` is the trait descriptor's raw
                # default_value (kind default for unspecified traits), computed
                # *without* invoking the dynamic-default generator.
                old_value = trait_default_value(trait)
        self._deleted.discard(name)
        self._values[name] = new_value
        try:
            silent = bool(old_value == new_value)
        except Exception:
            silent = False
        if silent is not True:
            self._emit(Event(name, "change", old_value, new_value, had_cached))

    def _validator_for(self, name: str) -> Callable | None:
        return self._cls.validators.get(name)

    def _validate_trait_value(self, trait: RefTrait, value: Any) -> Any:
        new_value = validate_trait(trait, self, value, self._cls.name)
        validator = None if self._cross_validation_lock else self._validator_for(trait.name_attr)
        if validator is not None:
            new_value = self._run_validator(trait.name_attr, validator, new_value)
        return new_value

    def _run_validator(self, name: str, validator: Callable, value: Any) -> Any:
        # Counters are owned by the hooks themselves (shared by real/reference
        # constructions), so the model does not double-count here.
        return validator(self, value)

    def cross_validate(self, name: str) -> Any:
        validator = self._validator_for(name)
        if validator is None:
            return self._values.get(name)
        return self._run_validator(name, validator, self._values[name])

    def cross_validate_value(self, name: str, raw: Any) -> Any:
        validator = self._validator_for(name)
        if validator is None:
            return raw
        return self._run_validator(name, validator, raw)

    def set_validated(self, name: str, value: Any, cache: dict) -> None:
        """Store an already cross-validated value during a held flush.

        The framework re-runs ``TraitType.set`` here; when silent (current
        equals final) it does not emit a *new* notification, but the cached,
        compressed bunch is still delivered afterwards in ``finally``.
        """
        cached = cache.get(name)
        self._deleted.discard(name)
        self._values[name] = value
        if cached and cached[-1].type == "change":
            cached[-1].new = value

    def delete_trait(self, name: str) -> None:
        # Mirrors ``__delete__``: the cached value is replaced by the DELETED
        # sentinel (not removed).  A subsequent successful assignment then
        # reports the sentinel object as ``old`` in the change event.
        self._values[name] = DELETED_SENTINEL
        self._deleted.add(name)

    # -- observers ------------------------------------------------------------

    def observe(self, handler: Callable, names: Any, type: str = "change") -> None:
        names = self._parse_names(names)
        for name in names:
            bucket = self._notifiers.setdefault(name, {})
            nlist = bucket.setdefault(type, [])
            # Mirrors _add_notifiers: re-registering the same callable is a
            # no-op rather than a duplicate subscription.
            if handler not in nlist:
                nlist.append(handler)

    def unobserve(self, handler: Callable, names: Any, type: str = "change") -> None:
        for name in self._parse_names(names):
            try:
                self._notifiers[name][type].remove(handler)
            except KeyError:
                # A completely missing name/type bucket is a no-op in the
                # framework too; removing from an existing bucket that does
                # not contain the handler raises ValueError.
                if name in self._notifiers and type in self._notifiers[name]:
                    raise

    @staticmethod
    def _parse_names(names: Any) -> list[Any]:
        if names is ALL:
            return [ALL]
        if isinstance(names, str):
            return [names]
        return list(names)

    def _emit(self, event: Event) -> None:
        if self._holding:
            assert self._hold_cache is not None
            past = self._hold_cache.get(event.name)
            if past is None:
                self._hold_cache[event.name] = [event]
            elif past[-1].type == "change" and event.type == "change":
                past[-1].new = event.new
            else:
                past.append(event)
            return
        self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        self.events.append(event)
        callables: list[Callable] = []
        bucket = self._notifiers.get(event.name)
        if bucket:
            callables.extend(bucket.get(event.type, ()))
            callables.extend(bucket.get(ALL, ()))
        all_bucket = self._notifiers.get(ALL)
        if all_bucket:
            callables.extend(all_bucket.get(event.type, ()))
            callables.extend(all_bucket.get(ALL, ()))
        for callback in callables:
            callback(event)

    # -- contexts -------------------------------------------------------------

    @contextlib.contextmanager
    def cross_validation_lock(self):
        if self._cross_validation_lock:
            yield
            return
        self._cross_validation_lock = True
        try:
            yield
        finally:
            self._cross_validation_lock = False

    @contextlib.contextmanager
    def hold_trait_notifications(self):
        if self._cross_validation_lock:
            yield
            return
        cache: dict[str, list[Event]] = {}
        self._hold_cache = cache
        self._holding = True
        self._cross_validation_lock = True
        try:
            yield
            for name in list(cache.keys()):
                if cache[name][-1].type != "change":
                    # Only change events trigger flush cross-validation; a
                    # name whose held activity was just default materialization
                    # is skipped (its default event still drains in finally).
                    continue
                # ``getattr`` raises AttributeError for a trait left deleted
                # inside the block (delete-only traits never enter cache).
                raw = self.get_trait(name)
                value = self.cross_validate_value(name, raw)
                self.set_validated(name, value, cache)
        except TraitRefError:
            # Mirrors the framework rollback: change events are replayed in
            # reverse per name, using ``set_trait`` for a concrete old value
            # and a cache pop for the Undefined (never-materialized) case.
            # Restoring the DELETED sentinel via ``set_trait`` re-runs trait
            # type validation, which raises a *new* TraitError that replaces
            # the original and aborts the remainder of the rollback; the
            # cached notifications still drain in ``finally``.
            rollback_error: BaseException | None = None
            for name, changes in cache.items():
                if rollback_error is not None:
                    break
                for change in reversed(changes):
                    if change.type != "change":
                        continue
                    if not change.had_cached_value:
                        # Mirrors a framework rollback quirk: rather than
                        # leaving the slot empty (which would re-run a dynamic
                        # default on next read), the descriptor's kind-level
                        # default_value is left in the cache.  For Int that is
                        # 0; unions/containers were never pre-populated so the
                        # equivalent is absent there.
                        trait = self._cls.traits[name]
                        fallback = self._rollback_fallback(trait)
                        if fallback is not MISSING:
                            # The framework effectively leaves the kind-level
                            # static default in the cache after a rollback pop,
                            # even when a dynamic default generator exists.
                            self._values[name] = fallback
                        else:
                            self._values.pop(name, None)
                        self._deleted.discard(name)
                    else:
                        try:
                            self._rollback_value(name, change.old)
                        except TraitRefError as exc:
                            rollback_error = exc
                            break
            if rollback_error is not None:
                raise rollback_error
            cache.clear()
            raise
        finally:
            self._cross_validation_lock = False
            self._holding = False
            self._hold_cache = None
            for changes in cache.values():
                for change in changes:
                    self._dispatch(change)

    def _rollback_value(self, name: str, old: Any) -> None:
        # ``set_trait(name, old)`` path: type-validate the restored value.
        trait = self._cls.traits[name]
        validate_trait(trait, self, old, self._cls.name)
        if old is DELETED_SENTINEL:
            self._values[name] = DELETED_SENTINEL
            self._deleted.add(name)
        else:
            self._deleted.discard(name)
            self._values[name] = old

    def _rollback_fallback(self, trait: RefTrait) -> Any:
        # What the framework effectively leaves after popping an
        # never-materialized slot: only the immutable static kinds are
        # pre-populated at init, so Int/Unicode fall back to their kind
        # default; lazy traits (List/Union) end up absent.
        if trait.kind in ("int", "unicode"):
            return trait_default_value(trait)
        return MISSING


def _fresh_copy(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    return value
