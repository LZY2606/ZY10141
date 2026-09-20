"""Independent reference model of the observable traitlets contract.

Scope (deliberately finite):

* trait types: ``Int``, ``Unicode``, ``List`` and ``Union`` thereof;
* operations: read, (in)valid assignment, ``del``, ``set_trait``,
  ``cross_validation_lock``, ``hold_trait_notifications``, dynamic
  ``observe``/``unobserve``;
* class shapes: flat, single inheritance with overrides and mixins.

The model is table driven (field specs + linearised MRO tables).  It mirrors
the *contract* established by the library (and documented in
``DISCREPANCIES.md``), it does not copy ``HasTraits`` implementation branches.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional


class ModelError(Exception):
    """Raised by model operations, mirroring ``traitlets.TraitError``."""

    def __init__(self, key: tuple[Any, ...], message: str | None = None) -> None:
        self.key = key
        super().__init__(message if message is not None else key)


_DELETED = object()
_UNSET = object()


class _Undefined:
    def __repr__(self) -> str:
        return "Undefined"


UNDEFINED = _Undefined()


@dataclass(frozen=True)
class TraitSpec:
    """Specification of a trait type in the reference vocabulary."""

    kind: str  # one of "int", "unicode", "list", "union"
    default: Any = _UNSET
    default_kind: str = "static"  # "static" | "dynamic" | "list_factory"
    element: Optional["TraitSpec"] = None
    members: tuple["TraitSpec", ...] = ()
    read_only: bool = False
    tags: frozenset[tuple[str, Any]] = frozenset()

    def metadata(self) -> dict[str, Any]:
        return dict(self.tags)

    def with_(self, **changes: Any) -> "TraitSpec":
        data = {
            "kind": self.kind,
            "default": self.default,
            "default_kind": self.default_kind,
            "element": self.element,
            "members": self.members,
            "read_only": self.read_only,
            "tags": self.tags,
        }
        data.update(changes)
        return TraitSpec(**data)


@dataclass
class FieldSpec:
    """A named field together with optional dynamic default/validator hooks."""

    name: str
    trait: TraitSpec
    dynamic_default: bool = False
    validator: Optional[str] = None  # hook id resolved through the hook table


@dataclass
class ClassSpec:
    name: str
    bases: tuple["ClassSpec", ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    observer_fields: tuple[str, ...] = ()  # fields with a static @observe hook


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------


def _err(field_name: str | None, info: str, value: Any) -> ModelError:
    key: tuple[Any, ...] = ("type_error", field_name, info, (), _describe(value))
    if field_name is None:
        return ModelError(key)
    return ModelError(key, f"The '{field_name}' trait expected {info}, not {value!r}.")


def _describe(value: Any) -> str:
    if value is None:
        return "NoneType None"
    if isinstance(value, str):
        return f"str {value!r}"
    if isinstance(value, bool):
        return f"bool {value!r}"
    if isinstance(value, int):
        return f"int {value!r}"
    if isinstance(value, float):
        return f"float {value!r}"
    if isinstance(value, list):
        return f"list {value!r}"
    if isinstance(value, tuple):
        return f"tuple {value!r}"
    if isinstance(value, bytes):
        return f"bytes {value!r}"
    return f"{type(value).__name__} {value!r}"


def _coerce_int(spec: TraitSpec, value: Any) -> int:
    if isinstance(value, int):
        return value
    raise _err(None, "an int", value)


def _coerce_unicode(spec: TraitSpec, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("ascii", "strict")
        except UnicodeDecodeError as exc:
            raise ModelError(f"Could not decode {value!r}") from exc
    raise _err(None, "a unicode string", value)


def _coerce_list(spec: TraitSpec, value: Any) -> list[Any]:
    # List.set special-cases a bare string to a one-element list.
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise _err(None, "a list", value)
    if spec.element is None:
        return list(value)
    validated = []
    for element in value:
        try:
            validated.append(coerce(spec.element, element))
        except ModelError as error:
            raise _element_error(spec, error) from None
    return validated


def _element_error(container: TraitSpec, error: ModelError) -> ModelError:
    # Key layout matches the parsed real message
    # "contains an Int of a List which expected an int":
    # ("type_error", field, "Int", ("List",), repr(value)).
    _, _field, info, path, bad_repr = error.key
    inner_name = _CAP.get(info, info)
    path = (*path, _CAP_NAME[container.kind])
    return ModelError(("type_error", None, inner_name, path, bad_repr))


def _with_field(field_name: str, error: ModelError) -> ModelError:
    """Attach the root field name to a type error raised by a child trait."""
    kind, inner, *rest = error.key
    if kind == "type_error" and inner is None:
        return ModelError(("type_error", field_name, *rest))
    return error


def _coerce_union(spec: TraitSpec, value: Any) -> Any:
    for member in spec.members:
        try:
            return coerce(member, value)
        except ModelError:
            continue
    info = " or ".join(_info(m) for m in spec.members)
    raise _err(None, info, value)


def _info(spec: TraitSpec) -> str:
    return {
        "int": "an int",
        "unicode": "a unicode string",
        "list": "a list",
        "union": " or ".join(_info(m) for m in spec.members),
    }[spec.kind]


_CAP_NAME = {"int": "Int", "unicode": "Unicode", "list": "List", "union": "Union"}
_CAP = {"an int": "Int", "a unicode string": "Unicode", "a list": "List"}


def coerce(spec: TraitSpec, value: Any) -> Any:
    """Type validation/coercion for a single assignment value."""
    return {
        "int": _coerce_int,
        "unicode": _coerce_unicode,
        "list": _coerce_list,
        "union": _coerce_union,
    }[spec.kind](spec, value)


def static_default(spec: TraitSpec) -> Any:
    if spec.default is not _UNSET:
        return list(spec.default) if isinstance(spec.default, list) else spec.default
    if spec.kind == "list":
        return []
    if spec.kind == "int":
        return 0
    if spec.kind == "unicode":
        return ""
    if spec.kind == "union":
        return static_default(spec.members[0])
    raise AssertionError(spec.kind)


def is_preinitialised(spec: TraitSpec) -> bool:
    """Int/Unicode immutable defaults are present before the first read."""
    return spec.kind in ("int", "unicode")


def _descriptor_old(spec: TraitSpec) -> Any:
    """Old value used by a set on a field whose value was never generated."""
    if is_preinitialised(spec):
        return static_default(spec)
    if spec.kind == "list":
        return UNDEFINED
    if spec.kind == "union":
        return spec.default if spec.default is not _UNSET else UNDEFINED
    return UNDEFINED


# ---------------------------------------------------------------------------
# Resolved class tables
# ---------------------------------------------------------------------------


@dataclass
class FieldBinding:
    """Resolved metadata for a field on a concrete class."""

    spec: TraitSpec
    defined_in: str  # name of the class contributing this (overriding) trait
    default_hook: Optional[str]
    validator_hook: Optional[str]


@dataclass
class ResolvedModel:
    spec: ClassSpec
    fields: dict[str, FieldBinding]
    static_observers: dict[str, str]  # field name -> observer hook id


def resolve(spec: ClassSpec, hooks: "HookTable") -> ResolvedModel:
    """Linearise the hierarchy into the tables visible on *spec*.

    * The effective trait for a name is the one contributed by the most
      derived class (traits are replaced, metadata is not merged).
    * The dynamic default generator is searched from the leaf class down to,
      and including, the class that defined the effective trait.
    * The cross validator is the nearest handler for the name in the MRO.
    """
    chain = _mro(spec)
    fields: dict[str, FieldBinding] = {}
    defined_in: dict[str, str] = {}

    for cls in chain:
        for fs in cls.fields:
            if fs.name not in fields:
                fields[fs.name] = fs.trait
                defined_in[fs.name] = cls.name

    bindings: dict[str, FieldBinding] = {}
    for name, trait in fields.items():
        default_hook = None
        validator_hook = None
        limit = _index_of(chain, defined_in[name])
        for cls in chain[: limit + 1]:
            for fs in cls.fields:
                if fs.name != name:
                    continue
                if default_hook is None and fs.dynamic_default:
                    default_hook = f"{cls.name}.{name}.default"
        for cls in chain:
            for fs in cls.fields:
                if fs.name == name and fs.validator is not None:
                    validator_hook = fs.validator
                    break
            if validator_hook is not None:
                break
        bindings[name] = FieldBinding(trait, defined_in[name], default_hook, validator_hook)

    static_observers: dict[str, str] = {}
    for cls in chain:
        for name in cls.observer_fields:
            static_observers.setdefault(name, f"{cls.name}.{name}.observer")

    return ResolvedModel(spec, bindings, static_observers)


def _index_of(chain: list[ClassSpec], name: str) -> int:
    for idx, cls in enumerate(chain):
        if cls.name == name:
            return idx
    raise AssertionError(name)


def _mro(spec: ClassSpec) -> list[ClassSpec]:
    """C3-free linearisation: DFS preorder without repeats.

    The generated hierarchies are simple enough (flat / single inheritance /
    cooperative mixins without diamonds) for DFS order to agree with the
    observable MRO used by the real library.
    """
    ordered: list[ClassSpec] = []

    def visit(cls: ClassSpec) -> None:
        if cls in ordered:
            return
        ordered.append(cls)
        for base in cls.bases:
            visit(base)

    visit(spec)
    return ordered


HookTable = dict[str, Callable[..., Any]]


@dataclass
class Event:
    kind: str  # "change" | "default"
    name: str
    old: Any = _UNSET
    new: Any = _UNSET


class ModelInstance:
    """Stateful runtime instance driven by a resolved class table."""

    def __init__(self, resolved: ResolvedModel, hooks: HookTable) -> None:
        self.model = resolved
        self.hooks = hooks
        self.values: dict[str, Any] = {}
        self.locked = False
        self._hold: Optional[_HoldCache] = None
        self.counters: dict[str, dict[str, int]] = {"default": {}, "validator": {}}
        self.observer_counts: dict[str, int] = {}
        # name -> {"change": [callables]}; dynamic observers are keyed here
        self.notifiers: dict[str, dict[str, list[Callable[[Event], None]]]] = {}
        for name, binding in resolved.fields.items():
            if binding.default_hook is None and is_preinitialised(binding.spec):
                self.values[name] = static_default(binding.spec)

    # -- introspection ----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.model.fields:
            if name not in self.values:
                out[name] = "<UNSET>"
            elif self.values[name] is _DELETED:
                out[name] = "<DELETED>"
            else:
                out[name] = self.values[name]
        return out

    # -- hooks ------------------------------------------------------------

    def _default_for(self, name: str) -> Any:
        binding = self.model.fields[name]
        if binding.default_hook is not None:
            return self.hooks[binding.default_hook](self)
        return static_default(binding.spec)

    def _cross_validate(self, name: str, value: Any) -> Any:
        binding = self.model.fields[name]
        if binding.validator_hook is None:
            return value
        return self.hooks[binding.validator_hook](self, value)

    def _emit(self, event: Event) -> None:
        if event.kind == "change":
            static_hook = self.model.static_observers.get(event.name)
            if static_hook is not None:
                self.hooks[static_hook](self, event)
        for ctype in (event.kind,):
            for callback in self.notifiers.get(event.name, {}).get(ctype, []):
                callback(event)

    # -- descriptor operations -------------------------------------------

    def get(self, name: str) -> Any:
        if name not in self.model.fields:
            raise AttributeError(name)
        if self.values.get(name, _UNSET) is _DELETED:
            raise AttributeError(name)
        if name not in self.values:
            value = self._default_for(name)
            previous = self.locked
            self.locked = True
            try:
                value = coerce(self.model.fields[name].spec, value)
            except ModelError as error:
                raise _with_field(name, error) from None
            finally:
                self.locked = previous
            self.values[name] = value
            self._emit(Event("default", name, None, value))
            return value
        return self.values[name]

    def set_attr(self, name: str, value: Any) -> None:
        binding = self.model.fields[name]
        if binding.spec.read_only:
            raise ModelError(
                ("read_only_error", name), f'The "{name}" trait is read-only.'
            )
        self._set_validated(name, value)

    def set_trait(self, name: str, value: Any) -> None:
        if name not in self.model.fields:
            raise ModelError(
                f"Class {self.model.spec.name} does not have a trait named {name}"
            )
        self._set_validated(name, value)

    def delete(self, name: str) -> None:
        self.values[name] = _DELETED

    def _set_validated(self, name: str, value: Any) -> None:
        spec = self.model.fields[name].spec
        try:
            new_value = coerce(spec, value)
        except ModelError as error:
            raise _with_field(name, error) from None
        try:
            if not self.locked:
                new_value = self._cross_validate(name, new_value)
        except BaseException:
            # A rejected value never reaches the stored state.
            raise
        if name in self.values:
            old_value = self.values[name]
        else:
            # Before generation the descriptor reports Undefined as the old
            # value for trait types whose default is built lazily (lists) or
            # for types without a static default; a concrete scalar/union
            # default is still visible as the comparison old value.
            old_value = _descriptor_old(spec)
        self.values[name] = new_value
        silent = False
        try:
            silent = bool(old_value == new_value)
        except Exception:
            silent = False
        if not silent:
            self._change(name, old_value, new_value)

    def _change(self, name: str, old_value: Any, new_value: Any) -> None:
        event = Event("change", name, old_value, new_value)
        if self._hold is not None:
            self._hold.record(event)
        else:
            self._emit(event)

    # -- observers --------------------------------------------------------

    def observe(self, callback: Callable[[Event], None], name: str) -> None:
        self.notifiers.setdefault(name, {}).setdefault("change", [])
        bucket = self.notifiers[name]["change"]
        if callback not in bucket:
            bucket.append(callback)

    def unobserve(self, callback: Callable[[Event], None], name: str) -> None:
        bucket = self.notifiers.get(name, {}).get("change")
        if bucket is not None:
            bucket.remove(callback)

    # -- contexts ---------------------------------------------------------

    @contextmanager
    def cross_validation_lock(self) -> Any:  # type: ignore[override]
        if self.locked:
            yield
            return
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    @contextmanager
    def hold_trait_notifications(self) -> Any:
        if self.locked:
            yield
            return
        hold = _HoldCache()
        self._hold = hold
        self.locked = True
        try:
            yield
            for name in list(hold.order):
                value = self._cross_validate(name, self.values[name])
                # Re-enter the ordinary setter while notifications remain held:
                # the stored value is replaced and the held event compressed.
                self._set_validated(name, value)
        except BaseException:
            self._rollback(hold)
            raise
        else:
            for event in hold.flushed_events():
                self._emit(event)
        finally:
            self._hold = None
            self.locked = False

    def _rollback(self, hold: "_HoldCache") -> None:
        for name, events in hold.per_name.items():
            first = events[0]
            if first.old not in (_UNSET, UNDEFINED):
                # Restore a concrete previous value without validation.
                self.values[name] = first.old
            else:
                self.values.pop(name, None)


class _HoldCache:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.per_name: dict[str, list[Event]] = {}

    def record(self, event: Event) -> None:
        if event.name not in self.per_name:
            self.order.append(event.name)
            self.per_name[event.name] = [event]
            return
        prior = self.per_name[event.name]
        if prior[-1].kind == "change" and event.kind == "change":
            prior[-1].new = event.new
        else:
            prior.append(event)

    def flushed_events(self) -> list[Event]:
        out: list[Event] = []
        for name in self.order:
            out.extend(self.per_name[name])
        return out
