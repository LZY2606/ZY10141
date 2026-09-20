"""Build matching real-traitlets classes and model hook tables from specs."""

from __future__ import annotations

import weakref
from typing import Any, Callable

import traitlets
from traitlets import HasTraits, Int, List, Unicode, Union, default, observe, validate

from .families import BAD_VALUE, DYNAMIC_DEFAULTS, FAMILIES, VALIDATORS
from .model import ClassSpec, FieldSpec, ModelError, ModelInstance, TraitSpec, resolve, _UNSET


# Per-instance accounting for hooks attached to the real classes.
_LEDGER: "weakref.WeakKeyDictionary[Any, dict[str, dict[str, int]]]" = (
    weakref.WeakKeyDictionary()
)
_OBS_LEDGER: "weakref.WeakKeyDictionary[Any, dict[str, int]]" = weakref.WeakKeyDictionary()


def _ensure_ledger(obj: Any) -> dict[str, dict[str, int]]:
    book = _LEDGER.get(obj)
    if book is None:
        book = {"default": {}, "validator": {}}
        _LEDGER[obj] = book
    return book


def _bump(obj: Any, kind: str, name: str) -> None:
    book = _ensure_ledger(obj)
    book[kind][name] = book[kind].get(name, 0) + 1


def _bump_observer(obj: Any, name: str) -> None:
    book = _OBS_LEDGER.get(obj)
    if book is None:
        book = {}
        _OBS_LEDGER[obj] = book
    book[name] = book.get(name, 0) + 1


def real_counts(obj: Any) -> dict[str, dict[str, int]]:
    book = _ensure_ledger(obj)
    return {k: dict(v) for k, v in book.items()}


def real_observer_counts(obj: Any) -> dict[str, int]:
    return dict(_OBS_LEDGER.get(obj, {}))


# ---------------------------------------------------------------------------
# Real traitlets class construction (one cached class per spec)
# ---------------------------------------------------------------------------

_REAL_CACHE: dict[str, type[HasTraits]] = {}


def _tag(trait: Any, spec: TraitSpec) -> Any:
    if spec.tags:
        return trait.tag(**dict(spec.tags))
    return trait


def _real_trait(spec: TraitSpec) -> Any:
    if spec.kind == "int":
        kwargs: dict[str, Any] = {"read_only": spec.read_only}
        if spec.default is not _UNSET:
            kwargs["default_value"] = spec.default
        return _tag(Int(**kwargs), spec)
    if spec.kind == "unicode":
        return _tag(Unicode(), spec)
    if spec.kind == "list":
        element = _real_trait(spec.element) if spec.element is not None else None
        return _tag(List(element) if element is not None else List(), spec)
    if spec.kind == "union":
        kwargs = {}
        if spec.default is not _UNSET:
            kwargs["default_value"] = spec.default
        return _tag(Union([_real_trait(m) for m in spec.members], **kwargs), spec)
    raise AssertionError(spec.kind)


def _handler_name(kind: str, class_name: str, field: str, token: str = "") -> str:
    safe = token.replace(".", "_")
    return f"_statemodel_{kind}_{class_name}_{field}_{safe}"


def _make_default_hook(class_name: str, fs: FieldSpec) -> Callable[..., Any]:
    key = f"{class_name}.{fs.name}.default"

    def hook(self: Any) -> Any:
        _bump(self, "default", fs.name)
        return DYNAMIC_DEFAULTS[key]

    hook.__name__ = _handler_name("default", class_name, fs.name)
    return hook


def _make_validate_hook(hook_id: str, field_name: str) -> Callable[..., Any]:
    add = VALIDATORS[hook_id].get("add", 0)

    def hook(self: Any, proposal: Any) -> Any:
        _bump(self, "validator", field_name)
        value = proposal["value"]
        if value == BAD_VALUE:
            raise traitlets.TraitError(f"{field_name} rejects {BAD_VALUE}")
        return value + add

    hook.__name__ = _handler_name("validate", "cls", field_name, hook_id)
    return hook


def _make_observer_hook(class_name: str, field_name: str) -> Callable[..., Any]:
    def hook(self: Any, change: Any) -> None:
        _bump_observer(self, field_name)

    hook.__name__ = _handler_name("observe", class_name, field_name)
    return hook


def build_real_class(spec: ClassSpec) -> type[HasTraits]:
    if spec.name in _REAL_CACHE:
        return _REAL_CACHE[spec.name]
    bases = tuple(build_real_class(b) for b in spec.bases) or (HasTraits,)
    namespace: dict[str, Any] = {"__module__": __name__}
    for fs in spec.fields:
        namespace[fs.name] = _real_trait(fs.trait)
        if fs.dynamic_default:
            namespace[_handler_name("default", spec.name, fs.name)] = default(fs.name)(
                _make_default_hook(spec.name, fs)
            )
        if fs.validator is not None:
            namespace[_handler_name("validate", spec.name, fs.name, fs.validator)] = (
                validate(fs.name)(_make_validate_hook(fs.validator, fs.name))
            )
    for field_name in spec.observer_fields:
        namespace[_handler_name("observe", spec.name, field_name)] = observe(field_name)(
            _make_observer_hook(spec.name, field_name)
        )
    created = type(spec.name, bases, namespace)
    _REAL_CACHE[spec.name] = created
    return created


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(spec: ClassSpec):
    resolved = resolve(spec, {})
    hooks: dict[str, Callable[..., Any]] = {}

    for name, binding in resolved.fields.items():
        if binding.default_hook is not None:
            result = DYNAMIC_DEFAULTS[binding.default_hook]

            def make_default(field: str = name, value: Any = result) -> Callable[..., Any]:
                def hook(inst: ModelInstance) -> Any:
                    inst.counters["default"][field] = (
                        inst.counters["default"].get(field, 0) + 1
                    )
                    return value

                return hook

            hooks[binding.default_hook] = make_default()

        if binding.validator_hook is not None:
            add = VALIDATORS[binding.validator_hook].get("add", 0)

            def make_validator(field: str = name, offset: int = add) -> Callable[..., Any]:
                def hook(inst: ModelInstance, value: Any) -> Any:
                    inst.counters["validator"][field] = (
                        inst.counters["validator"].get(field, 0) + 1
                    )
                    if value == BAD_VALUE:
                        raise ModelError(
                            ("validator_error", field), f"{field} rejects {BAD_VALUE}"
                        )
                    return value + offset

                return hook

            hooks[binding.validator_hook] = make_validator()

    for field_name, hook_id in resolved.static_observers.items():

        def make_observer(field: str = field_name) -> Callable[..., Any]:
            def hook(inst: ModelInstance, event: Any) -> None:
                inst.observer_counts[field] = inst.observer_counts.get(field, 0) + 1

            return hook

        hooks[hook_id] = make_observer()

    def factory() -> ModelInstance:
        inst = ModelInstance(resolved, hooks)
        return inst

    return resolved, factory


def pair_for(spec_name: str) -> tuple[type[HasTraits], Callable[[], ModelInstance], Any]:
    spec = FAMILIES[spec_name]
    real_cls = build_real_class(spec)
    resolved, model_factory = build_model(spec)
    return real_cls, model_factory, resolved
