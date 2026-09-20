"""Build reference-model classes (:class:`RefClass`) from the shared shapes."""

from __future__ import annotations

from typing import Any

from .model import RefClass, RefTrait, TraitRefError
from .shapes import Shape, TraitErrorHook, TraitSpec


def build_ref_trait(spec: TraitSpec) -> RefTrait:
    rdefault = _convert_default(spec.default)
    if spec.kind == "int":
        trait = RefTrait("int", rdefault, metadata=dict(spec.metadata))
    elif spec.kind == "unicode":
        trait = RefTrait("unicode", rdefault, metadata=dict(spec.metadata))
    elif spec.kind == "list":
        inner = RefTrait("int")
        inner.name_attr = None
        trait = RefTrait("list", rdefault, inner=inner, metadata=dict(spec.metadata))
    elif spec.kind == "union":
        assert isinstance(spec.inner_kind, tuple)
        children: list[RefTrait] = []
        for idx, kind in enumerate(spec.inner_kind):
            meta = {**dict(spec.metadata), "branch": kind, "branch_index": idx}
            child = RefTrait(kind, metadata=meta)
            child.name_attr = None
            children.append(child)
        trait = RefTrait("union", rdefault, inner=children, metadata=dict(spec.metadata))
    else:  # pragma: no cover
        raise AssertionError(spec.kind)
    return trait


def _convert_default(default: Any):
    from .model import MISSING

    if default is ...:
        return MISSING
    return default


def _make_default_method(hook: str, hooks):
    def method(self_ref):
        return hooks[hook](self_ref)

    method.__name__ = "_default_" + hook
    return method


def _make_validator_method(hook: str, hooks):
    def method(self_ref, value):
        try:
            return hooks[hook](self_ref, {"value": value})
        except TraitErrorHook as exc:
            raise TraitRefError(str(exc)) from exc

    method.__name__ = "_validate_" + hook
    return method


def build_ref_shape(shape: Shape, hooks) -> RefClass:
    classes: list[RefClass] = []
    for layer, layer_specs in enumerate(shape.layers):
        traits: dict[str, RefTrait] = {}
        for tname, spec in layer_specs.items():
            trait = build_ref_trait(spec)
            trait.name_attr = tname
            traits[tname] = trait
        default_generators: dict[str, Any] = {}
        validators: dict[str, Any] = {}
        for (hlayer, tname), hook in shape.defaults.items():
            if hlayer == layer:
                default_generators[tname] = _make_default_method(hook, hooks)
        for (hlayer, tname), hook in shape.validators.items():
            if hlayer == layer:
                validators[tname] = _make_validator_method(hook, hooks)
        if shape.layer_bases is not None:
            bases = tuple(classes[idx] for idx in shape.layer_bases[layer])
        elif layer == 0:
            bases = ()
        else:
            bases = (classes[-1],)
        cls = RefClass(
            shape.name + ("Base" if layer == 0 else str(layer)),
            traits,
            default_generators,
            validators,
            bases=bases,
        )
        classes.append(cls)
    return classes[-1]
