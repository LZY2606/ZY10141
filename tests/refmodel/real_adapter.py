"""Adapter that builds real ``traitlets.HasTraits`` classes from :mod:`shapes`.

Classes are built once per process from the finite :data:`ALL_SHAPES`
library; random tests instantiate them but never create new classes, keeping
the runtime class/metaclass cache bounded.
"""

from __future__ import annotations

from typing import Any

import traitlets as T

from .shapes import ALL_SHAPES, Shape, TraitErrorHook, TraitSpec


def build_trait(spec: TraitSpec):
    metadata = dict(spec.metadata)
    if spec.kind == "int":
        trait = T.Int() if spec.default is ... else T.Int(spec.default)
    elif spec.kind == "unicode":
        trait = T.Unicode() if spec.default is ... else T.Unicode(spec.default)
    elif spec.kind == "list":
        assert spec.inner_kind == "int"
        trait = T.List(T.Int()) if spec.default is ... else T.List(T.Int(), spec.default)
    elif spec.kind == "union":
        assert isinstance(spec.inner_kind, tuple)
        children = []
        for idx, kind in enumerate(spec.inner_kind):
            child_meta = {
                **metadata,
                "branch": kind,
                "branch_index": idx,
            }
            if kind == "int":
                children.append(T.Int().tag(**child_meta))
            elif kind == "unicode":
                children.append(T.Unicode().tag(**child_meta))
            else:  # pragma: no cover - shape library is finite
                raise AssertionError(kind)
        trait = T.Union(children)
        if spec.default is not ...:
            trait.default_value = spec.default
    else:  # pragma: no cover
        raise AssertionError(spec.kind)
    if spec.kind != "union" and metadata:
        trait.tag(**metadata)
    return trait


_HOOK_SOURCE = {
    "default:i": """
@T.default('{trait}')
def {fname}(self):
    self._box.bump('default:i')
    return 100
""",
    "default:u": """
@T.default('{trait}')
def {fname}(self):
    self._box.bump('default:u')
    return 'dyn'
""",
    "default:l": """
@T.default('{trait}')
def {fname}(self):
    self._box.bump('default:l')
    return [1, 2]
""",
    "default:z": """
@T.default('{trait}')
def {fname}(self):
    self._box.bump('default:z')
    return 0
""",
    "default:u_child": """
@T.default('{trait}')
def {fname}(self):
    self._box.bump('default:u_child')
    return 'dyn-child'
""",
    "validate:i": """
@T.validate('{trait}')
def {fname}(self, proposal):
    self._box.bump('validate:i')
    if proposal['value'] == -7:
        raise T.TraitError('negative i')
    return proposal['value']
""",
    "validate:u": """
@T.validate('{trait}')
def {fname}(self, proposal):
    self._box.bump('validate:u')
    if proposal['value'] == 'BAD':
        raise T.TraitError('bad u')
    return proposal['value'] + '!'
""",
    "validate:z": """
@T.validate('{trait}')
def {fname}(self, proposal):
    self._box.bump('validate:z')
    if proposal['value'] == 99:
        raise T.TraitError('bad z')
    return proposal['value']
""",
    "validate:l": """
@T.validate('{trait}')
def {fname}(self, proposal):
    self._box.bump('validate:l')
    value = proposal['value']
    if any(v == 66 for v in value):
        raise T.TraitError('bad list')
    return value
""",
}


def _hook_namespace(shape: Shape, layer: int, kind: str, hook: str) -> dict:
    if hook.startswith("default:"):
        name = f"_{kind}_default"
    else:
        name = f"_{kind}_validate"
    ns: dict[str, Any] = {"T": T, "TraitErrorHook": TraitErrorHook}
    exec(_HOOK_SOURCE[hook].format(fname=name, trait=kind), ns)
    return {name: ns[name]}


class CounterBox:
    def __init__(self):
        self.counts: dict[str, int] = {}

    def bump(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1


def _build_shape(shape: Shape) -> type:
    classes: list[type] = []
    for layer, layer_specs in enumerate(shape.layers):
        attrs: dict[str, Any] = {}
        for tname, spec in layer_specs.items():
            trait = build_trait(spec)
            attrs[tname] = trait
        for (hlayer, tname), hook in shape.defaults.items():
            if hlayer == layer:
                attrs.update(_hook_namespace(shape, layer, tname, hook))
        for (hlayer, tname), hook in shape.validators.items():
            if hlayer == layer:
                attrs.update(_hook_namespace(shape, layer, tname, hook))
        cls_name = shape.name + ("Base" if layer == 0 else str(layer))
        if shape.layer_bases is not None:
            base_indices = shape.layer_bases[layer]
            bases = tuple(classes[idx] for idx in base_indices) or (T.HasTraits,)
        elif layer == 0:
            bases = (T.HasTraits,)
        else:
            bases = (classes[-1],)
        cls = type(cls_name, bases, attrs)
        classes.append(cls)
    return classes[-1]


_CLASS_CACHE: dict[str, type] = {}


def build_classes() -> dict[str, type]:
    if not _CLASS_CACHE:
        for shape in ALL_SHAPES:
            _CLASS_CACHE[shape.name] = _build_shape(shape)
    return dict(_CLASS_CACHE)
