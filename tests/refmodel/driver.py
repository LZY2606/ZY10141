"""Stateful differential driver.

Replays one operation sequence against both a real ``traitlets`` instance and
the independent reference model, comparing publicly observable effects after
each operation boundary.

Operations are JSON-serializable tuples, which makes failing sequences
shrinkable and persistable as ordinary regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import traitlets as T
from traitlets.traitlets import _DELETED

from .model import (
    ALL,
    DELETED_SENTINEL,
    MISSING,
    Event,
    RefAttributeError,
    RefInstance,
    TraitRefError,
)
from .real_adapter import CounterBox, build_classes
from .ref_builder import build_ref_shape
from .shapes import Shape, make_hooks

NO_OLD = "<NO_OLD>"

INT_VALUES = (0, 1, 2, 5, 42, 100, -7, 99)
UNICODE_VALUES = ("", "x", "hi", "dyn", "BAD", "ok")
LIST_VALUES = (
    [],
    [1],
    [1, 2],
    [1, "bad"],
    [1, [2]],
    [66],
    [1, 2, 3],
)
UNION_VALUES = (0, 5, 99, "x", "hi", [1], None, True)

VALUES_BY_KIND = {
    "int": INT_VALUES,
    "unicode": UNICODE_VALUES,
    "list": LIST_VALUES,
    "union": UNION_VALUES,
}

# Operations:
#   ("get", name)
#   ("set", name, value, mode)          mode in {"attr", "set_trait"}
#   ("delete", name)
#   ("begin", kind)                     kind in {"hold", "lock"}
#   ("end", kind)
#   ("observe", handler_id, target, type_)
#   ("unobserve", handler_id, target, type_)


def normalize_value(value: Any) -> Any:
    if value is MISSING:
        return "<MISSING>"
    if isinstance(value, tuple):
        return ["<tuple>", [normalize_value(v) for v in value]]
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    return value


def normalize_error(exc: BaseException) -> tuple[str, str | None]:
    """(exception type name, message).  ``AttributeError`` args is the name."""
    if isinstance(exc, TraitRefError):
        name = "TraitError"
    elif isinstance(exc, RefAttributeError):
        name = "AttributeError"
    else:
        name = type(exc).__name__
    if isinstance(exc, AttributeError):
        return ("AttributeError", str(exc.args[0]) if exc.args else "")
    message = str(exc)
    # The DELETED/Undefined sentinels render as "object at '0x...'"; the
    # address is not a stable part of the contract.
    import re

    message = re.sub(r"object at '0x[0-9a-fA-F]+'", "object at '<SENTINEL>'", message)
    return (name, message)


@dataclass
class Observation:
    values: dict
    deleted: dict
    error: tuple | None
    counts: dict
    # list of flush groups; each group is a sorted list of event tuples so the
    # comparison never depends on notification *order* across traits
    groups: list

    def as_dict(self) -> dict:
        return {
            "values": self.values,
            "deleted": self.deleted,
            "error": self.error,
            "counts": self.counts,
            "groups": self.groups,
        }


@dataclass
class Pair:
    shape: Shape
    real: Any
    ref: RefInstance
    real_box: CounterBox
    ref_box: Any
    real_log: list = field(default_factory=list)
    ref_log: list = field(default_factory=list)
    real_mark: int = 0
    ref_mark: int = 0
    real_groups: list = field(default_factory=list)
    ref_groups: list = field(default_factory=list)
    real_handlers: dict = field(default_factory=dict)
    ref_handlers: dict = field(default_factory=dict)

    def trait_kind(self, name: str) -> str:
        for layer in self.shape.layers:
            if name in layer:
                return layer[name].kind
        return self.shape.layers[0][name].kind


def make_pair(shape: Shape) -> Pair:
    real_cls = build_classes()[shape.name]
    hooks, ref_box = make_hooks()
    ref_cls = build_ref_shape(shape, hooks)
    real_box = CounterBox()

    real_obj = real_cls()
    real_obj._box = real_box
    ref_obj = RefInstance(ref_cls)
    ref_obj.counters = ref_box.counts

    pair = Pair(shape, real_obj, ref_obj, real_box, ref_box)

    for handler_id in ("h1", "h2"):
        def make_real(handler_id=handler_id):
            def real_handler(change):
                pair.real_log.append(
                    (
                        handler_id,
                        change["name"],
                        change["type"],
                        change.get("old", NO_OLD),
                        change.get("new", change.get("value", NO_OLD)),
                    )
                )

            real_handler.__name__ = "real_" + handler_id
            return real_handler

        def make_ref(handler_id=handler_id):
            def ref_handler(event: Event):
                old = event.old if event.old is not MISSING else NO_OLD
                new = event.new if event.new is not MISSING else NO_OLD
                pair.ref_log.append((handler_id, event.name, event.type, old, new))

            ref_handler.__name__ = "ref_" + handler_id
            return ref_handler

        pair.real_handlers[handler_id] = make_real()
        pair.ref_handlers[handler_id] = make_ref()
    return pair


# -- event groups -------------------------------------------------------------


def _norm_events(events: list) -> list:
    normalized = []
    for event in events:
        handler_id, name, type_, old, new = event
        if old is T.Undefined:
            old = NO_OLD
        if old is _DELETED or old is DELETED_SENTINEL:
            old = "<DELETED_SENTINEL>"
        normalized.append(
            (handler_id, name, type_, normalize_value(old), normalize_value(new))
        )
    return sorted(normalized, key=lambda e: (e[0], e[1], e[2], repr(e[3]), repr(e[4])))


def _drain_groups(log: list, mark: int) -> tuple[list, int, list]:
    new_events = log[mark:]
    group = _norm_events(new_events) if new_events else None
    return log, len(log), group


# -- value snapshots ----------------------------------------------------------


def _real_snapshot(pair: Pair) -> tuple[dict, dict]:
    values, deleted = {}, {}
    for name in pair.shape.traits:
        sentinel = pair.real._trait_values.get(name)
        if sentinel is _DELETED:
            deleted[name] = True
            values[name] = "<DELETED>"
        elif name in pair.real._trait_values:
            values[name] = normalize_value(pair.real._trait_values[name])
        else:
            values[name] = MISSING_STR
    for name in pair.shape.traits:
        if pair.trait_kind(name) == "union":
            meta = getattr(pair.real, "_" + name + "_metadata", None)
            if meta is not None:
                values[name + ".__union_meta__"] = normalize_value(meta)
    return values, deleted


MISSING_STR = "<MISSING>"


def _ref_snapshot(pair: Pair) -> tuple[dict, dict]:
    values, deleted = {}, {}
    for name in pair.shape.traits:
        if name in pair.ref._deleted:
            deleted[name] = True
            values[name] = "<DELETED>"
        elif name in pair.ref._values:
            values[name] = normalize_value(pair.ref._values[name])
        else:
            values[name] = MISSING_STR
    for name in pair.shape.traits:
        if pair.trait_kind(name) == "union":
            meta = pair.ref.union_metadata.get(name)
            if meta is not None:
                values[name + ".__union_meta__"] = normalize_value(meta)
    return values, deleted


# -- operation execution ------------------------------------------------------


@dataclass
class _Ctx:
    hold: Any = None
    lock: Any = None


class _EndContext(Exception):
    """Marker: the exception (if any) is raised while leaving a context."""


def _parse_target(target: Any):
    if target == "ALL":
        return T.All, ALL
    return target, target


def _exec_one(pair: Pair, op: tuple, ctx: _Ctx, side: str) -> None:
    kind = op[0]
    real, ref = pair.real, pair.ref
    if kind == "get":
        name = op[1]
        if side == "real":
            getattr(real, name)
        else:
            ref.get_trait(name)
    elif kind == "set":
        name, value, mode = op[1], op[2], op[3]
        target = real if side == "real" else ref
        if mode == "attr":
            if side == "real":
                setattr(target, name, value)
            else:
                target.set_trait_value(name, value)
        else:
            if side == "real":
                target.set_trait(name, value)
            else:
                target.set_trait_value(name, value)
    elif kind == "delete":
        target = real if side == "real" else ref
        if side == "real":
            delattr(target, op[1])
        else:
            target.delete_trait(op[1])
    elif kind == "begin":
        which = op[1]
        target = real if side == "real" else ref
        if which == "hold":
            ctx_obj = target.hold_trait_notifications()
            ctx_obj.__enter__()
            _set_ctx(ctx, which, ctx_obj)
        else:
            if side == "real":
                ctx_obj = target.cross_validation_lock
            else:
                ctx_obj = target.cross_validation_lock()
            ctx_obj.__enter__()
            _set_ctx(ctx, which, ctx_obj)
    elif kind == "end":
        which = op[1]
        held = ctx.hold if which == "hold" else ctx.lock
        held.__exit__(None, None, None)
        _set_ctx(ctx, which, None)
    elif kind == "observe":
        handler_id, target, type_ = op[1], op[2], op[3]
        inst = real if side == "real" else ref
        rt, ft = _parse_target(target)
        handler = pair.real_handlers[handler_id] if side == "real" else pair.ref_handlers[handler_id]
        inst.observe(handler, names=(rt if side == "real" else ft), type=type_)
    elif kind == "unobserve":
        handler_id, target, type_ = op[1], op[2], op[3]
        inst = real if side == "real" else ref
        rt, ft = _parse_target(target)
        handler = pair.real_handlers[handler_id] if side == "real" else pair.ref_handlers[handler_id]
        inst.unobserve(handler, names=(rt if side == "real" else ft), type=type_)
    else:  # pragma: no cover - generated ops are closed set
        raise AssertionError(op)


def _set_ctx(ctx: _Ctx, which: str, value: Any) -> None:
    if which == "hold":
        ctx.hold = value
    else:
        ctx.lock = value


def run_sequence(shape: Shape, ops: list) -> list[dict]:
    """Run ``ops`` against fresh real/reference instances.

    Returns a list with one normalized observation per operation.  Raises
    :class:`Mismatch` on the first divergence.
    """
    pair = make_pair(shape)
    real_ctx, ref_ctx = _Ctx(), _Ctx()
    results: list[dict] = []
    for index, op in enumerate(ops):
        real_err = _attempt_real(pair, op, real_ctx)
        ref_err = _attempt_ref(pair, op, ref_ctx)
        pair.real_log, pair.real_mark, real_group = _drain_groups(
            pair.real_log, pair.real_mark
        )
        pair.ref_log, pair.ref_mark, ref_group = _drain_groups(
            pair.ref_log, pair.ref_mark
        )
        real_groups = [real_group] if real_group is not None else []
        ref_groups = [ref_group] if ref_group is not None else []
        real_obs = _observe(pair, real_err, pair.real_box.counts, real_groups, "real")
        ref_obs = _observe(pair, ref_err, pair.ref_box.counts, ref_groups, "ref")
        if real_obs != ref_obs:
            raise Mismatch(index, op, real_obs, ref_obs, shape.name)
        results.append(real_obs)
    return results


def _attempt_real(pair: Pair, op: tuple, ctx: _Ctx):
    try:
        _exec_one(pair, op, ctx, "real")
    except BaseException as exc:  # normalize *raised* public exceptions
        return normalize_error(exc)
    return None


def _attempt_ref(pair: Pair, op: tuple, ctx: _Ctx):
    try:
        _exec_one(pair, op, ctx, "ref")
    except BaseException as exc:
        return normalize_error(exc)
    return None


def _observe(pair: Pair, error, counts, groups, side):
    if side == "real":
        values, deleted = _real_snapshot(pair)
    else:
        values, deleted = _ref_snapshot(pair)
    return {
        "values": values,
        "deleted": deleted,
        "error": error,
        "counts": dict(sorted(counts.items())),
        "groups": [list(g) for g in groups],
    }


class Mismatch(AssertionError):
    def __init__(self, index, op, real_obs, ref_obs, shape_name):
        self.index = index
        self.op = op
        self.real_obs = real_obs
        self.ref_obs = ref_obs
        self.shape_name = shape_name
        super().__init__(self._format())

    def _format(self) -> str:
        import json

        return (
            f"divergence at op {self.index} ({self.op!r}) on shape {self.shape_name}\n"
            f"real: {json.dumps(self.real_obs, sort_keys=True, default=str)}\n"
            f"ref:  {json.dumps(self.ref_obs, sort_keys=True, default=str)}"
        )


# -- shrinking ----------------------------------------------------------------


def shrink(shape: Shape, ops: list) -> list:
    """Greedily reduce a failing sequence while it still fails."""
    current = list(ops)
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            candidate = current[:i] + current[i + 1 :]
            if _fails(shape, candidate):
                current = candidate
                changed = True
                break
    return current


def _fails(shape: Shape, ops: list) -> bool:
    try:
        run_sequence(shape, ops)
    except Mismatch:
        return True
    except Exception:
        return False
    return False
