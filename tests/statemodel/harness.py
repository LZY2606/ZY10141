"""Replay operation sequences against real traitlets and the reference model.

Both sides produce an :class:`Outcome` per step.  Comparison is based on
publicly observable state only:

* the values visible on the instance (including ``<UNSET>``/``<DELETED>``);
* exception kind, field name and (for containers) element path;
* dynamic default and cross-validator invocation counts;
* notification sequences.

Per-name notification order is always asserted.  Cross-name ordering is only
asserted where the library commits to it (immediate, un-held notifications);
the order in which held notifications are flushed across different names is
not part of the contract, so that comparison is order-insensitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import traitlets
from traitlets.traitlets import _DELETED as REAL_DELETED

from . import factories as F
from .model import UNDEFINED, ModelError, ModelInstance


TYPE_ERROR_RE = re.compile(
    r"^The '(?P<field>[^']+)' trait(?: of .+? instance)? "
    r"(?:expected (?P<info>.+?), not |contains (?P<chain>.+?) which expected .+?, not )"
)


@dataclass(frozen=True)
class Operation:
    op: str
    field: str | None = None
    value: Any = None
    observer_id: int | None = None
    names: tuple[str, ...] = ()


@dataclass
class Outcome:
    values: dict[str, Any]
    error: Optional[tuple[Any, ...]]
    counts: dict[str, dict[str, int]]
    # change events delivered to the dynamic sink:
    # list of (name, canonical_old, canonical_new)
    events: list[tuple[str, str, str]]
    static_observer_hits: dict[str, int]


@dataclass
class Trace:
    outcomes: list[Outcome] = field(default_factory=list)


def canonical_value(value: Any) -> str:
    if value is traitlets.Undefined:
        return "<Undefined>"
    if value is UNDEFINED:
        return "<Undefined>"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(canonical_value(v) for v in value) + "]"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if value is None:
        return "None"
    return f"<{type(value).__name__}>"


def canonical_real_error(exc: BaseException) -> tuple[Any, ...]:
    if isinstance(exc, AttributeError):
        return ("attribute_error", str(exc))
    if type(exc) is not traitlets.TraitError:
        return (type(exc).__name__,)
    message = str(exc)
    validator = re.match(r"^(?P<field>[a-zA-Z_]\w*) rejects -99$", message)
    if validator:
        return ("validator_error", validator.group("field"))
    if "is read-only" in message:
        match = re.search(r'"(?P<field>[^"]+)"', message)
        return ("read_only_error", match.group("field") if match else None)
    match = TYPE_ERROR_RE.match(message)
    if match:
        field_name = match.group("field")
        if match.group("chain") is not None:
            chain = tuple(
                _strip_article(part.strip())
                for part in match.group("chain").split(" of ")
                if part
            )
            # message is "an Int of a List"; the model stores deepest first.
            return ("type_error", field_name, chain[0], tuple(reversed(chain[1:])), _tail(message))
        info = match.group("info")
        return ("type_error", field_name, info, (), _tail(message))
    return ("trait_error", message)


def _tail(message: str) -> str:
    tail = message.rsplit("not ", 1)[-1].strip()
    if tail.endswith("."):
        tail = tail[:-1]
    # describe("the", value) prefixes articles; strip only the leading article.
    tail = re.sub(r"^the ", "", tail)
    return tail


def _strip_article(part: str) -> str:
    return re.sub(r"^(an|a) ", "", part)


def canonical_model_error(exc: BaseException) -> tuple[Any, ...]:
    if isinstance(exc, AttributeError):
        return ("attribute_error", str(exc))
    if isinstance(exc, ModelError):
        return exc.key
    return (type(exc).__name__,)



# ---------------------------------------------------------------------------
# Dynamic observer sinks
# ---------------------------------------------------------------------------


class RealSink:
    """Bound dynamic observer used on the real side; identity is stable."""

    def __init__(self, record: Callable[[str, Any, Any], None]) -> None:
        self._record = record

    def __call__(self, change: Any) -> None:
        self._record(change["name"], change["old"], change["new"])


class ModelSink:
    def __init__(self, record: Callable[[str, Any, Any], None]) -> None:
        self._record = record

    def __call__(self, event: Any) -> None:
        self._record(event.name, event.old, event.new)


@dataclass
class _RealState:
    obj: Any
    resolved: Any
    events: list[tuple[str, str, str]]
    sinks: dict[int, RealSink]
    held: int = 0


@dataclass
class _ModelState:
    inst: ModelInstance
    events: list[tuple[str, str, str]]
    sinks: dict[int, ModelSink]
    held: int = 0


def _record_event(store: list[tuple[str, str, str]], name: str, old: Any, new: Any) -> None:
    store.append((name, canonical_value(old), canonical_value(new)))


def _real_snapshot(state: _RealState) -> dict[str, Any]:
    obj = state.obj
    out: dict[str, Any] = {}
    for name in state.resolved.fields:
        if name in obj._trait_values:
            value = obj._trait_values[name]
            if value is REAL_DELETED:
                out[name] = "<DELETED>"
            else:
                out[name] = value
        else:
            out[name] = "<UNSET>"
    return out


def _canon_events(events: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return list(events)


def make_real_pair(spec_name: str):
    real_cls, _model_factory, resolved = F.pair_for(spec_name)
    return real_cls, resolved


def init_real(spec_name: str) -> _RealState:
    real_cls, resolved = make_real_pair(spec_name)
    state = _RealState(obj=real_cls(), resolved=resolved, events=[], sinks={})
    F._LEDGER[state.obj] = {"default": {}, "validator": {}}
    F._OBS_LEDGER[state.obj] = {}
    return state


def init_model(spec_name: str) -> _ModelState:
    _real_cls, model_factory, resolved = F.pair_for(spec_name)
    return _ModelState(inst=model_factory(), events=[], sinks={})


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def _fields(resolved: Any) -> tuple[str, ...]:
    return tuple(resolved.fields)


def _run_real(state: _RealState, operation: Operation) -> None:
    obj = state.obj
    fields = _fields(state.resolved)
    field = operation.field
    if operation.op == "get":
        getattr(obj, field)
    elif operation.op == "set":
        setattr(obj, field, operation.value)
    elif operation.op == "set_trait":
        obj.set_trait(field, operation.value)
    elif operation.op == "delete":
        delattr(obj, field)
    elif operation.op == "observe":
        sink = RealSink(lambda n, o, nw, s=state.events: _record_event(s, n, o, nw))
        state.sinks[operation.observer_id] = sink
        names = [field] if field is not None else list(fields)
        for name in names:
            obj.observe(sink, names=name)
    elif operation.op == "unobserve":
        sink = state.sinks.pop(operation.observer_id)
        obj.unobserve(sink, names=field)
    elif operation.op == "hold_enter":
        state._cm = obj.hold_trait_notifications()
        state._cm.__enter__()
    elif operation.op == "hold_exit":
        state._cm.__exit__(None, None, None)
    elif operation.op == "lock_enter":
        state._cm = obj.cross_validation_lock
        state._cm.__enter__()
    elif operation.op == "lock_exit":
        state._cm.__exit__(None, None, None)
    else:
        raise AssertionError(operation.op)


def _run_model(state: _ModelState, operation: Operation) -> None:
    inst = state.inst
    fields = _fields(inst.model)
    field = operation.field
    if operation.op == "get":
        inst.get(field)
    elif operation.op == "set":
        inst.set_attr(field, operation.value)
    elif operation.op == "set_trait":
        inst.set_trait(field, operation.value)
    elif operation.op == "delete":
        inst.delete(field)
    elif operation.op == "observe":
        sink = ModelSink(lambda n, o, nw, s=state.events: _record_event(s, n, o, nw))
        state.sinks[operation.observer_id] = sink
        names = [field] if field is not None else list(fields)
        for name in names:
            inst.observe(sink, name)
    elif operation.op == "unobserve":
        sink = state.sinks.pop(operation.observer_id)
        inst.unobserve(sink, field)
    elif operation.op == "hold_enter":
        state._cm = inst.hold_trait_notifications()
        state._cm.__enter__()
    elif operation.op == "hold_exit":
        state._cm.__exit__(None, None, None)
    elif operation.op == "lock_enter":
        state._cm = inst.cross_validation_lock()
        state._cm.__enter__()
    elif operation.op == "lock_exit":
        state._cm.__exit__(None, None, None)
    else:
        raise AssertionError(operation.op)


def _real_outcome(state: _RealState) -> Outcome:
    return Outcome(
        values=_real_snapshot(state),
        counts=F.real_counts(state.obj),
        events=_canon_events(state.events),
        static_observer_hits=F.real_observer_counts(state.obj),
        error=None,
    )


def _model_outcome(state: _ModelState) -> Outcome:
    inst = state.inst
    return Outcome(
        values=inst.snapshot(),
        counts={kind: dict(values) for kind, values in inst.counters.items()},
        events=list(state.events),
        static_observer_hits=dict(inst.observer_counts),
        error=None,
    )


def execute_sequence(spec_name: str, operations: list[Operation], *, side: str) -> Trace:
    state = init_real(spec_name) if side == "real" else init_model(spec_name)
    runner = _run_real if side == "real" else _run_model
    outcome_builder = _real_outcome if side == "real" else _model_outcome
    canonical_error = canonical_real_error if side == "real" else canonical_model_error
    trace = Trace()
    for operation in operations:
        try:
            runner(state, operation)
        except BaseException as exc:  # captured as the observable error
            outcome = outcome_builder(state)
            outcome.error = canonical_error(exc)
        else:
            outcome = outcome_builder(state)
        trace.outcomes.append(outcome)
    return trace


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _events_by_name(events: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, old, new in events:
        grouped.setdefault(name, []).append((old, new))
    return grouped


def _unordered_events(events: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return sorted(events, key=lambda item: item[0])


def assert_outcomes_equal(
    step: int,
    operation: Operation,
    real: Outcome,
    model: Outcome,
    *,
    unordered_flush: bool,
) -> None:
    assert real.values == model.values, (
        f"step {step} {operation}: values differ\nreal : {real.values}\nmodel: {model.values}"
    )
    assert real.error == model.error, (
        f"step {step} {operation}: error differs\nreal : {real.error}\nmodel: {model.error}"
    )
    assert real.counts == model.counts, (
        f"step {step} {operation}: counts differ\nreal : {real.counts}\nmodel: {model.counts}"
    )
    assert real.static_observer_hits == model.static_observer_hits, (
        f"step {step} {operation}: static observers differ\n"
        f"real : {real.static_observer_hits}\nmodel: {model.static_observer_hits}"
    )
    if unordered_flush:
        # Only the set of notifications and the per-name sequence are promised
        # when a held batch is flushed across different trait names.
        assert _events_by_name(real.events) == _events_by_name(model.events), (
            f"step {step} {operation}: per-name events differ\n"
            f"real : {_events_by_name(real.events)}\n"
            f"model: {_events_by_name(model.events)}"
        )
    else:
        assert real.events == model.events, (
            f"step {step} {operation}: event order differs\n"
            f"real : {real.events}\nmodel: {model.events}"
        )


def compare_sequences(
    spec_name: str,
    operations: list[Operation],
    *,
    unordered_steps: set[int] | None = None,
) -> None:
    real_trace = execute_sequence(spec_name, operations, side="real")
    model_trace = execute_sequence(spec_name, operations, side="model")
    unordered_steps = unordered_steps or set()
    for index, operation in enumerate(operations):
        assert_outcomes_equal(
            index,
            operation,
            real_trace.outcomes[index],
            model_trace.outcomes[index],
            unordered_flush=index in unordered_steps,
        )
