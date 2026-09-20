"""Bounded, deterministic generator of operation sequences."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .families import BAD_VALUE, FAMILIES
from .harness import Operation


MAX_LENGTH = 40
MAX_OBSERVERS = 3


def value_pool(kind: str, element: Any = None, members: Any = ()) -> tuple[Any, ...]:
    if kind == "int":
        return (0, 1, 7, 11, 42, -3, "bad", 1.5, None, BAD_VALUE)
    if kind == "unicode":
        return ("", "a", "hello", 3, b"ok", None)
    if kind == "list":
        return ([], [1], [1, 2], (1, 2), [1, "x"], "hi", 5, None)
    if kind == "union":
        return (0, 5, "", "x", "5", 1.5, [1], None)
    raise AssertionError(kind)


@dataclass
class Generated:
    family: str
    operations: list[Operation]
    # Step indices where a held batch is flushed: cross-name order there is
    # not part of the public contract.
    unordered_steps: set[int]


def generate(
    family: str,
    seed: int,
    length: int = MAX_LENGTH,
    *,
    allow_context: bool = True,
) -> Generated:
    if length > MAX_LENGTH:
        raise ValueError("sequence length above bound")
    rng = random.Random(f"{family}:{seed}")
    spec = FAMILIES[family]
    fields = _all_field_specs(spec)
    names = tuple(fields)

    operations: list[Operation] = []
    unordered: set[int] = set()
    active_observers: dict[int, str] = {}
    next_observer = 0
    context: str | None = None  # "hold" | "lock" | None
    deleted: set[str] = set()

    def choose_field() -> str:
        return rng.choice(names)

    for _ in range(length):
        index = len(operations)
        while True:
            if context is not None and rng.random() < 0.25:
                operations.append(
                    Operation("hold_exit" if context == "hold" else "lock_exit")
                )
                if context == "hold":
                    unordered.add(index)
                context = None
                break

            choices = ["get", "set", "set", "set_trait", "observe", "unobserve"]
            if context is None:
                choices.append("delete")
            # Holding while any field is deleted exercises a documented
            # framework rollback quirk (DISCREPANCIES.md); defer the context
            # choice until deleted fields have been reassigned.
            if allow_context and context is None and not deleted:
                choices += ["hold_enter", "lock_enter"]
            if not active_observers:
                choices = [c for c in choices if c != "unobserve"]

            kind = rng.choice(choices)
            if kind in ("get", "set", "set_trait", "delete"):
                name = choose_field()
                if kind == "get":
                    operations.append(Operation("get", name))
                    break
                if kind == "delete":
                    operations.append(Operation("delete", name))
                    deleted.add(name)
                    break
                # Avoid mutating a deleted field inside a held batch: that
                # exercises a documented framework bug (DISCREPANCIES.md).
                if context == "hold" and name in deleted:
                    continue
                if kind in ("set", "set_trait") and name in deleted:
                    deleted.discard(name)
                spec_field = fields[name]
                pool = value_pool(spec_field.kind, spec_field.element, spec_field.members)
                if context == "hold":
                    # A rejected held assignment to a previously deleted field
                    # hits a documented rollback quirk; keep the rejected value
                    # available outside of held batches.
                    pool = tuple(v for v in pool if v != BAD_VALUE)
                value = rng.choice(pool)
                operations.append(Operation(kind, name, value=value))
            elif kind == "observe" and len(active_observers) < MAX_OBSERVERS:
                name = rng.choice(names + (None,))
                observer_id = next_observer
                next_observer += 1
                field_arg = name if name is not None else rng.choice(names)
                active_observers[observer_id] = field_arg
                operations.append(Operation("observe", field_arg, observer_id=observer_id))
            elif kind == "unobserve":
                observer_id = rng.choice(tuple(active_observers))
                field_arg = active_observers.pop(observer_id)
                operations.append(Operation("unobserve", field_arg, observer_id=observer_id))
            elif kind in ("hold_enter", "lock_enter"):
                context = "hold" if kind == "hold_enter" else "lock"
                operations.append(Operation(kind))
            else:
                operations.append(Operation("get", choose_field()))
            break

    if context is not None:
        index = len(operations)
        operations.append(Operation("hold_exit" if context == "hold" else "lock_exit"))
        if context == "hold":
            unordered.add(index)

    return Generated(family, operations, unordered)


def _all_field_specs(spec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(cls: Any) -> None:
        for base in cls.bases:
            if base.name not in out:
                visit(base)
        for fs in cls.fields:
            out[fs.name] = fs.trait

    visit(spec)
    return out
