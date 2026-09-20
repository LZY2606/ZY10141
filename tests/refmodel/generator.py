"""Bounded, stateful operation-sequence generator for differential tests.

Guarantees that keep generation safe and deterministic:

* operations come from a closed, JSON-serializable set,
* contexts are generated as properly matched ``begin``/``end`` pairs via a
  stack (nested holds/locks are possible, depth capped),
* trait names and values are restricted to the chosen shape,
* sequences are bounded by ``max_length`` so shrinking and replay are cheap,
* no classes are created at generation time -- the finite shape library in
  :mod:`shapes` is built once and reused.
"""

from __future__ import annotations

import random

from .driver import VALUES_BY_KIND
from .shapes import Shape

HANDLER_IDS = ("h1", "h2")
OBSERVE_TYPES = ("change", "All")
SET_MODES = ("attr", "set_trait")
CONTEXT_KINDS = ("hold", "lock")
MAX_CONTEXT_DEPTH = 2


def _trait_kind(shape: Shape, name: str) -> str:
    for layer in shape.layers:
        if name in layer:
            return layer[name].kind
    raise KeyError(name)


def generate(rng: random.Random, shape: Shape, max_length: int = 24) -> list[tuple]:
    """Generate one well-formed operation sequence for ``shape``."""
    names = tuple(shape.traits)
    ops: list[tuple] = []
    stack: list[str] = []
    length = rng.randint(2, max_length)
    for _ in range(length):
        choice = rng.random()
        name = rng.choice(names)
        kind = _trait_kind(shape, name)
        target = rng.choice((name, "ALL"))
        handler = rng.choice(HANDLER_IDS)
        obs_type = rng.choice(OBSERVE_TYPES)

        if choice < 0.22:
            ops.append(("get", name))
        elif choice < 0.52:
            value = rng.choice(VALUES_BY_KIND[kind])
            ops.append(("set", name, value, rng.choice(SET_MODES)))
        elif choice < 0.62:
            ops.append(("delete", name))
        elif choice < 0.74:
            if len(stack) < MAX_CONTEXT_DEPTH:
                context = rng.choice(CONTEXT_KINDS)
                ops.append(("begin", context))
                stack.append(context)
            elif stack:
                ops.append(("end", stack[-1]))
                stack.pop()
            else:
                ops.append(("get", name))
        elif choice < 0.80:
            ops.append(("observe", handler, target, obs_type))
        elif choice < 0.86:
            ops.append(("unobserve", handler, target, obs_type))
        else:
            ops.append(("get", name))

    while stack:
        ops.append(("end", stack.pop()))
    return ops


def generate_many(
    seed: int,
    shapes: tuple[Shape, ...],
    count: int,
    max_length: int = 24,
) -> list[tuple[Shape, tuple]]:
    """Generate ``count`` (shape, ops) pairs deterministically from ``seed``.

    ``count`` is intentionally capped by callers (see the test module) so the
    default suite runs a fixed, small sample.
    """
    rng = random.Random(seed)
    sequences: list[tuple[Shape, tuple]] = []
    for index in range(count):
        shape = shapes[index % len(shapes)]
        sequences.append((shape, tuple(generate(rng, shape, max_length))))
    return sequences
