"""Mutation analysis for the reference-model differential harness.

Development tool, not part of the default pytest suite.  It loads isolated
copies of ``model.py`` with one targeted mutation at a time and replays a
fixed sequence against the mutated reference model and real traitlets, using
the production differential driver for observation/normalization.  A mutation
is "killed" when the fixed sequence diverges (or the mutated model errors).

Run explicitly (this is the required mutation-analysis entry point)::

    .venv/bin/python tests/refmodel/mutation_analysis.py

Branch families covered: descriptor ``set``/``delete``, default cache,
validation rollback, and notification batching.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_SOURCE = HERE / "model.py"

MUTATIONS: dict[str, tuple[str, str]] = {
    "descriptor_set_leaks_unvalidated_value": (
        "        new_value = self._validate_trait_value(trait, value)\n"
        "        had_cached = name in self._values\n",
        "        new_value = value  # MUTATION: skip trait + cross validation\n"
        "        had_cached = name in self._values\n",
    ),
    "descriptor_delete_removes_instead_of_sentinel": (
        "        self._values[name] = DELETED_SENTINEL\n"
        "        self._deleted.add(name)",
        "        self._values.pop(name, None)  # MUTATION: remove, no sentinel\n"
        "        self._deleted.add(name)",
    ),
    "default_cache_reruns_generator_every_read": (
        "        if name not in self._values:\n"
        "            value = self._compute_default(name)\n"
        "            self._values[name] = value\n"
        "            # Default events bypass",
        "        value = self._compute_default(name)  # MUTATION: no cache\n"
        "        self._values[name] = value\n"
        "        if False:\n"
        "            pass\n"
        "            # Default events bypass",
    ),
    "validation_rollback_keeps_mutated_values": (
        "            rollback_error: BaseException | None = None\n"
        "            for name, changes in cache.items():\n",
        "            rollback_error: BaseException | None = None\n"
        "            for name, changes in {}.items():  # MUTATION: no rollback\n",
    ),
    "notification_batching_no_compression": (
        "            if past is None:\n"
        "                self._hold_cache[event.name] = [event]\n"
        "            elif past[-1].type == \"change\" and event.type == \"change\":\n"
        "                past[-1].new = event.new\n"
        "            else:\n"
        "                past.append(event)\n",
        "            self._hold_cache.setdefault(event.name, []).append(event)\n",
    ),
}

KILL_SEQUENCES: dict[str, tuple[str, list]] = {
    "descriptor_set_leaks_unvalidated_value": (
        "Basic",
        [("observe", "h1", "i", "change"), ("set", "i", "nope", "attr"), ("get", "i")],
    ),
    "descriptor_delete_removes_instead_of_sentinel": (
        "Basic",
        [("observe", "h1", "i", "change"), ("delete", "i"), ("set", "i", 4, "attr")],
    ),
    "default_cache_reruns_generator_every_read": (
        "Dynamic",
        [("get", "i"), ("get", "i")],
    ),
    "validation_rollback_keeps_mutated_values": (
        "Dynamic",
        [
            ("observe", "h1", "ALL", "change"),
            ("begin", "hold"),
            ("set", "i", 5, "attr"),
            ("set", "z", 99, "attr"),
            ("end", "hold"),
        ],
    ),
    "notification_batching_no_compression": (
        "Basic",
        [
            ("observe", "h1", "i", "change"),
            ("begin", "hold"),
            ("set", "i", 1, "attr"),
            ("set", "i", 2, "attr"),
            ("end", "hold"),
        ],
    ),
}


_MUTATION_SERIAL = 0


def _load_mutated_model(mutated_source: str):
    global _MUTATION_SERIAL
    _MUTATION_SERIAL += 1
    unique = f"refmodel_mut_{_MUTATION_SERIAL}"
    module = types.ModuleType(unique)
    module.__file__ = str(MODEL_SOURCE) + " (mutated)"
    sys.modules[unique] = module
    # ``from __future__`` must be the first statement of a file; it is
    # unnecessary (and rejected positionally) when exec-ing dynamically.
    mutated_source = "\n".join(
        line for line in mutated_source.splitlines()
        if not line.strip().startswith("from __future__")
    )
    exec(compile(mutated_source, module.__file__, "exec"), module.__dict__)
    return module


def _diverges_with(shape_name: str, ops: list, model_module) -> bool:
    from . import driver, ref_builder, shapes

    original = {
        "RefClass": ref_builder.RefClass,
        "RefTrait": ref_builder.RefTrait,
        "TraitRefError": ref_builder.TraitRefError,
    }
    # driver references the model symbols at call time, so patch those too.
    driver_original = {
        "RefInstance": driver.RefInstance,
        "RefAttributeError": driver.RefAttributeError,
        "TraitRefError": driver.TraitRefError,
        "DELETED_SENTINEL": driver.DELETED_SENTINEL,
        "MISSING": driver.MISSING,
    }
    ref_builder.RefClass = model_module.RefClass
    ref_builder.RefTrait = model_module.RefTrait
    ref_builder.TraitRefError = model_module.TraitRefError
    driver.RefAttributeError = model_module.RefAttributeError
    driver.TraitRefError = model_module.TraitRefError
    driver.DELETED_SENTINEL = model_module.DELETED_SENTINEL
    driver.MISSING = model_module.MISSING
    driver.RefInstance = model_module.RefInstance
    try:
        shape = shapes.SHAPES_BY_NAME[shape_name]
        try:
            driver.run_sequence(shape, ops)
        except driver.Mismatch:
            return True
        except Exception:
            # A mutation that crashes the model is also detected as a kill.
            return True
        return False
    finally:
        for key, value in original.items():
            setattr(ref_builder, key, value)
        for key, value in driver_original.items():
            setattr(driver, key, value)


def main() -> int:
    source = MODEL_SOURCE.read_text()
    survivors = []
    for mutation_name, (needle, replacement) in MUTATIONS.items():
        if needle not in source:
            survivors.append((mutation_name, "pattern not found"))
            continue
        mutated_source = source.replace(needle, replacement, 1)
        try:
            model_module = _load_mutated_model(mutated_source)
        except SyntaxError as exc:
            print(f"{mutation_name}: MUTATION SOURCE INVALID ({exc})")
            survivors.append((mutation_name, "invalid mutation source"))
            continue
        shape_name, ops = KILL_SEQUENCES[mutation_name]
        killed = _diverges_with(shape_name, ops, model_module)
        status = "killed" if killed else "SURVIVED"
        print(f"{mutation_name}: {status}")
        if not killed:
            survivors.append((mutation_name, "survived its fixed sequence"))

    if survivors:
        print("\nMutation analysis FAILED for:")
        for name, reason in survivors:
            print(f"  - {name}: {reason}")
        return 1
    print(f"\nAll {len(MUTATIONS)} mutations were killed by fixed sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
