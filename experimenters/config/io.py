from __future__ import annotations
"""Utility helpers to load/override typed configs.

* Replaces the old YAML+dacite loader.
* Keeps **zero dependencies** - relies only on stdlib.
"""
from importlib import import_module, util as im_util
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "load_py_cfg",
    "apply_dotlist_overrides",
]

############################################################
# 1. Loading a Python module that exposes `cfg`
############################################################

def _import_from_file(py_file: Path) -> ModuleType:
    spec = im_util.spec_from_file_location(py_file.stem, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {py_file}")
    mod = im_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_py_cfg(path_or_dotted: str):
    """Execute the python file / module and return its top‑level ``cfg`` object."""
    if path_or_dotted.endswith(".py") or Path(path_or_dotted).exists():
        mod = _import_from_file(Path(path_or_dotted))
    else:  # dotted module path e.g. configs.moe_256
        mod = import_module(path_or_dotted)
    if not hasattr(mod, "cfg"):
        raise AttributeError(
            f"Expected a variable named 'cfg' in {path_or_dotted}, found none."
        )
    return getattr(mod, "cfg")

############################################################
# 2. Dot‑list overrides ("a.b.c=42")
############################################################

def _cast_value(old: Any, new_str: str):
    if isinstance(old, bool):
        return new_str.lower() in {"1", "true", "yes"}
    if isinstance(old, int):
        return int(new_str)
    if isinstance(old, float):
        return float(new_str)
    return new_str  # str or unknown - leave as is


def apply_dotlist_overrides(cfg, kv_list: list[str]):
    """Mutate *cfg* in‑place according to overrides in KEY=VAL form."""
    for expr in kv_list:
        if "=" not in expr:
            raise ValueError(f"Override must be key=value, got '{expr}'")
        key, val_str = expr.split("=", 1)
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not hasattr(cur, p):
                raise AttributeError(f"Config has no field '{p}' in '{key}'")
            cur = getattr(cur, p)
        last = parts[-1]
        if not hasattr(cur, last):
            raise AttributeError(f"Config has no field '{last}' in '{key}'")
        old_val = getattr(cur, last)
        setattr(cur, last, _cast_value(old_val, val_str))
    return cfg
