"""Recovery after failed two-object trips: which trips failed and how many objects recovery delivered."""

from __future__ import annotations

import json
from pathlib import Path

from dexcarry.paths import ROOT

MODEL = ROOT / "configs" / "recovery_model.json"
# Errands read only ``recovery_model`` from here: the model in configs/recovery_model.json.
STEP_ACTIONS = ("retry", "pair", "deliver", "single")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def failed_pairs(r: dict) -> list[dict]:
    return [t for t in r["trips"] if t["kind"] == "pair" and "refused" not in t and t.get("failure")]


def role_slot(t: dict, name: str) -> str:
    """The trip's ``delivered`` is keyed by slot (a string once in JSON); ``name`` is the role."""
    return str(t["i"] if name == "first" else t["j"])


def saved(t: dict) -> int:
    """Objects of a failed pair trip that end delivered and that a recovery step handled."""
    handled = {name for step in (t.get("recovery") or []) if step["action"] in (*STEP_ACTIONS, "deliver") for name in step["objects"]}
    return sum(bool(t["delivered"][role_slot(t, name)]) for name in handled)


def recovery_model() -> dict:
    """The recovery model the planner uses (configs/recovery_model.json)."""
    return load(MODEL)
