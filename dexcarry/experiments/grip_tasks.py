"""Start states for the grip experiments, built from the plan library, and the three simulator settings."""

from __future__ import annotations

import json
from pathlib import Path

from dexcarry.experiments.pickups import OBJECTS, REACH_MIN, reached, robust_choice
from dexcarry.rl.grasp_env import Task
from dexcarry.sim.two_object import TwoObjectPlan

# The training pairs: the learned models and the RL grip learn from these only.
TRAIN_PAIRS = (("cube-s", "tall"), ("stick", "tall"), ("slim", "cube-s"), ("tall", "slab"), ("dgn-bottle", "tall"),
               ("dgn-carton", "tall"), ("ycb-sugar-box", "tall"), ("cube-s", "dgn-bottle"), ("bar-2cm", "cube-s"),
               ("bar-4cm", "tall"), ("stick", "slab"), ("dgn-carton", "cube-s"))


def settings(finestep_scaled: bool) -> dict:
    """The three simulator settings (standard, friction 0.8, halved time step) as physics draws; with ``finestep_scaled``
    the halved step scales the motor gains with the time step."""
    return {"standard": {"contact_friction": .4, "dt": 1 / 240},
            "mu0.8": {"contact_friction": .8, "dt": 1 / 240},
            "finestep": {"contact_friction": .4, "dt": 1 / 480, "scaled": finestep_scaled}}


def load_tasks(folder: Path, pairs=TRAIN_PAIRS, per_pair=3, robust=False):
    """Plans of the given pairs that the planner's dry run accepts in at least REACH_MIN scenes, in
    search order (or the robust filter's choice with ``robust``), at most ``per_pair`` each."""

    tasks = []
    for first, second in pairs:
        name = f"{first}+{second}"
        path = folder / f"plans-{name}.json"
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))["plans"]
        if robust:
            chosen = robust_choice(folder, name, per_pair)
        else:
            accepted = reached(folder, name) or {}
            chosen = sorted(k for k, v in accepted.items() if v >= REACH_MIN)[:per_pair]
        tasks += [Task(OBJECTS[first], OBJECTS[second], TwoObjectPlan.from_dict(stored[k]), f"{name}#{k}") for k in chosen]
    return tasks

