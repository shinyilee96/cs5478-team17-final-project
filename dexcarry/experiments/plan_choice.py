"""Plan choice: each pair's candidate plans and how R, P and E rank them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexcarry.config import load_hand_config
from dexcarry.experiments.pickups import OBJECTS, REACH_MIN, reached, robust_rows
from dexcarry.models.plan_predictor import plan_features, robust_rank_key
from dexcarry.sim.two_object import TwoObjectPlan

# Candidate set: the first MAX_CANDIDATES plans, in search order, that the arm planner's dry run accepts in at least
# REACH_MIN of REACH_SCENES scenes. Each gets 20 robot pickups at the standard setting.
MAX_CANDIDATES = 25


def candidates(folder: Path, name: str) -> list[int] | None:
    """The pair's candidate set, as ``pickups.py robot --accepted --max-plans 25`` runs it; None
    before the dry run."""
    accepted = reached(folder, name)
    if accepted is None:
        return None
    return sorted(k for k, scenes in accepted.items() if scenes >= REACH_MIN)[:MAX_CANDIDATES]


def candidate_rows(folder: Path, name: str) -> list[dict]:
    """Features and the robust rank key of each candidate, built as ``predictors.py.dataset`` builds them."""

    cfg = load_hand_config("leap")
    first, second = (OBJECTS[n] for n in name.split("+"))
    stored = json.loads((folder / f"plans-{name}.json").read_text(encoding="utf-8"))["plans"]
    reach = {r["plan"]: r["refusals"] for r in json.loads((folder / f"reach-{name}.json").read_text(encoding="utf-8"))}
    robust = {r["plan"]: r for r in (robust_rows(folder, name) or [])}
    rows = []
    for k in candidates(folder, name) or []:
        accepted = sum(r is None for r in reach[k])
        assert accepted >= REACH_MIN
        plan = TwoObjectPlan.from_dict(stored[k])
        rows.append({"plan": k, "x": plan_features(plan, first, second, reach[k], robust.get(k), cfg),
                     "robust_key": robust_rank_key(robust.get(k), accepted), "replayed": k in robust})
    return rows


def ranking(plans: list[int], scores) -> list[int]:
    """Plans by score, highest first, ties by plan index; a missing (NaN) score ranks last."""
    scores = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    return [plans[i] for i in sorted(range(len(plans)), key=lambda i: (-scores[i], plans[i]))]


def e_order(scores, r_order: list[int], plans: list[int]) -> list[int]:
    """E's order: score highest first, ties by R's order."""
    rank_r = {k: i for i, k in enumerate(r_order)}
    score = dict(zip(plans, np.asarray(scores, float), strict=True))
    return sorted(plans, key=lambda k: (-score[k], rank_r[k]))
