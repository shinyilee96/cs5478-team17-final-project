"""Success odds measured in practice errands (privileged), used by the planner on the partner-scarcity layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexcarry.experiments import errands
from dexcarry.paths import ROOT
from dexcarry.pipeline.two_object import FAILURES, PLANNER_FAILURES
from dexcarry.plan.errand_planner import Knocks, Option, pair_codes

ODDS = ROOT / "configs" / "scenario_odds.json"
STOOD_UP_SECONDS = 20. # severity "stood up again": seconds per toppled object (about one pickup)


def table_codes() -> frozenset:
    """A pair's pickup failures that happen at the table with the arm moving: ``FAILURES`` minus the planner's own refusals."""
    return frozenset(FAILURES) - frozenset(PLANNER_FAILURES)


def load_odds(path: Path = ODDS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_odds(odds: dict, a: str, b: str) -> float:
    """An ordered name pair's measured pickup success; a pair never tried takes its wrapped object's, an object never wrapped the pooled."""
    if f"{a}+{b}" in odds["pick"]:
        return odds["pick"][f"{a}+{b}"]["p"]
    return odds["wrap"][a]["p"] if a in odds["wrap"] else odds["pooled"]["pick"]


def spread(p: float, mean: float, s: float) -> float:
    """Dial (c): an option's pickup success moved away from (``s`` > 1) or onto (``s`` = 0) the pooled mean."""
    return float(np.clip(mean + s * (p - mean), .02, .98))


def true_options(names: list[str], pair_rows: list[dict], slots, target: float, odds: dict, s: float = 1.) -> list:
    """The options with measured odds: each pair its own pickup success (spread by ``s``), the pooled failure codes, its wrapped
    object's after-pickup table; each single its own rate. ``pair_rows`` are errand_plans.json's rows of the layout's found pairs."""

    out = []
    for row in pair_rows:
        i, j = row["i"], row["j"]
        a, b = names[i], names[j]
        p = spread(pick_odds(odds, a, b), odds["pooled"]["pick"], s)
        post = tuple(tuple(x) for x in odds["post"].get(a, odds["pooled"]["post"]))
        out.append(Option("pair", i, j, pair_codes(p, odds["failures"]), target - errands.stance_of(slots, i), post, errands.traverse(slots, i, j)))
    for i, name in enumerate(names):
        rate = odds["single"][name]["p"] if name in odds["single"] else odds["pooled"]["single"]
        out.append(Option("single", i, drive=target - errands.stance_of(slots, i), rate=rate))
    return out


def knock_model(positions, p: float, r: float, lost: bool, odds: dict):
    """The planner's ``Knocks`` for objects at ``positions`` (slot -> xy), knock chance ``p`` within ``r`` metres, severity ``lost`` (or
    stood up again for ``STOOD_UP_SECONDS`` each)."""
    xy = np.asarray(positions, float)
    near = tuple(frozenset(int(j) for j in range(len(xy)) if j != i and np.hypot(*(xy[j] - xy[i])) <= r + 1e-9) for i in range(len(xy)))
    return Knocks(p, near, lost, 0. if lost else STOOD_UP_SECONDS, table_codes(), odds["single_table"])

