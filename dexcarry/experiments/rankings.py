"""Rebuilds the R, P and E rankings from the plan library and checks them against the recorded ones."""

from __future__ import annotations

import numpy as np

from dexcarry.experiments.plan_choice import candidate_rows, e_order, ranking
from dexcarry.experiments.predictors import frozen_estimator, frozen_rules
from dexcarry.paths import INPUTS, WEIGHTS


def rebuilt_top3(rankings: list[dict]) -> bool:
    """Rebuild each pair's candidates and the top 3 of R, P and E; print any pair that differs from ``rankings``."""
    library = INPUTS / "plan_choice_library"
    p_model, _, _, _ = frozen_rules(WEIGHTS / "plan_predictor")
    e_model, _ = frozen_estimator()
    same = True
    for row in rankings:
        rows = candidate_rows(library, row["pair"])
        plans = [r["plan"] for r in rows]
        if " ".join(map(str, plans)) != row["candidates"]:
            print(f"  {row['pair']}: candidates differ")
            same = False
            continue
        X = np.stack([r["x"] for r in rows])
        r_order = [r["plan"] for r in sorted(rows, key=lambda r: (r["robust_key"], r["plan"]))]
        orders = {"replay_ranking_R": r_order, "learned_predictor_P": ranking(plans, p_model.score(X))}
        if row["learned_predictor_E"]:
            orders["learned_predictor_E"] = e_order(e_model.score(X), r_order, plans)
        for column, order in orders.items():
            if " ".join(map(str, order[:3])) != row[column]:
                print(f"  {row['pair']}: {column} top 3 {order[:3]} against frozen {row[column]}")
                same = False
    return same
