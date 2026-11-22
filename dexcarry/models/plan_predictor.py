"""Predicts how often a two-object plan holds both objects on the robot, from features known before execution
(object sizes, plan geometry, the arm planner's dry run and the floating-hand replays).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROBUST_KEYS = [(.8, 240), (.4, 240), (1.2, 240), (.8, 480), (.4, 480), (1.2, 480)]


def _flex(cfg, joints, finger):
    order, groups, flexion = cfg["joint_order"], cfg["finger_groups"], set(cfg["flexion_joints"])
    return float(sum(joints[order.index(j)] for j in groups[finger] if j in flexion))


def plan_features(plan, first, second, refusals, robust_row, cfg) -> np.ndarray:
    """One feature row (``FEATURE_NAMES``); NaN where a source is missing."""
    f_pose, s_pose = np.asarray(plan.first_pose, float), np.asarray(plan.second_pose, float)
    row = [*(np.asarray(first.half_extents) * 200), *(np.asarray(second.half_extents) * 200), first.mass, second.mass,
           float(bool(first.visual_mesh)), float(bool(second.visual_mesh))]
    heading = np.arctan2(f_pose[1, 0], f_pose[0, 0])
    row += [*(f_pose[:3, 3] * 100), f_pose[2, 2], np.cos(heading), np.sin(heading)]
    if plan.first_held is not None:
        held = np.asarray(plan.first_held, float)
        row += [*(held[:3, 3] * 100), float(np.degrees(np.arccos(np.clip(held[2, 2], -1, 1))))]
    else:
        row += [np.nan] * 4
    row += [*(s_pose[:3, 3] * 100), s_pose[2, 2], s_pose[2, 0], (s_pose[2, 3] - second.half_extents[2]) * 100]
    joints = np.asarray(plan.second_closed, float)
    row += [_flex(cfg, joints, f) for f in ("thumb", "index", "middle")] + [float(len(plan.second_fingers))]
    row += [float(sum(r is None for r in refusals)), float(len(refusals))]
    held = {(s["contact_friction"], round(1 / s["dt"])): sum(s["both_held"]) for s in (robust_row or {}).get("settings", [])}
    values = [held.get(key, np.nan) for key in ROBUST_KEYS]
    row += values + [np.nanmin(values) if not np.all(np.isnan(values)) else np.nan,
                     np.nansum(values) if not np.all(np.isnan(values)) else np.nan]
    return np.asarray(row, float)


def robust_rank_key(robust_row, accepted):
    """The robust filter's order (``pickups.py.robust_choice``): worst setting, then scenes the dry
    run accepted, then the total held. Applied to every accepted plan, since a pair may have no plan the
    filter keeps; a plan without replays sorts last."""
    if not robust_row or not robust_row.get("settings"):
        return (np.inf, -accepted, np.inf)
    held = [sum(s["both_held"]) for s in robust_row["settings"]]
    return (-min(held), -accepted, -sum(held))


@dataclass
class PlanPredictor:
    """``kind`` is "logistic" (held in half or more, probability) or "boosting" (the held rate)."""

    kind: str = "boosting"
    seed: int = 0

    def fit(self, X, rate):
        if self.kind == "logistic":
            # liblinear, not the default L-BFGS: SciPy's L-BFGS-B crashes in cs5478-team-17 on 17 Oct (Windows fatal
            # exception 0xc06d007f, a DLL entry point not found; reproduced with a bare scipy.optimize.minimize call).
            self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegression(C=.5, solver="liblinear", max_iter=2000)).fit(X, np.asarray(rate) >= .5)
        elif self.kind == "boosting":
            self.model = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300,
                                                       min_samples_leaf=10, random_state=self.seed).fit(X, rate)
        else:
            raise ValueError("kind is 'logistic' or 'boosting'")
        return self

    def score(self, X) -> np.ndarray:
        if self.kind == "logistic":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

