"""The long table's break-even distance: trip time is fitted as T = h + b L for each trip kind, and L* is where
two-object trips start delivering more per second (interval from 2,000 layout resamples).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESAMPLES = 2000
SEED = 0


def fit(trips: pd.DataFrame) -> dict:
    out = {}
    for kind in ("single", "pair"):
        k = trips[trips.kind == kind]
        b, h = np.polyfit(k.distance.to_numpy(float), k.seconds.to_numpy(float), 1)
        out[kind] = {"h": float(h), "b": float(b), "D": float(k.delivered.mean()), "n": int(len(k))}
    s, p = out["single"], out["pair"]
    out["L_star"] = (s["D"] * p["h"] - p["D"] * s["h"]) / (p["D"] * s["b"] - s["D"] * p["b"])
    return out


def interval(trips: pd.DataFrame) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    layouts = sorted(trips.layout.unique())
    groups = {g: d for g, d in trips.groupby("layout")}
    values = []
    for _ in range(RESAMPLES):
        pick = rng.choice(layouts, len(layouts), replace=True)
        values.append(fit(pd.concat([groups[g] for g in pick]))["L_star"])
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))
