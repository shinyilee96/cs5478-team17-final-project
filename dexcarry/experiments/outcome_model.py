"""The planner's outcome model, built from training objects only: pickup success, what each failure leaves
behind, and trip times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexcarry.experiments.pickups import REACH_MIN, reached, robust_rows
from dexcarry.experiments.predictors import TRAIN_PAIRS
from dexcarry.models.plan_predictor import robust_rank_key

MODEL_DISTANCE = 1.0 # metres between the tables
CONSEQUENCE = {
    "FIRST_DISTURBED": ("fallen", "fallen"),
    "FIRST_NO_CONTACT": ("fallen", "standing"),
    "FIRST_NOT_LIFTED": ("fallen", "standing"),
    "FIRST_OTHER_FINGER": ("held", "standing"),
    "SECOND_UNREACHABLE": ("held", "standing"),
    "SECOND_PATH_COLLISION": ("held", "standing"),
    "FIRST_LOST_MOVING": ("fallen", "standing"),
    "SECOND_PUSHED": ("held", "fallen"),
    "SECOND_NOT_LIFTED": ("held", "fallen"),
    "FIRST_LOST_ON_SECOND_GRASP": ("fallen", "fallen"),
    "BOTH_LOST": ("fallen", "fallen"),
}
# Where in the pickup each code ends it, for its duration: after the first lift ("first"), on the way to the
# second grasp ("between", the midpoint), or at the end of the pickup ("pickup").
FAILURE_STAGE = {
    "FIRST_DISTURBED": "first", "FIRST_NO_CONTACT": "first", "FIRST_NOT_LIFTED": "first", "FIRST_OTHER_FINGER": "first",
    "SECOND_UNREACHABLE": "first", "SECOND_PATH_COLLISION": "first",
    "FIRST_LOST_MOVING": "between", "SECOND_PUSHED": "between",
    "SECOND_NOT_LIFTED": "pickup", "FIRST_LOST_ON_SECOND_GRASP": "pickup", "BOTH_LOST": "pickup",
}


@dataclass(frozen=True)
class Outcome:
    """One outcome of an attempt: its probability, objects delivered, the attempted objects that no longer
    stand, the attempt's duration including the drive back, and when its deliveries are complete."""
    p: float
    delivered: int
    gone: frozenset
    seconds: float
    deliver_at: float
    failed_at_table: bool = False


@dataclass
class Model:
    """Everything the outcome model knows, built from existing data (``build_model``) or by hand."""
    pair_codes: dict # "first+second" -> list of label codes (None = pickup succeeded)
    post_pickup: list # [(p, first delivered, second delivered)] after a successful pickup
    held_first: float # chance that a first object still held after a failed pickup is delivered
    single_rate: dict # object -> chance its single trip delivers it
    single_seconds: dict # object -> single trip from the pick table to its release, at 1 m
    t_first: float # pickup time to the first lift
    t_pickup: float # pickup time to the end of the hold
    place_both: float # set-down of both, from arrival
    place_first: float # set-down of the wrapped object alone, from arrival
    table: dict = field(default_factory=lambda: dict(CONSEQUENCE))
    plans: dict = field(default_factory=dict) # pair -> plan index (reported)

    def single(self, name):
        if name in self.single_rate:
            return self.single_rate[name], self.single_seconds[name]
        return self.single_rate["*"], self.single_seconds["*"]


def build_model(library: Path, trips: Path, singles: Path, pickup_times="trips") -> Model:

    pair_codes, plans, first_lift, pickup = {}, {}, [], []
    for name in TRAIN_PAIRS:
        path = library / f"robot-labels-{name}.json"
        if not path.exists():
            continue
        labels = {r["plan"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        accepted = reached(library, name)
        robust = {r["plan"]: r for r in (robust_rows(library, name) or [])}
        usable = [k for k in labels if accepted.get(k, 0) >= REACH_MIN]
        if not usable:
            continue
        # The replay rank's top plan among the labelled ones (predictors.py.dataset's order), ties by index.
        k = min(usable, key=lambda k: (robust_rank_key(robust.get(k), accepted[k]), k))
        pair_codes[name], plans[name] = list(labels[k]["first_failure"]), k
        for row in labels.values():
            for code, diag in zip(row["first_failure"], row["diagnostics"], strict=True):
                timeline = diag.get("timeline", {})
                if "first_lifted" in timeline:
                    first_lift.append(timeline["first_lifted"])
                if code is None:
                    pickup.append(timeline["both_lifted"])
    post, place_both, place_first, trip_first, trip_pickup = [], [], [], [], []
    for path in sorted(trips.glob("trip-fix-*-d1.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            for r in row["results"]:
                tl = r["diagnostics"]["timeline"]
                if pickup_times == "trips" and "first_lifted" in tl:
                    trip_first.append(tl["first_lifted"])
                if not r["pickup"]:
                    continue
                if pickup_times == "trips":
                    trip_pickup.append(tl["both_lifted"])
                post.append(tuple(r["delivered"]))
                if "done" in tl:
                    place_both.append(tl["done"] - tl["arrived"])
                    place_first.append(tl["done"] - tl["second_released"])
    counts = {key: post.count(key) for key in sorted(set(post))}
    post_pickup = [(n / len(post), a, b) for (a, b), n in counts.items()]
    rate, seconds, pooled = {}, {}, []
    for path in sorted(singles.glob("single-*-d1.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            for r in row["results"]:
                rate.setdefault(row["object"], []).append(r["delivered"])
                seconds.setdefault(row["object"], []).append(r["seconds"])
                pooled.append((r["delivered"], r["seconds"]))
    single_rate = {k: float(np.mean(v)) for k, v in rate.items()}
    single_seconds = {k: float(np.mean(v)) for k, v in seconds.items()}
    single_rate["*"], single_seconds["*"] = (float(np.mean([p[i] for p in pooled])) for i in (0, 1))
    if pickup_times == "trips":
        first_lift, pickup = trip_first, trip_pickup
    return Model(pair_codes=pair_codes, post_pickup=post_pickup, held_first=float(np.mean([a for a, _ in post])),
                 single_rate=single_rate, single_seconds=single_seconds, t_first=float(np.mean(first_lift)),
                 t_pickup=float(np.mean(pickup)), place_both=float(np.mean(place_both)),
                 place_first=float(np.mean(place_first)), plans=plans)


@dataclass
class Errand:
    objects: tuple
    pairs: list # (i, j): i wrapped, j by three fingers
    distance: float = MODEL_DISTANCE
    knock: float = 0.
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def options(self, standing):
        out = [("pair", i, j) for i, j in self.pairs if i in standing and j in standing]
        return out + [("single", i) for i in sorted(standing)]


