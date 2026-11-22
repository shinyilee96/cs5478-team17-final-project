"""Recomputes every table and number in the report body from the per-trial files in ``results/``."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dexcarry.eval.stats import cluster_bootstrap, cluster_mean, paired_difference, wilson
from dexcarry.experiments.break_even import fit, interval
from dexcarry.paths import INPUTS, RESULTS

DISTANCES = (1., 2., 4., 8., 12., 16.)
SETTINGS = ("standard", "mu0.8", "finestep")


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ratio(means):
    return means[..., 0].mean(axis=-1) / means[..., 1].mean(axis=-1) - 1


def load_errands(folder: Path = RESULTS) -> dict:
    """{(table, layouts, method, distance): {(layout, seed, setting): row}} from errands.csv."""
    out: dict = defaultdict(dict)
    for r in read_csv(Path(folder) / "errands.csv"):
        key = (r["table"], r["layouts"], r["method"], float(r["distance"]))
        out[key][(int(r["layout"]), int(r["seed"]), r["setting"])] = {
            "objects": int(r["objects"]), "delivered": int(r["delivered"]), "upright": int(r["upright"]), "seconds": float(r["seconds"]),
            "first_trip_tempting_pair": r["first_trip_tempting_pair"], "scarce_object_delivered": r["scarce_object_delivered"]}
    return out


def per_minute(r: dict) -> float:
    return 60 * r["delivered"] / r["seconds"]


def share(r: dict) -> float:
    return r["delivered"] / r["objects"]


def errand_effect(a: dict, b: dict, settings=SETTINGS) -> dict:
    """``a`` against ``b`` (errands keyed by (layout, seed, setting)), paired by layout and by unit (seed and setting):
    the delivery-rate change (ratio of mean rates) and the delivery-share difference."""
    a = {k: v for k, v in a.items() if k[2] in settings}
    layouts = sorted({k[0] for k in a} & {k[0] for k in b})
    units = {layout: sorted(k[1:] for k in a if k[0] == layout and k in b) for layout in layouts}
    layouts = [layout for layout in layouts if units[layout]]

    def arm(rows, measure):
        return [[measure(rows[(layout, *u)]) for u in units[layout]] for layout in layouts]
    return {"rate": cluster_bootstrap([arm(a, per_minute), arm(b, per_minute)], ratio),
            "share": paired_difference(arm(a, share), arm(b, share)), "errands": sum(map(len, units.values()))}


def table1(folder: Path = RESULTS) -> dict:
    """Table 1 and Figure 2a: DexCarry against the single-object baseline on both tables."""
    e = load_errands(folder)
    out = {}
    for table in ("round", "long"):
        for d in DISTANCES:
            out[(table, f"{d:g} m")] = errand_effect(e[(table, "random", "dexcarry", d)], e[(table, "random", "single_object", d)], ("standard",))
        out[(table, "4 m, pooled")] = errand_effect(e[(table, "random", "dexcarry", 4.)], e[(table, "random", "single_object", 4.)])
        for method in ("dexcarry", "single_object"):
            rows = list(e[(table, "random", method, 4.)].values())
            out[(table, f"share {method} 4 m")] = float(np.mean([share(r) for r in rows]))
    return out


def trips_at(folder: Path, table: str, layouts: str = "random", distance: float = 4.) -> list[dict]:
    return [t for t in read_csv(Path(folder) / "trips.csv")
            if t["table"] == table and t["layouts"] == layouts and float(t["distance"]) == distance]


def trip_statistics(folder: Path = RESULTS) -> dict:
    """Section 5.2: two-object against single-object trips at 4 m (all three settings), refused options left out."""
    out = {}
    for table in ("round", "long"):
        trips = [t for t in trips_at(folder, table) if not t["refused"]]
        for label, chosen in (("two-object trips (DexCarry)", [t for t in trips if t["method"] == "dexcarry" and t["kind"] == "pair"]),
                              ("single-object trips (baseline)", [t for t in trips if t["method"] == "single_object" and t["kind"] == "single"])):
            failed = sum(bool(t["failure"]) for t in chosen)
            out[(table, label)] = {"trips": len(chosen), "median_s": statistics.median(float(t["ended"]) - float(t["began"]) for t in chosen),
                                   "failure_share": failed / len(chosen)}
    return out


def break_even(folder: Path = RESULTS) -> dict:
    """The long table's break-even distance: trip seconds against drive distance by least squares (standard setting, 1-16 m,
    refused options left out), single-object trips of the baseline and two-object trips of DexCarry; interval over layouts."""

    rows = []
    for t in read_csv(Path(folder) / "trips.csv"):
        if t["table"] != "long" or t["layouts"] != "random" or t["setting"] != "standard" or t["refused"]:
            continue
        kind = {"single_object": "single", "dexcarry": "pair"}[t["method"]]
        if t["kind"] == kind:
            rows.append({"layout": int(t["layout"]), "distance": float(t["distance"]), "kind": kind,
                         "seconds": float(t["ended"]) - float(t["began"]), "delivered": len(t["delivered"].split())})
    trips = pd.DataFrame(rows)
    f = fit(trips)
    lo, hi = interval(trips)
    return {**f, "low": float(lo), "high": float(hi)}


def recovery(folder: Path = RESULTS) -> dict:
    """Table 2: objects saved by recovery per failed two-object trip, DexCarry at 4 m (all settings), clustered by layout."""
    out = {}
    errands = load_errands(folder)
    trips = read_csv(Path(folder) / "trips.csv")
    for table in ("round", "long"):
        saved, failed = defaultdict(int), defaultdict(int)
        for t in trips:
            if (t["table"], t["layouts"], t["method"], float(t["distance"])) == (table, "random", "dexcarry", 4.) and t["saved_by_recovery"] != "":
                key = (int(t["layout"]), int(t["seed"]), t["setting"])
                saved[key] += int(t["saved_by_recovery"])
                failed[key] += 1
        keys = sorted(errands[(table, "random", "dexcarry", 4.)])
        layouts = sorted({k[0] for k in keys})
        arms = [[[saved[k] for k in keys if k[0] == g] for g in layouts], [[failed[k] for k in keys if k[0] == g] for g in layouts]]
        d = cluster_bootstrap(arms, lambda m: m[..., 0].mean(axis=-1) / np.maximum(m[..., 1].mean(axis=-1), 1e-12))
        out[table] = {**d, "saved": sum(saved.values()), "failed": sum(failed.values())}
    return out


def planners(folder: Path = RESULTS) -> dict:
    """Figure 2b and Section 5.4: the failure-aware planner (DexCarry) against the greedy planner on the round table at 4 m."""
    e = load_errands(folder)
    out = {"random layouts": errand_effect(e[("round", "random", "dexcarry", 4.)], e[("round", "random", "greedy", 4.)]),
           "partner scarcity": errand_effect(e[("round", "partner_scarcity", "dexcarry", 4.)], e[("round", "partner_scarcity", "greedy", 4.)])}
    for method in ("greedy", "dexcarry"):
        rows = list(e[("round", "partner_scarcity", method, 4.)].values())
        out[f"scarcity {method}"] = {"errands": len(rows), "first trip took the tempting pair": sum(int(r["first_trip_tempting_pair"]) for r in rows),
                                     "scarce object delivered": sum(int(r["scarce_object_delivered"]) for r in rows)}
    return out


def plan_choice(folder: Path = RESULTS) -> dict:
    """Each rule's top 3 plans per pair (20 pickups each, in rank then seed order) against every candidate's pickups (the
    expected outcome of a random plan), on the pairs with at least 5 candidates, all labelled."""
    labels: dict = defaultdict(dict)
    rows = defaultdict(list)
    for r in read_csv(Path(folder) / "plan_choice_pickups.csv"):
        rows[(r["pair"], int(r["plan"]))].append((int(r["seed"]), int(r["both_held"])))
    for (pair, plan), v in rows.items():
        labels[pair][plan] = np.array([h for _, h in sorted(v)], float)
    top: dict = {"R": [], "P": [], "E": [], "random": []}
    for c in read_csv(Path(folder) / "plan_choice_rankings.csv"):
        candidates = [int(k) for k in c["candidates"].split()]
        lab = labels.get(c["pair"], {})
        if len(candidates) < 5 or not all(k in lab and len(lab[k]) == 20 for k in candidates):
            continue
        for rule, column in (("R", "replay_ranking_R"), ("P", "learned_predictor_P"), ("E", "learned_predictor_E")):
            top[rule].append(np.concatenate([lab[int(k)] for k in c[column].split()[:3]]))
        top["random"].append(np.concatenate([lab[k] for k in candidates]))
    out = {"pairs": len(top["R"])}
    for rule in ("R", "P", "E"):
        out[f"{rule} - random"] = paired_difference(top[rule], top["random"], paired_units=False)
        out[f"{rule} top 3"] = cluster_mean(top[rule])
    out["random"] = cluster_mean(top["random"])
    out["P - R"] = paired_difference(top["P"], top["R"])
    out["E - R"] = paired_difference(top["E"], top["R"])
    return out


def perception(folder: Path = RESULTS) -> dict:
    """Known objects: the same plan's 20 pickups planned on poses estimated from the depth image and on exact poses."""
    by: dict = defaultdict(list)
    order = []
    for r in read_csv(Path(folder) / "perception_pickups.csv"):
        key = (r["pair"], int(r["plan"]))
        if key not in order:
            order.append(key)
        by[(*key, r["poses"])].append(float(r["both_held"]))
    exact = [by[(*k, "exact")] for k in order]
    estimated = [by[(*k, "estimated")] for k in order]
    drop = paired_difference(exact, estimated)
    return {"plans": len(order), "exact": float(np.mean(exact)), "estimated": float(np.mean(estimated)),
            "estimated - exact": {"estimate": -drop["estimate"], "low": -drop["high"], "high": -drop["low"]}}


def unseen_objects(folder: Path = RESULTS, choices_path: Path = INPUTS / "unseen_objects" / "choices.json") -> dict:
    """Unseen objects: the primitive-shape plan library's top 3 for the fitted shapes against a random plan of the library,
    how often both objects are held, and the true shape's top 3 (privileged) against the library's."""
    got: dict = defaultdict(list)
    for r in read_csv(Path(folder) / "unseen_pickups.csv"):
        got[(r["pair"], r["plans"], int(r["plan"]))].append((int(r["seed"]), float(r["both_held"])))
    got = {k: np.array([h for _, h in sorted(v)]) for k, v in got.items()}
    top, rand, ref, held = [], [], [], []
    for name, c in json.loads(Path(choices_path).read_text(encoding="utf-8")).items():
        if not c["candidates"]:
            held.append(np.zeros(60))
            continue
        t = np.concatenate([got[(name, "primitive_library", k)] for k in c["atlas_order"][:3]])
        top.append(t)
        held.append(t)
        rand.append(np.concatenate([got[(name, "primitive_library", k)] for k in c["candidates"]]))
        if len(c["reference_top"]) == 3 == len(c["atlas_order"][:3]):
            ref.append((len(top) - 1, np.concatenate([got[(name, "true_shape", k)] for k in c["reference_top"]])))
    return {"pairs": len(top), "library - random": paired_difference(top, rand, paired_units=False), "both held": cluster_mean(held),
            "true shape - library": paired_difference([r for _, r in ref], [top[i] for i, _ in ref])}


def grip(folder: Path = RESULTS) -> dict:
    """Success (held and not crushed) per method and the effects, paired by phase and clustered by object pair."""
    out = {}
    for test, name in (("stationary", "grip_stationary.csv"), ("carry", "grip_carry.csv")):
        rows = read_csv(Path(folder) / name)
        for cap in ("1.5", "1.0"):
            found = {m: [r for r in rows if r["cap_Nm"] == cap and r["method"] == m] for m in ("rl", "reflex", "fixed")}
            pairs = sorted({r["pair"] for r in found["rl"]})

            def by_pair(m):
                return [[float(r["outcome"] == "held") for r in found[m] if r["pair"] == p] for p in pairs]
            success = {m: (float(np.mean([r["outcome"] == "held" for r in found[m]])) if test == "stationary"
                           else cluster_mean(by_pair(m))["estimate"]) for m in found}
            out[(test, cap)] = {"n": len(found["rl"]), "success": success,
                                "rl - reflex": paired_difference(by_pair("rl"), by_pair("reflex")),
                                "rl - fixed": paired_difference(by_pair("rl"), by_pair("fixed"))}
    return out


def llm(folder: Path = RESULTS) -> dict:
    """Requests compiled exactly; trip selection by the planner against the LLM choosing trips itself, in the planner's model
    (mean over layouts of unit means) and in simulated errands."""
    rows = read_csv(Path(folder) / "llm_requests.csv")
    compiled = {r["request"]: int(r["compiled_exactly"]) for r in rows}
    k, n = sum(compiled.values()), len(compiled)

    def clusters(method, measure):
        by = defaultdict(list)
        for r in rows:
            if r["method"] == method and r[measure] != "":
                by[int(r["layout"])].append(float(r[measure]))
        return by

    def mean(method, measure):
        c = clusters(method, measure)
        return cluster_mean([c[g] for g in sorted(c)])["estimate"]
    a, b = clusters("planner", "rate"), clusters("direct_llm", "rate")
    layouts = sorted(set(a) & set(b))
    model_gain = cluster_bootstrap([[a[g] for g in layouts], [b[g] for g in layouts]],
                                   lambda m: (m[..., 0] - m[..., 1]).mean(axis=-1) / m[..., 1].mean(axis=-1))
    sim = read_csv(Path(folder) / "llm_errands.csv")
    units = {(r["method"], int(r["layout"]), r["request"], int(r["seed"])): float(r["rate"]) for r in sim}
    keys = sorted({k[1:] for k in units if k[0] == "planner"} & {k[1:] for k in units if k[0] == "direct_llm"})
    xa, xb = defaultdict(list), defaultdict(list)
    for key in keys:
        xa[key[0]].append(units[("planner", *key)])
        xb[key[0]].append(units[("direct_llm", *key)])
    g = sorted(xa)
    sim_gain = cluster_bootstrap([[xa[x] for x in g], [xb[x] for x in g]],
                                 lambda m: (m[..., 0] - m[..., 1]).mean(axis=-1) / m[..., 1].mean(axis=-1))
    return {"compiled": (k, n, wilson(k, n)), "violations": {m: mean(m, "any_violation") for m in ("planner", "direct_llm")},
            "priority lost": {m: mean(m, "priority_lost") for m in ("planner", "direct_llm")}, "model gain": model_gain,
            "simulated gain": sim_gain, "simulated errands": {m: sum(1 for k in units if k[0] == m) for m in ("planner", "direct_llm")}}
