"""Writes the per-trial result files (one row per errand, trip, pickup or grip phase) that every table is computed from."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dexcarry.experiments import rl_grip
from dexcarry.experiments.partner_scarcity import errand_story
from dexcarry.experiments.recovery import failed_pairs, saved

# Errand arms: (family, configuration) -> (table, layouts, method)
ERRAND_ARMS = {
    ("round", "round-dexcarry"): ("round", "random", "dexcarry"),
    ("round", "round-baseline"): ("round", "random", "single_object"),
    ("round", "round-greedy"): ("round", "random", "greedy"),
    ("long", "long-dexcarry"): ("long", "random", "dexcarry"),
    ("long", "long-baseline"): ("long", "random", "single_object"),
    ("scarcity", "scarcity-dexcarry"): ("round", "partner_scarcity", "dexcarry"),
    ("scarcity", "scarcity-greedy"): ("round", "partner_scarcity", "greedy"),
}
ERRAND_FIELDS = ["table", "layouts", "method", "layout", "seed", "setting", "distance", "objects", "delivered", "upright", "seconds",
                 "metres", "two_object_trips", "delivered_objects", "first_trip_tempting_pair", "scarce_object_delivered"]
TRIP_FIELDS = ["table", "layouts", "method", "layout", "seed", "setting", "distance", "trip", "kind", "first", "second", "began", "ended",
               "failure", "refused", "delivered", "recovery_steps", "saved_by_recovery"]


def arm_of(record: dict) -> tuple[str, str, str] | None:
    return ERRAND_ARMS.get((record["family"], record["config"]))


def errand_row(record: dict) -> dict:
    """One errand: its counts, time and driving, and for partner-scarcity layouts what happened to the scarce object."""
    table, layouts, method = arm_of(record)
    row = {"table": table, "layouts": layouts, "method": method,
           **{k: record[k] for k in ("layout", "seed", "setting", "distance", "objects")},
           "delivered": len(record["delivered"]), "upright": len(record["upright"]), "seconds": repr(float(record["seconds"])),
           "metres": repr(float(record["metres"])),
           "two_object_trips": sum(t["kind"] == "pair" and "refused" not in t for t in record["trips"]),
           "delivered_objects": " ".join(map(str, sorted(record["delivered"]))), "first_trip_tempting_pair": "", "scarce_object_delivered": ""}
    if layouts == "partner_scarcity":
        story = errand_story(record)
        row["first_trip_tempting_pair"] = int(story["lure_first"])
        row["scarce_object_delivered"] = int(story["u_delivered"])
    return row


def trip_rows(record: dict) -> list[dict]:
    """Every trip of one errand; a failed two-object trip also gets the objects its recovery saved."""
    table, layouts, method = arm_of(record)
    failed = {id(t) for t in failed_pairs(record)}
    rows = []
    for k, t in enumerate(record["trips"]):
        rows.append({"table": table, "layouts": layouts, "method": method,
                     **{f: record[f] for f in ("layout", "seed", "setting", "distance")}, "trip": k, "kind": t["kind"],
                     "first": t.get("i", ""), "second": "" if t.get("j") is None else t["j"],
                     "began": repr(float(t["began"])) if "began" in t else "", "ended": repr(float(t["ended"])) if "ended" in t else "",
                     "failure": t.get("failure") or "", "refused": t.get("refused") or "",
                     "delivered": " ".join(sorted(str(s) for s, ok in t.get("delivered", {}).items() if ok)),
                     "recovery_steps": " ".join(step["action"] for step in (t.get("recovery") or [])),
                     "saved_by_recovery": saved(t) if id(t) in failed else ""})
    return rows


def errand_records(folders: list[Path]) -> list[dict]:
    """Every errand record of the report's arms in ``folders``, in file order."""
    out = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*.json")):
            if path.stem.startswith("poses-"):
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(r, dict) and "family" in r and arm_of(r) is not None:
                out.append(r)
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
    return len(rows)


def write_errands(records: list[dict], out: Path) -> tuple[int, int]:
    records = sorted(records, key=lambda r: (arm_of(r), r["distance"], r["layout"], r["seed"], r["setting"]))
    return (write_csv(out / "errands.csv", ERRAND_FIELDS, [errand_row(r) for r in records]),
            write_csv(out / "trips.csv", TRIP_FIELDS, [t for r in records for t in trip_rows(r)]))


GRIP_FIELDS = ["pair", "plan", "state", "setting", "cap_Nm", "method", "outcome", "peak_N", "firm_hold_N", "limit_N", "margin", "mass_kg",
               "failure"]


def grip_rows(rows: list[dict], carry: bool) -> list[dict]:
    """One row per (phase, motor cap, method) with some constant grip that holds: the outcome scored against the crush limit.
    For the carry test the outcome is at the end of the carry."""
    frozen = rl_grip.FROZEN
    out = []
    for cap in rl_grip.MOTORS:
        if carry:
            res = rl_grip.carry_results(rows, cap)
            named = (("rl", "ppo"), ("reflex", "reflex"), ("fixed", "fixed"))
        else:
            res = rl_grip.results(rows, cap, fixed=frozen["fixed"][cap], specs=[frozen["ppo"], frozen["reflex"]])
            named = (("rl", frozen["ppo"]), ("reflex", frozen["reflex"]), ("fixed", "fixed"))
        for method, key in named:
            for r, outcome, run in res.get(key, []):
                g, force, limit = rl_grip.world(r, cap)
                out.append({"pair": r["task"].split("#")[0], "plan": r["task"], "state": r["state"], "setting": r["setting"], "cap_Nm": cap,
                            "method": method, "outcome": outcome["end"] if carry else outcome, "peak_N": round(run["peak"], 3),
                            "firm_hold_N": round(force, 3), "limit_N": round(limit, 3), "margin": round(r["margin"], 4),
                            "mass_kg": round(r["mass"], 4), "failure": run["failure"] or ""})
    return out


LLM_ARMS = {"2": "planner", "5": "direct_llm"}
LLM_MEASURES = ("rate", "priority_lost", "share", "violations", "any_violation")
LLM_REQUEST_FIELDS = ["request", "kind", "layout", "seed", "compiled_exactly", "method", *LLM_MEASURES]
LLM_ERRAND_FIELDS = ["request", "layout", "seed", "method", *LLM_MEASURES, "seconds", "trips"]
LLM_PHYSICS_ARMS = {"request-planner": "planner", "request-llm": "direct_llm"}


def llm_request_rows(units: list[dict]) -> list[dict]:
    """Per request-and-scene unit (in evaluation order) and method: the measures of its errand valued in the planner's model."""
    out = []
    for u in units:
        for arm, method in LLM_ARMS.items():
            m = u["arms"][arm]
            out.append({"request": u["id"], "kind": u["kind"], "layout": u["layout"], "seed": u["seed"],
                        "compiled_exactly": int(u["accuracy"]["whole"]), "method": method,
                        **{k: "" if m[k] is None else repr(float(m[k])) for k in LLM_MEASURES}})
    return out


def llm_errand_rows(measured: list[dict]) -> list[dict]:
    """Per simulated errand (rows of ``language.read_physics``'s measures, in file order)."""
    return [{"request": r["instruction"], "layout": r["layout"], "seed": r["seed"], "method": LLM_PHYSICS_ARMS[r["arm"]],
             **{k: "" if r[k] is None else repr(float(r[k])) for k in LLM_MEASURES}, "seconds": repr(float(r["seconds"])), "trips": r["trips"]}
            for r in measured if r["arm"] in LLM_PHYSICS_ARMS]


PICKUP_FIELDS = ["pair", "plan", "seed", "both_held", "first_failure"]


def plan_choice_rows(folder: Path) -> list[dict]:
    """Robot pickups of every candidate plan (files ``robot-labels-PAIR.json`` of a plan-choice run), in pair then plan order."""
    out = []
    for path in sorted(Path(folder).glob("robot-labels-*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            for seed, held, failure in zip(row["seeds"], row["both_held"], row["first_failure"], strict=True):
                out.append({"pair": row["pair"], "plan": row["plan"], "seed": seed, "both_held": int(held), "first_failure": failure or ""})
    return out


def perception_rows(folder: Path, order: list[tuple[str, int]]) -> list[dict]:
    """Known objects: each plan's pickups with exact and with estimated poses, plans in ``order``."""
    by = {}
    for path in sorted(Path(folder).glob("pickups-*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            by[(row["pair"], row["plan"], row["arm"])] = row["results"]
    return [{"pair": pair, "plan": plan, "poses": poses, "seed": x["seed"], "both_held": int(x["both_held"]), "first_failure": x["first_failure"] or ""}
            for pair, plan in order for poses in ("exact", "estimated") if (pair, plan, poses) in by for x in by[(pair, plan, poses)]]


def unseen_rows(folder: Path) -> list[dict]:
    """Unseen objects: each plan's pickups, from the primitive-shape library or (privileged) from the true shape."""
    out = []
    for path in sorted(Path(folder).glob("label-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        for x in row["results"]:
            out.append({"pair": row["pair"], "plans": {"atlas": "primitive_library", "reference": "true_shape"}[row["arm"]], "plan": row["plan"],
                        "seed": x["seed"], "both_held": int(x["both_held"]), "first_failure": x["first_failure"] or ""})
    return out
