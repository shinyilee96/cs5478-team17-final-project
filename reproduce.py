"""Reproduces the results in the body of the DexCarry report.

    python reproduce.py check # install check
    python reproduce.py tables # every table and number from results/
    python reproduce.py figures # Figure 2
    python reproduce.py PART [--quick | --full] # plan-choice, perception, recovery, errands, planners, ext-grip, ext-llm

--quick (the default) reruns one recorded trial; --full reruns the whole experiment (hours) with at most --jobs processes
(default 14). Reruns on another machine may differ slightly. Do not set OMP_NUM_THREADS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from importlib import metadata
from pathlib import Path

from dexcarry.experiments import compact, rl_grip, tables
from dexcarry.experiments.figures import distance_figure, planner_figure
from dexcarry.experiments.language import physics_measures
from dexcarry.experiments.perception import plans
from dexcarry.experiments.rankings import rebuilt_top3
from dexcarry.experiments.rl_grip import FROZEN
from dexcarry.experiments.unseen_objects import jobs
from dexcarry.paths import INPUTS, RESULTS, RUNS
from setup.prepare_data import verify

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SUMMARY = RESULTS / "summary"
NOTE = ("Note: on another machine a rerun can differ slightly from the recorded row (floating-point order); "
        "results should agree within the reported intervals.")


def p1(d: dict) -> str:
    return f"{100 * d['estimate']:+.1f} ({100 * d['low']:+.1f} to {100 * d['high']:+.1f})"


def r1(d: dict) -> str:
    return f"{100 * d['estimate']:+.1f}% ({100 * d['low']:+.1f} to {100 * d['high']:+.1f})"


REPORT = {
    "plan-choice": {"R - random": "+14.5 (+9.4 to +19.8)", "P - random": "+19.4 (+14.0 to +24.7)", "E - random": "+17.6 (+13.3 to +21.9)",
                    "P - R": "+4.9 (-0.8 to +10.3)", "E - R": "+3.0 (-2.8 to +8.9)"},
    "perception": {"known objects: estimated - exact poses": "+1.7 (-4.0 to +7.9)",
                   "unseen objects: primitive library - random primitive plan": "+12.9 (+6.9 to +19.0)",
                   "unseen objects: primitive library, both held (%)": "42.0 (26.8 to 58.2)",
                   "unseen objects: true shape (privileged) - primitive library": "+18.2 (+0.4 to +35.9)"},
    "recovery": {"round table: objects saved per failed two-object trip": "0.25 (0.06 to 0.50)",
                 "long table: objects saved per failed two-object trip": "0.70 (0.33 to 1.10)"},
    "table1": {("round", "1 m"): ("+11.6% (-2.2 to +23.9)", "-5.3 (-16.7 to +4.7)"), ("long", "1 m"): ("-0.2% (-16.2 to +15.2)", "-13.5 (-26.9 to -1.3)"),
               ("round", "2 m"): ("+15.8% (-1.0 to +30.7)", "-4.7 (-17.6 to +6.8)"), ("long", "2 m"): ("+3.7% (-9.7 to +15.5)", "-13.0 (-24.0 to -2.6)"),
               ("round", "4 m"): ("+25.0% (+5.3 to +44.9)", "-3.4 (-18.1 to +10.0)"), ("long", "4 m"): ("+10.8% (-3.9 to +24.3)", "-12.6 (-24.4 to -2.2)"),
               ("round", "8 m"): ("+30.8% (+16.0 to +44.7)", "-4.7 (-14.8 to +5.0)"), ("long", "8 m"): ("+22.9% (+9.7 to +35.3)", "-12.8 (-23.7 to -2.5)"),
               ("round", "12 m"): ("+36.4% (+22.0 to +49.4)", "-3.3 (-11.9 to +5.2)"), ("long", "12 m"): ("+31.5% (+17.2 to +45.3)", "-7.6 (-17.9 to +1.9)"),
               ("round", "16 m"): ("+40.0% (+24.7 to +53.9)", "-0.7 (-9.7 to +8.7)"), ("long", "16 m"): ("+40.0% (+24.5 to +56.2)", "-6.7 (-16.8 to +2.9)"),
               ("round", "4 m, pooled"): ("+25.4% (+12.7 to +37.8)", "-3.9 (-13.5 to +5.3)"),
               ("long", "4 m, pooled"): ("+16.2% (+5.2 to +26.9)", "-8.7 (-16.7 to -1.3)")},
    "errands": {"delivery share at 4 m, DexCarry (round, long)": "84%, 84%", "delivery share at 4 m, baseline (round, long)": "88%, 93%",
                "median trip at 4 m: two-object (round, long)": "75 s, 76 s", "median trip at 4 m: single-object (round, long)": "58 s, 56 s",
                "trips ending with a failure code at 4 m (round): two-object, single-object": "32%, 12%",
                "trips ending with a failure code at 4 m (long): two-object, single-object": "46%, 7%",
                "long table break-even distance (fit, 'around 1 m')": "0.66 m (-0.49 to 2.73)"},
    "planners": {"random layouts: failure-aware - greedy": "+2.0% (-4.0 to +8.7)", "partner scarcity: failure-aware - greedy": "+6.2% (-1.6 to +14.1)",
                 "partner scarcity, greedy: first trip took the tempting pair": "72 of 72",
                 "partner scarcity: box delivered (greedy, failure-aware)": "57 of 72, 72 of 72"},
    "table3": {("stationary", "1.5"): ("539", "63.1", "53.2", "32.8", "+8.9 (+2.7 to +15.5)", "+30.2 (+18.2 to +42.0)"),
               ("stationary", "1.0"): ("469", "64.2", "61.0", "37.7", "+1.9 (-4.6 to +8.4)", "+24.6 (+10.5 to +38.1)"),
               ("carry", "1.5"): ("180", "54.8", "40.8", "37.7", "+14.0 (+4.5 to +24.4)", "+17.0 (-1.0 to +34.1)"),
               ("carry", "1.0"): ("157", "55.0", "44.3", "43.3", "+10.8 (+3.2 to +18.7)", "+11.7 (-9.5 to +30.7)")},
    "ext-llm": {"requests compiled exactly (Wilson 95%)": "40 of 40 (91 to 100%)",
                "instruction violations: planner, direct LLM planning": "0.0%, 6.8%",
                "weighted delivery rate in the planner's model: planner - direct LLM": "+6.1% (+1.2 to +11.1)",
                "priority object lost: planner, direct LLM planning": "14.9%, 5.8%",
                "simulated errands (24 per method): planner - direct LLM": "+5.0% (-11.1 to +19.2)"},
}


def compare(rows: list[tuple[str, str, str]], title: str, write: str | None = None) -> bool:
    """Print (label, computed, report) rows; True if every computed value matches the report's."""
    ok = all(c == r for _, c, r in rows)
    width = max(len(label) for label, _, _ in rows)
    lines = [title] + [f"  {label:<{width}}  {c:<26} report {r:<26} {'matches' if c == r else 'DIFFERS'}" for label, c, r in rows]
    lines.append(f"  {'all values match the report' if ok else 'SOME VALUES DIFFER FROM THE REPORT'}")
    print("\n".join(lines) + "\n")
    if write:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / write).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def show_plan_choice(folder=RESULTS, write="plan_choice.txt"):
    v = tables.plan_choice(folder)
    rows = [(k, p1(v[k]), REPORT["plan-choice"][k]) for k in ("R - random", "P - random", "E - random", "P - R", "E - R")]
    return compare(rows, f"Plan choice (Table 2): each rule's top 3 plans against a random plan, {v['pairs']} held-out pairs, 20 robot "
                         "pickups per plan; points of both objects held", write)


def show_perception(folder=RESULTS, write="perception.txt"):
    k, u = tables.perception(folder), tables.unseen_objects(folder)
    held = u["both held"]
    rows = [("known objects: estimated - exact poses", p1(k["estimated - exact"]), REPORT["perception"]["known objects: estimated - exact poses"]),
            ("unseen objects: primitive library - random primitive plan", p1(u["library - random"]),
             REPORT["perception"]["unseen objects: primitive library - random primitive plan"]),
            ("unseen objects: primitive library, both held (%)", f"{100 * held['estimate']:.1f} ({100 * held['low']:.1f} to {100 * held['high']:.1f})",
             REPORT["perception"]["unseen objects: primitive library, both held (%)"]),
            ("unseen objects: true shape (privileged) - primitive library", p1(u["true shape - library"]),
             REPORT["perception"]["unseen objects: true shape (privileged) - primitive library"])]
    return compare(rows, f"Perception (Table 2): known objects, {k['plans']} plans x 20 pickups per pose source; unseen objects, "
                         f"{u['pairs']} pairs; points of both objects held", write)


def show_recovery(folder=RESULTS, write="recovery.txt"):
    v = tables.recovery(folder)
    rows = [(f"{t} table: objects saved per failed two-object trip", f"{v[t]['estimate']:.2f} ({v[t]['low']:.2f} to {v[t]['high']:.2f})",
             REPORT["recovery"][f"{t} table: objects saved per failed two-object trip"]) for t in ("round", "long")]
    return compare(rows, f"Recovery (Table 2): DexCarry at 4 m, all three settings; {v['round']['failed']} and {v['long']['failed']} failed "
                         "two-object trips", write)


def show_errands(folder=RESULTS, write=True):
    t1 = tables.table1(folder)
    rows, table_rows = [], []
    for table in ("round", "long"):
        for d in ("1 m", "2 m", "4 m", "8 m", "12 m", "16 m", "4 m, pooled"):
            v = t1[(table, d)]
            rate, share = r1(v["rate"]), p1(v["share"])
            want = REPORT["table1"][(table, d)]
            rows += [(f"{table} table, {d}: delivery rate", rate, want[0]), (f"{table} table, {d}: delivery share (pts)", share, want[1])]
            table_rows.append({"table": table, "drive": d, "errands_per_method": v["errands"], "delivery_rate_change_pct": round(100 * v["rate"]["estimate"], 1),
                               "rate_low": round(100 * v["rate"]["low"], 1), "rate_high": round(100 * v["rate"]["high"], 1),
                               "delivery_share_diff_pts": round(100 * v["share"]["estimate"], 1), "share_low": round(100 * v["share"]["low"], 1),
                               "share_high": round(100 * v["share"]["high"], 1)})
    ok = compare(rows, "Table 1: DexCarry against the single-object baseline (delivery rate change; delivery share difference in points)",
                 "table1.txt" if write else None)
    if write:
        compact.write_csv(SUMMARY / "table1.csv", list(table_rows[0]), table_rows)
    ts, be = tables.trip_statistics(folder), tables.break_even(folder)
    two, one = "two-object trips (DexCarry)", "single-object trips (baseline)"
    rows = [("delivery share at 4 m, DexCarry (round, long)", f"{t1[('round', 'share dexcarry 4 m')]:.0%}, {t1[('long', 'share dexcarry 4 m')]:.0%}"),
            ("delivery share at 4 m, baseline (round, long)", f"{t1[('round', 'share single_object 4 m')]:.0%}, {t1[('long', 'share single_object 4 m')]:.0%}"),
            ("median trip at 4 m: two-object (round, long)", f"{ts[('round', two)]['median_s']:.0f} s, {ts[('long', two)]['median_s']:.0f} s"),
            ("median trip at 4 m: single-object (round, long)", f"{ts[('round', one)]['median_s']:.0f} s, {ts[('long', one)]['median_s']:.0f} s"),
            ("trips ending with a failure code at 4 m (round): two-object, single-object",
             f"{ts[('round', two)]['failure_share']:.0%}, {ts[('round', one)]['failure_share']:.0%}"),
            ("trips ending with a failure code at 4 m (long): two-object, single-object",
             f"{ts[('long', two)]['failure_share']:.0%}, {ts[('long', one)]['failure_share']:.0%}"),
            ("long table break-even distance (fit, 'around 1 m')", f"{be['L_star']:.2f} m ({be['low']:.2f} to {be['high']:.2f})")]
    ok &= compare([(label, c, REPORT["errands"][label]) for label, c in rows],
                  "Section 5.2: arms and trips at 4 m (all three settings), and the long table's break-even fit (standard setting, 1-16 m)",
                  "trips.txt" if write else None)
    return ok


def show_planners(folder=RESULTS, write=True):
    v = tables.planners(folder)
    g, m = v["scarcity greedy"], v["scarcity dexcarry"]
    rows = [("random layouts: failure-aware - greedy", r1(v["random layouts"]["rate"]), REPORT["planners"]["random layouts: failure-aware - greedy"]),
            ("partner scarcity: failure-aware - greedy", r1(v["partner scarcity"]["rate"]), REPORT["planners"]["partner scarcity: failure-aware - greedy"]),
            ("partner scarcity, greedy: first trip took the tempting pair", f"{g['first trip took the tempting pair']} of {g['errands']}",
             REPORT["planners"]["partner scarcity, greedy: first trip took the tempting pair"]),
            ("partner scarcity: box delivered (greedy, failure-aware)", f"{g['scarce object delivered']} of {g['errands']}, {m['scarce object delivered']} of {m['errands']}",
             REPORT["planners"]["partner scarcity: box delivered (greedy, failure-aware)"])]
    return compare(rows, "Figure 2b and Section 5.4: the failure-aware planner against the greedy planner, round table, 4 m, delivery rate change",
                   "planners.txt" if write else None)


def show_grip(folder=RESULTS, write="table3.txt"):
    v = tables.grip(folder)
    rows = []
    for key, want in REPORT["table3"].items():
        x = v[key]
        got = (str(x["n"]), *(f"{100 * x['success'][m]:.1f}" for m in ("rl", "reflex", "fixed")), p1(x["rl - reflex"]), p1(x["rl - fixed"]))
        label = f"{key[0]} ({key[1]} N.m)"
        for name, c, r in zip(("n", "RL %", "reflex %", "fixed %", "RL - reflex", "RL - fixed"), got, want):
            rows.append((f"{label}: {name}", c, r))
    return compare(rows, "Table 3: fragile-grip success (held without crushing) and effects in points. Stationary success is pooled over "
                         "phases; carry success is the mean over object pairs; effects are paired by phase and clustered by pair", write)


def show_llm(folder=RESULTS, write="llm.txt"):
    v = tables.llm(folder)
    k, n, (lo, hi) = v["compiled"]
    rows = [("requests compiled exactly (Wilson 95%)", f"{k} of {n} ({100 * lo:.0f} to {100 * hi:.0f}%)"),
            ("instruction violations: planner, direct LLM planning", f"{100 * v['violations']['planner']:.1f}%, {100 * v['violations']['direct_llm']:.1f}%"),
            ("weighted delivery rate in the planner's model: planner - direct LLM", r1(v["model gain"])),
            ("priority object lost: planner, direct LLM planning", f"{100 * v['priority lost']['planner']:.1f}%, {100 * v['priority lost']['direct_llm']:.1f}%"),
            ("simulated errands (24 per method): planner - direct LLM", r1(v["simulated gain"]))]
    return compare([(label, c, REPORT["ext-llm"][label]) for label, c in rows],
                   "LLM extension (Section 6.2): 40 explicit requests x 2 seeds over the 12 round-table layouts, scored in the planner's model "
                   "from cached LLM replies; 12 requests also run as simulated errands", write)


def figures(folder=RESULTS, out=RESULTS / "figures"):
    t1, pl = tables.table1(folder), tables.planners(folder)
    # The report's figures were drawn from the values as printed (one decimal); so are these.
    points = {name: {float(d.split()[0]): tuple(round(100 * t1[(table, d)]["rate"][k], 1) for k in ("estimate", "low", "high"))
                     for d in ("1 m", "2 m", "4 m", "8 m", "12 m", "16 m")} for name, table in (("Round table", "round"), ("Long table", "long"))}
    bars = [tuple(round(100 * pl[s]["rate"][k], 1) for k in ("estimate", "low", "high")) for s in ("random layouts", "partner scarcity")]
    a, b = distance_figure(points, out), planner_figure(bars, out)
    print(f"Figure 2a -> {a.relative_to(ROOT)}\nFigure 2b -> {b.relative_to(ROOT)}")
    for name, p in points.items():
        print(f"  {name}: " + ", ".join(f"{d:g} m {v[0]:+.1f}%" for d, v in p.items()))
    print(f"  failure-aware - greedy: random layouts {bars[0][0]:+.1f}%, partner scarcity {bars[1][0]:+.1f}%")


SHOW = {"plan-choice": [show_plan_choice], "perception": [show_perception], "recovery": [show_recovery], "errands": [show_errands],
        "planners": [show_planners], "ext-grip": [show_grip], "ext-llm": [show_llm]}


def module(name: str, *args) -> list[str]:
    """``python -m dexcarry.experiments.NAME ARGS``, run from the repository root (paths inside it given relative to it)."""
    def plain(a):
        return Path(a).relative_to(ROOT).as_posix() if isinstance(a, Path) and Path(a).is_relative_to(ROOT) else str(a)
    return [PY, "-m", f"dexcarry.experiments.{name}", *map(plain, args)]


def run_one(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  running: {' '.join(command[2:])}  (log {log.relative_to(ROOT)})", flush=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"the command failed (exit {code}); see {log}")
    print(f"  done in {time.time() - started:.0f} s", flush=True)


def run_pool(commands: list[list[str]], jobs: int, logs: Path, stagger: float = 3.) -> None:
    """Run ``commands`` with at most ``jobs`` processes, started ``stagger`` seconds apart; a log per command."""
    logs.mkdir(parents=True, exist_ok=True)
    running, failed, started = [], [], time.time()
    for i, command in enumerate(commands):
        while sum(p.poll() is None for p, _ in running) >= jobs:
            time.sleep(2)
        handle = (logs / f"{i:04d}.log").open("w", encoding="utf-8")
        running.append((subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT), handle))
        print(f"  started {i + 1} of {len(commands)} ({time.time() - started:.0f} s)", flush=True)
        time.sleep(stagger)
    for p, handle in running:
        p.wait()
        handle.close()
        if p.returncode != 0:
            failed.append(p.args)
    print(f"  {len(commands)} processes in {(time.time() - started) / 60:.0f} min; {len(failed)} failed (logs in {logs.relative_to(ROOT)})")
    if failed:
        raise SystemExit("some processes failed; rerun the same command to resume (finished outputs are kept)")


def same_rows(new: list[dict], old: list[dict], fields: list[str]) -> bool:
    """Compare rerun rows with recorded rows field by field and print the first differences."""
    diffs = [(i, f, str(a.get(f)), str(b.get(f))) for i, (a, b) in enumerate(zip(new, old)) for f in fields if str(a.get(f)) != str(b.get(f))]
    if len(new) != len(old):
        print(f"  {len(new)} rerun rows against {len(old)} recorded rows")
    for i, f, a, b in diffs[:8]:
        print(f"  row {i}, {f}: rerun {a}, recorded {b}")
    ok = not diffs and len(new) == len(old)
    print(f"  {'the rerun equals the recorded rows' if ok else 'the rerun DIFFERS from the recorded rows'} ({len(old)} rows compared)\n  {NOTE}\n")
    return ok


def recorded(name: str, **match) -> list[dict]:
    return [r for r in tables.read_csv(RESULTS / name) if all(str(r[k]) == str(v) for k, v in match.items())]


# Errands: family and configuration of each table's methods.
ARMS = {(table, layouts, method): (family, config) for (family, config), (table, layouts, method) in compact.ERRAND_ARMS.items()}


def errand_command(table, layouts, method, layout, seed, setting, distance, out: Path) -> list[str]:
    family, config = ARMS[(table, layouts, method)]
    return module("errands", "run", "--family", family, "--configs", config, "--layouts", layout, "--seeds", seed, "--settings", setting,
                  "--distance", distance, "--out", out)


def quick_errand(table, layouts, method, layout, seed, setting, distance) -> bool:
    out = RUNS / "quick" / "errands"
    family, config = ARMS[(table, layouts, method)]
    stem = f"{family}-{config}-L{layout}-s{seed}-{setting}-d{distance:g}"
    for old in (out / f"{stem}.json", out / f"poses-{stem}.npz"):
        old.unlink(missing_ok=True) # the runner skips an errand whose record exists
    run_one(errand_command(table, layouts, method, layout, seed, setting, distance, out), out / "logs" / f"{table}-{method}-L{layout}-s{seed}.log")
    record = next(r for r in compact.errand_records([out]) if (compact.arm_of(r), r["layout"], r["seed"], r["setting"], r["distance"])
                  == ((table, layouts, method), layout, seed, setting, distance))
    key = {"table": table, "layouts": layouts, "method": method, "layout": layout, "seed": seed, "setting": setting}
    old_errand = [r for r in recorded("errands.csv", **key) if float(r["distance"]) == distance]
    old_trips = [r for r in recorded("trips.csv", **key) if float(r["distance"]) == distance]
    scarcity = " (partner scarcity)" if layouts == "partner_scarcity" else ""
    print(f"  {table} table{scarcity}, {method}, layout {layout}, seed {seed}, {setting}, {distance:g} m: {record['objects']} objects, "
          f"{len(record['delivered'])} delivered in {record['seconds']:.1f} s")
    return same_rows([compact.errand_row(record)] + compact.trip_rows(record), old_errand + old_trips,
                     sorted(set(compact.ERRAND_FIELDS) | set(compact.TRIP_FIELDS)))


ROUND_ERRANDS = [("round", "random", m, layout, seed, setting, 4.) for m in ("dexcarry", "single_object") for layout in range(12)
                 for seed in (26300, 26301) for setting in ("standard", "mu0.8", "finestep")] + \
                [("round", "random", m, layout, seed, "standard", d) for d in (1., 2., 8., 12., 16.) for m in ("dexcarry", "single_object")
                 for layout in range(12) for seed in (26300, 26301)]
LONG_ERRANDS = [("long", *job[1:]) for job in ROUND_ERRANDS]
GREEDY_ERRANDS = [("round", "random", "greedy", layout, seed, setting, 4.) for layout in range(12) for seed in (26300, 26301)
                  for setting in ("standard", "mu0.8", "finestep")]
SCARCITY_ERRANDS = [("round", "partner_scarcity", m, layout, seed, "standard", 4.) for m in ("greedy", "dexcarry") for layout in range(12)
                    for seed in range(26500, 26506)]


def full_errands(jobs: list[tuple], workers: int) -> Path:
    """Run ``jobs`` (one process per errand, each skipped if its record exists) into runs/errands; write its per-trial files."""
    out = RUNS / "errands"
    run_pool([errand_command(*job, out) for job in jobs], workers, out / "logs")
    folder = out / "results"
    print("  per-trial files:", compact.write_errands(compact.errand_records([out]), folder), "(errands, trips) in", folder.relative_to(ROOT))
    return folder


QUICK_PLAN_PAIR = ("tall", "ndgn-6") # the smallest pair: 5 candidate plans; its first candidate runs first in a fresh process
QUICK_PERCEPTION_PLAN = 0 # the first of the 21 plans
QUICK_UNSEEN_PAIR = "unseen-8+unseen-6" # 12 library plans and 3 true-shape plans


def part_plan_choice(mode: str, workers: int) -> None:
    rankings = tables.read_csv(RESULTS / "plan_choice_rankings.csv")
    if mode == "quick":
        print("Rankings rebuilt from the plan library and the frozen models P and E:")
        same = rebuilt_top3(rankings)
        print(f"  {'every' if same else 'NOT every'} rule's top 3 equals the frozen ranking on all {len(rankings)} ranked pairs\n")
        first, second = QUICK_PLAN_PAIR
        name = f"{first}+{second}"
        out = RUNS / "quick" / "plan-choice"
        out.mkdir(parents=True, exist_ok=True)
        for f in (f"plans-{name}.json", f"reach-{name}.json"):
            shutil.copy2(INPUTS / "plan_choice_library" / f, out / f)
        plan = int(next(r for r in rankings if r["pair"] == name)["candidates"].split()[0])
        (out / f"robot-labels-{name}.json").unlink(missing_ok=True)
        run_one(module("pickups", "robot", "--pair", first, second, "--out", out, "--accepted", "--max-plans", 25, "--repeats", 20, "--seed", 9100,
                       "--base-steps", "--symmetry", "none", "--tag=-labels", "--plan", plan), out / "logs" / f"{name}.log")
        new = compact.plan_choice_rows(out)
        same_rows(new, recorded("plan_choice_pickups.csv", pair=name, plan=plan), compact.PICKUP_FIELDS)
    else:
        out = RUNS / "plan-choice"
        out.mkdir(parents=True, exist_ok=True)
        commands = []
        for r in rankings:
            name = r["pair"]
            for f in (f"plans-{name}.json", f"reach-{name}.json"):
                shutil.copy2(INPUTS / "plan_choice_library" / f, out / f)
            if not (out / f"robot-labels-{name}.json").exists():
                commands.append(module("pickups", "robot", "--pair", *name.split("+"), "--out", out, "--accepted", "--max-plans", 25, "--repeats", 20,
                                       "--seed", 9100, "--base-steps", "--symmetry", "none", "--tag=-labels"))
        run_pool(commands, workers, out / "logs")
        folder = out / "results"
        compact.write_csv(folder / "plan_choice_pickups.csv", compact.PICKUP_FIELDS, compact.plan_choice_rows(out))
        shutil.copy2(RESULTS / "plan_choice_rankings.csv", folder / "plan_choice_rankings.csv")
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_plan_choice(folder, None)


def part_perception(mode: str, workers: int) -> None:
    order = plans()
    if mode == "quick":
        out = RUNS / "quick" / "perception"
        pair, plan = order[QUICK_PERCEPTION_PLAN]
        (out / f"pickups-{pair}-p{plan}.json").unlink(missing_ok=True)
        run_one(module("perception", "run", "--shard", QUICK_PERCEPTION_PLAN, len(order), "--out", out), out / "logs" / "known.log")
        same_rows(compact.perception_rows(out, [(pair, plan)]), [r for r in recorded("perception_pickups.csv", pair=pair, plan=plan)],
                  ["pair", "plan", "poses", "seed", "both_held", "first_failure"])
        unseen = out / "unseen"
        unseen.mkdir(parents=True, exist_ok=True)
        for f in ("choices.json", "mapping.json"):
            shutil.copy2(INPUTS / "unseen_objects" / f, unseen / f)

        every = jobs(Namespace(out=unseen))
        shard = next(i for i in range(10) if len([j for j in every[i::10] if j[0] == QUICK_UNSEEN_PAIR]) == 1)
        name, arm, k = next(j for j in every[shard::10] if j[0] == QUICK_UNSEEN_PAIR)
        (unseen / f"label-{name}-{arm}-p{k}.json").unlink(missing_ok=True)
        run_one(module("unseen_objects", "run", "--shard", shard, 10, "--out", unseen, "--pair", QUICK_UNSEEN_PAIR), out / "logs" / "unseen.log")
        plans_name = {"atlas": "primitive_library", "reference": "true_shape"}[arm]
        same_rows(compact.unseen_rows(unseen), recorded("unseen_pickups.csv", pair=name, plans=plans_name, plan=k),
                  ["pair", "plans", "plan", "seed", "both_held", "first_failure"])
    else:
        out = RUNS / "perception"
        known, unseen = out / "known_objects", out / "unseen_objects"
        unseen.mkdir(parents=True, exist_ok=True)
        for f in ("choices.json", "mapping.json"):
            shutil.copy2(INPUTS / "unseen_objects" / f, unseen / f)
        n = min(workers, len(order))
        commands = [module("perception", "run", "--shard", i, n, "--out", known) for i in range(n)]
        commands += [module("unseen_objects", "run", "--shard", i, 10, "--out", unseen) for i in range(10)] # the recorded run's 10 shards
        run_pool(commands, workers, out / "logs")
        folder = out / "results"
        compact.write_csv(folder / "perception_pickups.csv", ["pair", "plan", "poses", "seed", "both_held", "first_failure"],
                          compact.perception_rows(known, order))
        compact.write_csv(folder / "unseen_pickups.csv", ["pair", "plans", "plan", "seed", "both_held", "first_failure"], compact.unseen_rows(unseen))
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_perception(folder, None)


def part_errands(mode: str, workers: int) -> None:
    if mode == "quick":
        quick_errand("round", "random", "dexcarry", 0, 26300, "standard", 4.)
        quick_errand("long", "random", "dexcarry", 0, 26300, "standard", 4.)
    else:
        folder = full_errands(ROUND_ERRANDS + LONG_ERRANDS, workers)
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_errands(folder, False)


def part_recovery(mode: str, workers: int) -> None:
    if mode == "quick":
        print("In this errand the wrapped object is lost while driving on the second trip, and recovery still delivers the other one:")
        quick_errand("round", "random", "dexcarry", 0, 26301, "standard", 4.)
    else:
        jobs = [j for j in ROUND_ERRANDS + LONG_ERRANDS if j[2] == "dexcarry" and j[6] == 4.]
        folder = full_errands(jobs, workers)
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_recovery(folder, None)


def part_planners(mode: str, workers: int) -> None:
    if mode == "quick":
        quick_errand("round", "partner_scarcity", "greedy", 0, 26500, "standard", 4.)
    else:
        jobs = [j for j in ROUND_ERRANDS if j[2] == "dexcarry" and j[6] == 4.] + GREEDY_ERRANDS + SCARCITY_ERRANDS
        folder = full_errands(jobs, workers)
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_planners(folder, False)


def grip_command(test: str, out: Path, *extra) -> list[str]:
    seed, more = (32600, ()) if test == "stationary" else (32700, ("--settings", "standard", "--carry", 1))
    return module("rl_grip", "run", "--out", out, "--pairs", "test", "--seed", seed, "--methods", FROZEN["reflex"], FROZEN["ppo"], *more, *extra)


def part_grip(mode: str, workers: int) -> None:
    if mode == "quick":
        for test in ("stationary", "carry"):
            out = RUNS / "quick" / "ext-grip" / test
            for p in out.glob("grip-*.json"):
                p.unlink()
            # Shard 8 of the recorded 14 starts with plan bar-3cm+slab#0; its first phase (state 0, standard setting) is scored at both caps.
            settings = ("--settings", "standard") if test == "stationary" else () # the carry test always runs the standard setting
            run_one(grip_command(test, out, "--shard", 8, 14, "--tasks", 1, "--states", 1, *settings), out / "logs" / "run.log")
            rows, _ = rl_grip.load(out)
            new = compact.grip_rows(rows, test == "carry")
            keys = {(r["plan"], str(r["state"]), r["setting"]) for r in new}
            old = [r for r in tables.read_csv(RESULTS / f"grip_{test}.csv") if (r["plan"], r["state"], r["setting"]) in keys]
            print(f"  {test}: phase {sorted(keys)}")
            same_rows(new, old, compact.GRIP_FIELDS)
    elif mode == "train":
        out = RUNS / "ext-grip" / "training"
        run_one(module("rl_grip", "train", "--workers", 13, "--steps", 6_000_000, "--n-steps", 512, "--checkpoint-every", 1_000_000, "--start", .1,
                       "--slip-cost", 0, "--out", out), out / "train.log")
        print(f"  the new policy's checkpoints are in {out.relative_to(ROOT)}; the results use the frozen weights/grip_policy")
    else:
        out = RUNS / "ext-grip"
        commands = [grip_command(test, out / test, "--shard", i, 14) for test in ("stationary", "carry") for i in range(14)]
        run_pool(commands, workers, out / "logs")
        folder = out / "results"
        for test in ("stationary", "carry"):
            rows, _ = rl_grip.load(out / test)
            compact.write_csv(folder / f"grip_{test}.csv", compact.GRIP_FIELDS, compact.grip_rows(rows, test == "carry"))
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_grip(folder, None)


LLM_REQUESTS = ("E03", "E06", "E09", "E12", "E15", "E16", "E22", "E25", "E29", "E32", "E36", "E40") # the 12 also run as errands


def llm_errand_command(method: str, request: str, seed: int, out: Path) -> list[str]:
    config = {"planner": "request-planner", "direct_llm": "request-llm"}[method]
    return module("errands", "run", "--family", "round", "--configs", config, "--instructions", request, "--seeds", seed, "--settings", "standard",
                  "--distance", 4, "--out", out)


def llm_measured(folder: Path) -> list[dict]:
    out = []
    for p in sorted(folder.glob("round-request-*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out.append({"arm": r["config"], "instruction": r["instruction"], "layout": r["layout"], "seed": r["seed"], **physics_measures(r)})
    return out


def part_llm(mode: str, workers: int) -> None:
    out = RUNS / ("quick" if mode == "quick" else "") / "ext-llm"
    reading = out / "model"
    if reading.exists():
        shutil.rmtree(reading)
    run_one(module("language", "reading", "--out", reading), out / "logs" / "model.log")
    units = json.loads((reading / "units.json").read_text(encoding="utf-8"))
    print("The requests scored in the planner's model (from the cached replies):")
    same_rows(compact.llm_request_rows(units), tables.read_csv(RESULTS / "llm_requests.csv"), compact.LLM_REQUEST_FIELDS)
    errands = out / "errands"
    if mode == "quick":
        for p in errands.glob("*E03*"):
            p.unlink()
        run_one(llm_errand_command("direct_llm", "E03", 26300, errands), out / "logs" / "errand.log")
        new = compact.llm_errand_rows(llm_measured(errands))
        same_rows(new, recorded("llm_errands.csv", request="E03", seed=26300, method="direct_llm"), compact.LLM_ERRAND_FIELDS)
    else:
        commands = [llm_errand_command(m, q, s, errands) for m in ("planner", "direct_llm") for q in LLM_REQUESTS for s in (26300, 26301)]
        run_pool(commands, workers, out / "logs" / "errands")
        folder = out / "results"
        compact.write_csv(folder / "llm_requests.csv", compact.LLM_REQUEST_FIELDS, compact.llm_request_rows(units))
        compact.write_csv(folder / "llm_errands.csv", compact.LLM_ERRAND_FIELDS, compact.llm_errand_rows(llm_measured(errands)))
        print(f"From this rerun ({folder.relative_to(ROOT)}):")
        show_llm(folder, None)


PARTS = {"plan-choice": part_plan_choice, "perception": part_perception, "recovery": part_recovery, "errands": part_errands,
         "planners": part_planners, "ext-grip": part_grip, "ext-llm": part_llm}


def check() -> None:
    """Install check: packages, robot model, prepared data, one table from results/, one robot pickup against its recorded row."""
    started = time.time()
    print(f"Python {sys.version.split()[0]}")
    for name in ("numpy", "scipy", "pybullet", "scikit-learn", "stable-baselines3", "torch"):
        print(f"  {name} {metadata.version(name)}")
    if not verify():
        raise SystemExit("prepared data or robot assets missing or different; see README, Setup")
    show_plan_choice(RESULTS, None)
    out = RUNS / "quick" / "check"
    name = "+".join(QUICK_PLAN_PAIR)
    out.mkdir(parents=True, exist_ok=True)
    for f in (f"plans-{name}.json", f"reach-{name}.json"):
        shutil.copy2(INPUTS / "plan_choice_library" / f, out / f)
    plan = int(next(r for r in tables.read_csv(RESULTS / "plan_choice_rankings.csv") if r["pair"] == name)["candidates"].split()[0])
    (out / f"robot-labels-{name}.json").unlink(missing_ok=True)
    run_one(module("pickups", "robot", "--pair", *QUICK_PLAN_PAIR, "--out", out, "--accepted", "--repeats", 2, "--seed", 9100, "--base-steps",
                   "--symmetry", "none", "--tag=-labels", "--plan", plan), out / "check.log")
    same_rows(compact.plan_choice_rows(out), recorded("plan_choice_pickups.csv", pair=name, plan=plan)[:2], compact.PICKUP_FIELDS)
    print(f"install check finished in {time.time() - started:.0f} s")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["check", "tables", "figures", *PARTS])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="mode", action="store_const", const="quick", help="rerun one recorded trial (default)")
    mode.add_argument("--full", dest="mode", action="store_const", const="full", help="rerun the whole experiment")
    mode.add_argument("--train", dest="mode", action="store_const", const="train", help="ext-grip only: train a new policy (optional)")
    parser.add_argument("--jobs", type=int, default=14, help="--full: at most this many simulator processes (default 14)")
    args = parser.parse_args(argv)
    if args.what == "check":
        return check()
    if args.what == "tables":
        ok = all(f() for fs in SHOW.values() for f in fs)
        print(f"Summaries written to {SUMMARY.relative_to(ROOT)}. {'Every value matches the report.' if ok else 'SOME VALUES DIFFER.'}")
        if not ok:
            raise SystemExit(1)
        return None
    if args.what == "figures":
        return figures()
    if args.mode == "train" and args.what != "ext-grip":
        parser.error("--train is for ext-grip only")
    print("From the recorded per-trial files (results/):")
    for f in SHOW[args.what]:
        f(RESULTS, None)
    PARTS[args.what](args.mode or "quick", max(1, min(args.jobs, 14)))


if __name__ == "__main__":
    main()
