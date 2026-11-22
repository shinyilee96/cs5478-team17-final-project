"""Runs errands on the robot and writes one record per errand.

    python -m dexcarry.experiments.errands run --family round --configs round-dexcarry --layouts 0 --seeds 26300 --distance 4

Families: ``round``, ``long`` and ``scarcity`` (partner-scarcity layouts). Configurations cover DexCarry, the single-object
baseline, the greedy planner and the two LLM arms.
"""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from dexcarry.config import load_hand_config
from dexcarry.envs.errand import DISC, LIBRARY_FIRST_Y, SLOT_X, ErrandEnv
from dexcarry.envs.round_table import draw_positions
from dexcarry.eval.provenance import begin_run, end_run
from dexcarry.experiments import measured_odds, partner_scarcity
from dexcarry.experiments.language import DEMO_ROLES, instructions, intent_of, make_client, roles
from dexcarry.experiments.outcome_model import CONSEQUENCE, FAILURE_STAGE, MODEL_DISTANCE, build_model
from dexcarry.experiments.pickups import OBJECTS, known_objects, reach_options, robust_rows
from dexcarry.experiments.predictors import ESTIMATOR_OUT, estimate, frozen_estimator
from dexcarry.experiments.recovery import recovery_model
from dexcarry.experiments.unseen_objects import primitive_cells, relabel
from dexcarry.models.plan_predictor import plan_features
from dexcarry.paths import INPUTS, ROOT, RUNS
from dexcarry.paths import stored as stored_folder
from dexcarry.perception.fit import nearest_cell
from dexcarry.perception.table_look import table_look
from dexcarry.pipeline.errand import ErrandRun, InjectedKnocks, blocking, known_grasps, run_errand
from dexcarry.pipeline.two_object import LIBRARY_SECOND_Y
from dexcarry.plan.errand_planner import ALL_PLANNERS as PLANNERS
from dexcarry.plan.errand_planner import Estimates, Intent, Option, apply_intent, drive_seconds, mdp_protect, option_values, pair_codes
from dexcarry.plan.llm_intent import ChooserLog, llm_chooser
from dexcarry.sim.loads import set_contact_friction
from dexcarry.sim.two_object import TwoObjectPlan

FAR, NEAR = 4., 1.
SETTINGS = ("standard", "mu0.8", "finestep") # the three simulator settings at 4 m; other distances use the standard setting only
LEAN_MAX = 10. # the plans under this predicted lean come first (degrees)
TRUSTED = .75 # 4S's lowest whole-trip estimate, 8 of 10 trips; fixed, not tuned
LAYOUTS = {"round": ROOT / "configs" / "errand_layouts_round.json", "long": ROOT / "configs" / "errand_layouts.json",
           "scarcity": ROOT / "configs" / "errand_layouts_scarcity.json"}
DEMO_INTENTS = ROOT / "runs" / "llm" / "demo" / "intents-demo.json" # the demo instruction's compiled intent (live, cached)
PLANS = INPUTS
SINGLE_GRASPS = INPUTS / "single_object_grasps.json" # the frozen order
UNRANKED_GRASPS = ROOT / "results" / "2026-10-18" / "single_errand" / "single_errand96" / "single-grasps.json" # force-closure order
PRIMITIVE_GRASPS = ROOT / "runs" / "primitive_single" / "single-grasps.json" # the 16 primitives', force-closure order
PRIMITIVE_PLANS = {"unseen": INPUTS / "unseen_objects", "known": ROOT / "runs" / "primitive_errand"}
OUT = RUNS / "errands"
TRIP_CHOICE = "practised-trips-n1.json" # whole-trip practice; practised-trips.json is the run before it
CLOSE_CAMERA = (np.array([1.5, -.8, 1.2]), np.array([.7, .1, .75])) # eye and target from the base: the hand and the table beside it
# The any-pair counter (configs/errand_layouts_anypair.json, not used in the report): any pair on the counter, plans in PLANS_A.
ANY_PAIR_LAYOUTS = ROOT / "configs" / "errand_layouts_anypair.json"
ANY_PAIR_PLANS = PLANS / "plans-a.json"
WIDE_LOOK = .72 # m between a pair's slots: wider gets a look at each object (one look covered two slots apart on the long table)
ERRAND_PLANS = INPUTS / "errand_plans.json" # the round table: each option's library candidates in E's order, no slot filter
SCENARIO = ROOT / "configs" / "scenario.json" # the scenario chosen by the model sweep's rule (p, r, severity)
TABLE_LOOK_STREAM = 9
LONG_TABLE_AIMS = (-.6, .6, 0.) # the long table's look, one image at each y (m) along the row at the slots' x; three cover 6 slots.
                                # The middle comes last, so the head ends facing the row's middle as after the round table's one image.
LONG_TABLE_DEPTH = .2 # the look's region reaches this far either side of the slots' x
ALLOWED = {"left1": 1, "left2": 2, "right1": -1} # where a pair's second slot may stand from the wrapped one (slots to the left)
STUDY_REACH = .24 # m: the long table's steps between the grasps (BASE_STEPS), whose time its t_pickup already holds
TRAVERSE_OVERHEAD = 1.5 # s: the hand's rise (about 1 s) and the stop after the drive (0.5 s), beside the drive itself
# What provenance hashes beside the whole source tree: the errand's code and every input it reads.
RUN_FILES = (*(ROOT / "dexcarry" / name for name in ("pipeline/errand.py", "pipeline/trip_stages.py", "pipeline/two_object.py", "pipeline/execution.py",
                                                    "envs/errand.py", "plan/errand_planner.py")),
             Path(__file__), *LAYOUTS.values(), PLANS / "plans.json", PLANS / "practised.json", PLANS / TRIP_CHOICE,
             SINGLE_GRASPS, UNRANKED_GRASPS, PRIMITIVE_GRASPS, PLANS / "estimator" / "frozen.json",
             ROOT / "dexcarry" / "perception" / "errand_look.py", ROOT / "dexcarry" / "experiments" / "recovery.py", ROOT / "configs" / "recovery_model.json")


@dataclass(frozen=True)
class Config:
    planner: str = "failure_aware"
    rule: str = "R" # which plan order: "R", "P", "E", "practised" or "atlas" (the primitive-shape library)
    recovery: str | None = "postures"
    vision: str = "model"
    release: str = "v1" # the wrapped object's release on pair trips
    lean: bool = False # prefer plans with a small predicted lean at the set-down
    selective: bool = False # offer only pairs whose practice estimate reaches TRUSTED
    ranked: bool = True # single trips use the robot-ranked grasps, not the force-closure order
    hold: float | None = None # seconds held after every lift; None is each trip's own (the hold-length sensitivity)
    retry: bool = False # one more single trip for an object still standing once every object has had its trip
    recovery_model: str | None = None # the planner's failure model knows this recovery's outcomes; None: the long table's
    any_pair: bool = False # the any-pair counter: every ordered pair with a wrappable first (PLANS_A), each option's own step
                           # with the hand raised, the carry measured from the base, two looks for a wide pair, Option.traverse
    study_first: bool = False # the any-pair counter: an option the long table could already do (a plan in plans.json) keeps the long table's
                              # plans, steps, carry, look and traverse 0, so it runs exactly as in the twin without any_pair
    study_only: bool = False # development proof only: offer only those options; must equal the twin errand for errand
    yaw_fix: bool = False # every look fits the yaw from four quarter turns, best residual kept (estimate's "quarter")
    round_table: bool = False # the round table: any pair from ERRAND_PLANS, the step with the hand raised and forward too,
                              # the carry from the base, two looks for a wide pair, traverse, the objects the whole-table look found,
                              # blocking (Option.needs_gone); family "round" only
    scenario: bool = False # knocks injected after a trip that fails at the table, as SCENARIO says (an
                           # assumption); round table only
    odds: str = "E" # "measured" gives the planner the measured odds of configs/scenario_odds.json (privileged)
    knows_knocks: bool = False # the planner's world model holds SCENARIO's knock model (Estimates.knocks)
    scarcity: bool = False # family "scarcity" only: the partner scarcity layouts (subsets of the round table's), positions by
                           # partner_scarcity.py.positions_for, errand_plans.json's rows re-indexed; no knocks
    intent: str | None = None # an instruction's errand on the round table (a job's ``instruction``):
                              # "compiled" plans on the evaluation's LLM-compiled intent (apply_intent, the planner weighted); "llm" asks
                              # the end-to-end LLM planner at every decision. None: no instruction, every earlier run


CONFIGS = {
    "round-baseline": Config("two_trips", "E", None, yaw_fix=True, round_table=True),
    "round-dexcarry": Config(rule="E", recovery="errand", yaw_fix=True, round_table=True),
    "round-greedy": Config("greedy_rate", rule="E", recovery="errand", yaw_fix=True, round_table=True),
    "long-baseline": Config("two_trips", "E", None, yaw_fix=True, round_table=True),
    "long-dexcarry": Config(rule="E", recovery="errand", yaw_fix=True, round_table=True),
    "scarcity-greedy": Config("greedy_rate", rule="E", recovery="errand", yaw_fix=True, round_table=True, scarcity=True, odds="measured"),
    "scarcity-dexcarry": Config(rule="E", recovery="errand", yaw_fix=True, round_table=True, scarcity=True, odds="measured"),
    "request-planner": Config(rule="E", recovery="errand", yaw_fix=True, round_table=True, intent="compiled"),
    "request-llm": Config(rule="E", recovery="errand", yaw_fix=True, round_table=True, intent="llm"),
}
INTENTS = INPUTS / "language" / "intents-eval.json" # the evaluation's compiled intents, frozen for physics


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def estimates():
    """Everything the planners believe that is not a plan's own odds: train objects only."""

    model = build_model(INPUTS / "plan_library", INPUTS / "trip_times" / "two_object", INPUTS / "trip_times" / "single_object")
    rows = [r for path in sorted((INPUTS / "trip_times" / "single_object_v2").glob("single-v2-*-d1.json")) for row in load(path) for r in row["results"]]
    est = Estimates(tuple(model.post_pickup), model.held_first, float(np.mean([r["delivered"] for r in rows])),
                    float(np.mean([r["seconds"] for r in rows])), MODEL_DISTANCE, model.t_first, model.t_pickup, model.place_both,
                    model.place_first, CONSEQUENCE, FAILURE_STAGE)
    return est, load(ESTIMATOR_OUT / "frozen.json")["failure_codes_pooled"]


def apply_setting(env, setting: str) -> None:
    """Apply a simulator setting (friction or time step); the next reset puts the time step in force."""
    if setting == "mu0.8":
        frictions = {spec.friction for spec in env.specs}
        if len(frictions) != 1:
            raise SystemExit("one hand coefficient cannot give the same contact friction on objects that differ")
        set_contact_friction(env.p, env.robot_id, .8, frictions.pop(), links=sorted(env.hand_link_indices))
    elif setting == "finestep":
        s = env.scenario
        env.scenario = replace(s, timestep=s.timestep / 2, substeps=s.substeps * 2, gain_reference_dt=1 / 240)
    elif setting != "standard":
        raise SystemExit(f"unknown setting {setting}")


def stance_of(slots, i: int) -> float:
    return slots[i][1] - LIBRARY_FIRST_Y


def predicted_lean(plan: dict) -> float:
    """Degrees the wrapped object would lean at the set-down with the palm as at the grasp and the object as it settled in
    the search. From the plan alone; a plan that records no settling counts as leaning."""
    if plan.get("first_held") is None:
        return np.inf
    pose, held = np.asarray(plan["first_pose"], float), np.asarray(plan["first_held"], float)
    return float(np.degrees(np.arccos(np.clip((pose[:3, :3] @ held[:3, :3])[2, 2], -1., 1.))))


def lean_first(order: list[int], stored: list[dict]) -> list[int]:
    """``order`` with the plans under ``LEAN_MAX`` first, each group kept in its own order, so no option is lost."""
    return sorted(order, key=lambda k: predicted_lean(stored[k]) >= LEAN_MAX)


def traverse(slots, i: int, j: int) -> float:
    """the any-pair counter's ``Option.traverse``: seconds between the grasps beyond the long table's. A step within ``STUDY_REACH`` costs nothing new
    (t_pickup holds it); a longer one costs its drive on the carry's profile plus ``TRAVERSE_OVERHEAD``. The step is the one that
    puts the second where the library's scene has it, from the nominal slots."""
    step = abs(slots[j][1] - slots[i][1] - (LIBRARY_SECOND_Y - LIBRARY_FIRST_Y))
    return 0. if step <= STUDY_REACH + 1e-9 else drive_seconds(step) + TRAVERSE_OVERHEAD


def known_options(layout: int, config: Config, slots, target: float):
    """The layout's pair options that have a plan, with the plans in the configuration's order and their odds."""
    if config.planner == "two_trips":
        return [], {}
    if config.any_pair and config.rule not in ("R", "P", "E"):
        raise SystemExit("the any-pair options have no practised plans")
    _, failures = estimates()
    rows = study_rows = [row for row in load(PLANS / "plans.json")["options"] if row["layout"] == layout]
    if config.any_pair:
        study_keys = {(row["i"], row["j"]) for row in study_rows} if config.study_first else set()
        own = [row for row in load(ANY_PAIR_PLANS)["options"] if row["layout"] == layout and (row["i"], row["j"]) not in study_keys]
        rows = [*study_rows, *([] if config.study_only else own)] if config.study_first else own
    e_orders = e_order_table() if config.rule == "E" else {}
    practised = load(PLANS / {"practised": "practised.json", "trips": TRIP_CHOICE}[config.rule]) if config.rule in ("practised", "trips") else {}
    options, plans = [], {}
    study_ids = {id(row) for row in study_rows} if not config.any_pair or config.study_first else set()
    for row in rows:
        on_study = id(row) in study_ids
        stored = load(stored_folder(row["library"]) / f"plans-{row['pair']}.json")["plans"]
        order = row["P"] if config.rule == "P" else e_orders[("study" if on_study else "a", row["name"])] if config.rule == "E" else row["R"]
        order = lean_first(order, stored) if config.lean else order
        success = row["E"][str(order[0])]
        if config.rule in ("practised", "trips"):
            chosen = practised[row["name"]]
            order = [chosen["plan"], *(k for k in row["R"] if k != chosen["plan"])]
            success = chosen["estimate"]
            if config.selective and success < TRUSTED:
                continue
        i, j = row["i"], row["j"]
        options.append(Option("pair", i, j, pair_codes(success, failures), target - stance_of(slots, i),
                              ((1., True, True),) if config.rule == "trips" else None,
                              traverse(slots, i, j) if config.any_pair and not on_study else 0.))
        plans[(i, j)] = [(k, TwoObjectPlan.from_dict(stored[k])) for k in order]
    return options, plans


@functools.lru_cache(maxsize=1)
def e_order_table() -> dict[tuple[str, str], list[int]]:
    """E's order of each option's plans, kept in PLANS_A: ("study",
    name) for the long table's options as plans.json holds them, ("a", name) for the any-pair counter's."""
    table = load(ANY_PAIR_PLANS)
    return {**{("study", name): order for name, order in table["study_E_order"].items()},
            **{("a", row["name"]): row["E_order"] for row in table["options"]}}


def round_options(layout: int, slots, target: float, found, rows=None) -> tuple[list, dict, dict]:
    """the round table: the layout's pair options among the found objects, every plan in E's order (ERRAND_PLANS), the odds E's
    estimate of the first plan until blocking says which is accepted, the drive and ``traverse`` from the estimated positions. Also
    returns each option's E estimates by plan. ``rows``: the rows to use, already the layout's; None reads
    ERRAND_PLANS's rows of ``layout``."""
    _, failures = estimates()
    options, plans, odds = [], {}, {}
    for row in (load(ERRAND_PLANS)["options"] if rows is None else rows):
        i, j = row["i"], row["j"]
        if (rows is None and row["layout"] != layout) or i not in found or j not in found:
            continue
        stored = load(stored_folder(row["library"]) / f"plans-{row['pair']}.json")["plans"]
        order = row["E_order"]
        options.append(Option("pair", i, j, pair_codes(row["E"][str(order[0])], failures), target - stance_of(slots, i), None,
                              traverse(slots, i, j)))
        plans[(i, j)] = [(k, TwoObjectPlan.from_dict(stored[k])) for k in order]
        odds[(i, j)] = row["E"]
    return options, plans, odds


def unblocked(everything: list, record: dict, plans: dict, odds: dict) -> list:
    """the round table: the options after the start's blocking dry runs: refused even with their blockers gone, left out; the rest with
    ``needs_gone``, and a pair's odds E's estimate of the plan accepted."""
    _, failures = estimates()
    out = []
    for o in everything:
        r = record[o.key]
        if r["open"] is None:
            continue
        o = replace(o, needs_gone=frozenset(r["needs_gone"]))
        if o.kind == "pair":
            k = plans[(o.i, o.j)][r["plan"]][0]
            o = replace(o, codes=pair_codes(odds[(o.i, o.j)][str(k)], failures))
        out.append(o)
    return out


def primitive_options(layout_names: list[str], family: str, slots, target: float):
    """Unseen objects planned from the depth image only: the reachable ordered pairs with a primitive-shape library choice, in that order. E's
    features describe the primitive cell pair: never the true object."""

    folder = PRIMITIVE_PLANS[family]
    choices, mapping = load(folder / "choices.json"), load(folder / "mapping.json")
    primitives = INPUTS / "primitive_shape_library"
    model, _ = frozen_estimator()
    _, failures = estimates()
    cfg = load_hand_config("leap")
    options, plans, kinds = [], {}, {}
    for i, first in enumerate(layout_names):
        for step in ALLOWED.values():
            j = i + step
            name = f"{first}+{layout_names[j]}" if 0 <= j < len(layout_names) else None
            choice = choices.get(name)
            if not choice or not choice.get("atlas_order"):
                continue
            cell_pair, second = choice["cell_pair"], layout_names[j]
            stored = load(primitives / f"plans-{cell_pair}.json")["plans"]
            top = choice["atlas_order"][0]
            reach = {r["plan"]: r["refusals"] for r in load(primitives / f"reach-{cell_pair}.json")}
            robust = {r["plan"]: r for r in (robust_rows(primitives, cell_pair) or [])}
            cell_first, cell_second = (OBJECTS[n] for n in cell_pair.split("+"))
            x = plan_features(TwoObjectPlan.from_dict(stored[top]), cell_first, cell_second, reach[top], robust.get(top), cfg)
            success = float(estimate(model, x[None])[0])
            options.append(Option("pair", i, j, pair_codes(success, failures), target - stance_of(slots, i)))
            plans[(i, j)] = [(k, relabel(stored[k], first, second)) for k in choice["atlas_order"]]
            kinds[(i, j)] = {role: mapping[name][role]["cell"].split("-")[2] for role in ("first", "second")}
    return options, plans, kinds


def primitive_grasps():
    """Single grasps for objects the system does not model: the nearest of the 16 primitives to the fit."""

    cells, library = primitive_cells(), load(PRIMITIVE_GRASPS)

    def grasps(i, fit):
        best = min(((name, gap) for role in ("w", "t") for name, gap in [nearest_cell(fit, role, cells)]), key=lambda v: (v[1], v[0]))[0]
        return known_grasps(cells[best], library[best]), best.split("-")[2]
    return grasps


def make_run(layout_names: list[str], layout: int, family: str, config: Config, slots, target: float, full: bool = True, env=None, seed=None,
             seen=None, instruction: dict | None = None):
    """The ``ErrandRun`` of one configuration on one layout; ``full`` adds each planning step's option values to the record. the round table
    (``config.round_table``) also needs the scene ``env``, the errand's ``seed`` and the whole-table look ``seen``: the blocking dry runs
    run here, before the errand, and their record is ``run.blocking``."""

    est, _ = estimates()
    if config.recovery_model == "errand":
        est = replace(est, recovery=recovery_model())
    kinds, odds = {}, {}
    found = sorted(seen.found) if seen is not None else range(len(layout_names))
    if config.round_table:
        if seen is None or env is None:
            raise SystemExit("these arms run on a table with a whole-table look")
        rows = None
        if config.scarcity:
            rows = partner_scarcity.scarcity_rows(partner_scarcity.entry_of(layout))
        elif family == "demo":
            rows = demo_rows(layout_names)
        pair_options, plans, odds = round_options(layout, slots, target, set(found), rows) if config.planner != "two_trips" else ([], {}, {})
        library = load(SINGLE_GRASPS)
        grasps = lambda i, fit: (known_grasps(OBJECTS[layout_names[i]], library[layout_names[i]]), None)
    elif config.rule == "atlas":
        pair_options, plans, kinds = primitive_options(layout_names, family, slots, target)
        grasps = primitive_grasps()
    else:
        pair_options, plans = known_options(layout, config, slots, target)
        library = load(SINGLE_GRASPS if config.ranked else UNRANKED_GRASPS)
        grasps = lambda i, fit: (known_grasps(OBJECTS[layout_names[i]], library[layout_names[i]]), None)
    singles = [Option("single", i, drive=target - stance_of(slots, i)) for i in found]
    everything = [*(() if config.planner == "two_trips" else pair_options), *singles]
    planner = PLANNERS[config.planner]
    retried = set()

    def decide(standing, closed):
        option = planner(est, [o for o in everything if o.key not in closed], standing)
        if option is None and config.retry:
            option = next((o for o in singles if o.i in standing and o.i not in retried), None)
            retried.update(() if option is None else {option.i})
        return option

    def values(standing, closed):
        return option_values(est, [o for o in everything if o.key not in closed], standing, "success" if config.planner == "mdp_success" else "world")
    options = (replace(reach_options(), step_to_second=True, step_forward=config.round_table) if config.any_pair or config.round_table
               else reach_options())
    study = None
    if config.any_pair and config.study_first:
        keys = {(row["i"], row["j"]) for row in load(PLANS / "plans.json")["options"] if row["layout"] == layout}
        study = lambda o: (o.i, o.j) in keys
    wide = config.any_pair or config.round_table
    run = ErrandRun(decide, lambda o: plans[(o.i, o.j)], grasps, options, config.recovery, config.vision,
                    lambda o: kinds[(o.i, o.j)], config.release, values if full else None, config.hold,
                    from_here=wide, wide_look=WIDE_LOOK if wide else None,
                    study=study, study_options=reach_options() if study is not None else None,
                    yaw_starts="quarter" if config.yaw_fix else "box", found=frozenset(found) if config.round_table else None)
    if config.round_table:
        run.blocking = blocking(env, run, everything, seed, seen.poses, seen.turns)
        everything[:] = unblocked(everything, run.blocking, plans, odds)
    if config.scenario or config.odds == "measured":
        scenario_run(run, config, everything, layout_names, slots, est)
    if config.intent is not None:
        intent_run(run, config, everything, layout_names, est, instruction, found)
    return run


def intent_run(run, config: Config, everything: list, names: list[str], est, instruction: dict, found) -> None:
    """the instruction's errand. "compiled": the evaluation's LLM-compiled intent (``INTENTS``) narrows the options
    (``apply_intent``) and weights the planner. "llm": the end-to-end LLM planner chooses every trip (replies from the cache),
    on every option. ``run.intent`` keeps what the record needs."""

    if instruction is None:
        raise SystemExit("an instruction arm needs a job's instruction")
    demo_table = roles(DEMO_ROLES) if instruction["id"].startswith("M") and instruction.get("roles") == "demo" else None #
    correct = intent_of(instruction["intent"], names, demo_table)
    record = {"instruction": instruction["id"], "text": instruction["text"], "correct": _intent_dict(correct)}
    if config.intent in ("compiled", "protect"):
        frozen = load(DEMO_INTENTS if instruction["id"].startswith("M") else INTENTS)[instruction["id"]]["compiled"]
        used = Intent.make(frozen["fetch"], dict(map(tuple, frozen["weights"])), map(tuple, frozen["before"]), frozen["never_pair"])
        allowed, weights = apply_intent(everything, used), used.weights_dict()
        planner = PLANNERS[config.planner]
        if config.intent == "protect": # the priority object first (within PROTECT_DELTA of its safest trip), then the weighted rate

            planner = functools.partial(mdp_protect, priority=used.priority)
        run.decide = lambda standing, closed: planner(est, [o for o in allowed if o.key not in closed], standing, weights=weights)
        if run.values is not None:
            run.values = lambda standing, closed: option_values(est, [o for o in allowed if o.key not in closed], standing, "world", weights)
        record["used"] = _intent_dict(used)
    elif config.intent == "llm":

        log = ChooserLog()
        client = make_client()
        chooser = llm_chooser(client, est, everything, instruction["text"], names, found, roles(), "v1", log)
        run.decide = chooser
        record["log"] = log
    else:
        raise SystemExit(f"unknown intent arm {config.intent}")
    run.intent = record
    run.intent_options = allowed if config.intent in ("compiled", "protect") else everything # the demo's model check reads these


def demo_rows(names: list[str]) -> list[dict]:
    """(DEMO ONLY): errand_plans.json's rows for the demo layout's object pairs, from any the round table layout (layout 5's first), the first row of
    each ordered pair of names, re-indexed to the demo's order."""
    round_names = load(LAYOUTS["round"])["layouts"]
    pos, rows, seen = {n: k for k, n in enumerate(names)}, [], set()
    for r in sorted(load(ERRAND_PLANS)["options"], key=lambda r: (r["layout"] != 5, r["layout"])):
        a, b = round_names[r["layout"]][r["i"]], round_names[r["layout"]][r["j"]]
        if a in pos and b in pos and (a, b) not in seen:
            seen.add((a, b))
            rows.append({**r, "i": pos[a], "j": pos[b], "source_i": r["i"], "source_j": r["j"]})
    return rows


def _intent_dict(intent) -> dict:
    return {"fetch": sorted(intent.fetch), "weights": [list(w) for w in intent.weights], "before": sorted(map(list, intent.before)),
            "never_pair": sorted(sorted(p) for p in intent.never_pair)}


def instruction_row(ident: str) -> dict:
    """an evaluation instruction by id (``configs/llm_intent/instructions_eval.json``); or the demo's ("M..", DEMO ONLY)."""
    rows = {r["id"]: r for r in instructions("eval")}
    rows.update({r["id"]: r for r in load(ROOT / "configs" / "llm_intent" / "instructions_demo.json")["instructions"]})
    if ident not in rows:
        raise SystemExit(f"no evaluation instruction {ident}")
    return rows[ident]


def scenario_run(run, config: Config, everything: list, names: list[str], slots, est) -> None:
    """the injected knocks on ``run``; measured odds in place of E's on every option (pairs: pickup success,
    failure codes, after-pickup table; singles: own rate), blocking kept; the knock model in the planner's world model. The planner
    reads ``est`` through ``decide`` and ``values``, so it is replaced in their closures' place: ``run.decide`` is rebuilt here."""

    chosen = load(SCENARIO)
    lost = chosen["severity"] == "lost"
    if config.scenario: # the partner scarcity arms take the measured odds only: no knocks
        run.knocks = InjectedKnocks(chosen["p"], chosen["r"], measured_odds.table_codes(), lost, chosen["seconds"])
    measured = measured_odds.load_odds()
    if config.odds == "measured":
        for k, o in enumerate(everything):
            if o.kind == "pair":
                post = tuple(tuple(x) for x in measured["post"].get(names[o.i], measured["pooled"]["post"]))
                everything[k] = replace(o, codes=pair_codes(measured_odds.pick_odds(measured, names[o.i], names[o.j]), measured["failures"]), post=post)
            else:
                everything[k] = replace(o, rate=measured["single"][names[o.i]]["p"] if names[o.i] in measured["single"] else measured["pooled"]["single"])
    if config.knows_knocks:
        est = replace(est, knocks=measured_odds.knock_model(list(slots), chosen["p"], chosen["r"], lost, measured))
    planner = PLANNERS[config.planner]
    run.decide = lambda standing, closed: planner(est, [o for o in everything if o.key not in closed], standing)
    if run.values is not None:
        run.values = lambda standing, closed: option_values(est, [o for o in everything if o.key not in closed], standing, "world")


def job_name(job: dict) -> str:
    name = f"{job['family']}-{job['config']}-L{job['layout']}-s{job['seed']}-{job['setting']}-d{job['distance']:g}"
    return f"{name}-{job['instruction']}" if job.get("instruction") else name # an instruction's errand names it


def scene_record(env) -> dict:
    """What a redrawing needs that the poses do not say: the scene's fixed bodies, as the simulator holds them, and the spots."""
    box = {name: [list(corner) for corner in env.p.getAABB(body)] for name, body in (("counter", env.table_id), ("target_table", env.target_table_id), ("floor", env.plane_id))}
    record = {"boxes": box, "spots": env.goal_spots.tolist(), "table_height": env.scenario.table_height}
    if getattr(env, "table", "box") == "disc": # the round table: the round top as the physics has it; "counter" is then its square AABB
        record["disc"] = {"centre": list(DISC["centre"]), "radius": DISC["radius"], "thickness": DISC["thickness"], "top": env.scenario.table_height}
    return record


def save_poses(env, path: Path, names: list[str]) -> None:
    """The demo's pose log: every logged step's joints and object poses, the scene's events and its
    fixed bodies, for redrawing offline."""
    steps, joints, bodies = zip(*env.pose_log, strict=True)
    np.savez_compressed(path, step=np.array(steps, np.int32), joints=np.array(joints, np.float32), bodies=np.array(bodies, np.float32),
                        joint_names=np.array(env.joint_names), objects=np.array(names), dt=env.dt, target_y=env.scenario.target_xy[1],
                        slots=np.array(env.slots), body_columns=np.array(["x", "y", "z", "qx", "qy", "qz", "qw"]),
                        events=np.array(json.dumps(env.events)), scene=np.array(json.dumps(scene_record(env))))


def run_job(job: dict, out: Path, record=None, poses=False, full=True, close=False) -> dict:
    """One errand. ``full`` keeps the trip records; off, with ``poses`` off, nothing is logged at all. ``close``
    makes a recording follow the robot."""

    layouts = load(LAYOUTS[job["family"]])
    names, config = layouts["layouts"][job["layout"]], CONFIGS[job["config"]]
    if config.round_table != (job["family"] in ("round", "long", "scarcity", "demo")) or config.scarcity != (job["family"] == "scarcity"):
        raise SystemExit(f"configuration {job['config']} does not run on the {job['family']} layouts")
    scene = {}
    if config.scarcity:
        entry = partner_scarcity.entry_of(job["layout"])
        positions, _ = partner_scarcity.positions_for(entry, job["seed"], entry["lure"])
        if positions is None:
            raise SystemExit(f"partner-scarcity layout {job['layout']}: no position qualifies at seed {job['seed']}")
        scene = {"positions": positions, "table": "disc"}
    elif job["family"] == "demo": # positions set by hand (DEMO ONLY)
        scene = {"positions": [tuple(xy) for xy in layouts["positions"][job["layout"]]], "table": "disc"}
    elif job["family"] == "long": # the long table: a counter with a row of slots
        scene = {}
    elif config.round_table:
        spacing = job.get("spacing") or layouts["spacing"] # the crowding test (development only) may override it
        positions = draw_positions(len(names), job["layout"], job["seed"], spacing)
        if positions is None:
            raise SystemExit(f"layout {job['layout']} cannot be drawn at seed {job['seed']}")
        scene = {"positions": positions, "table": "disc"}
    with ErrandEnv([OBJECTS[n] for n in names], seed=job["seed"], spacing=layouts["spacing"], distance=job["distance"], **scene) as env:
        apply_setting(env, job["setting"])
        if record:
            env.follow = CLOSE_CAMERA if close else None
            env.start_recording(record, eye=(3.2, env.scenario.target_xy[1] / 2 - .3, 2.2), target=(.1, env.scenario.target_xy[1] / 2 - .3, .7))
        env.reset(seed=job["seed"])
        if poses:
            env.pose_log, env.events = [], []
        if full:
            env.check_log = []
        seen, extra = None, {}
        if config.round_table:
            seen = whole_table_look(env, job["seed"], config)
            extra = {"positions": [list(xy) for xy in env.positions], "table_look": seen.record()}
        instruction = None
        if config.intent is not None or job.get("instruction"):
            if config.intent is None or not job.get("instruction") or job["family"] not in ("round", "demo"):
                raise SystemExit("request configurations run on the round table, each job with its request")
            instruction = instruction_row(job["instruction"])
            if instruction["layout"] != job["layout"]:
                raise SystemExit(f"instruction {job['instruction']} belongs to layout {instruction['layout']}")
        run = make_run(names, job["layout"], job["family"], config, env.slots, env.scenario.target_xy[1], full, env=env, seed=job["seed"], seen=seen,
                       instruction=instruction)
        if config.round_table:
            extra["blocking"] = [{"option": list(key), **value} for key, value in run.blocking.items()]
        result = {**run_errand(env, run, job["seed"]), **extra}
        if instruction is not None:
            intent = dict(run.intent)
            log = intent.pop("log", None)
            if log is not None:
                intent["llm"] = {"calls": log.calls, "stops": log.stops, "fallbacks": log.fallbacks, "keys": log.keys}
            result["intent"] = intent
        env.stop_recording()
        if poses:
            save_poses(env, out / f"poses-{job_name(job)}.npz", names)
        target = env.scenario.target_xy[1]
    return {**job, "names": names, "target_y": target, **result}


def whole_table_look(env, seed: int, config: Config):
    """the round table: the whole-table look at the errand's start, its time on the errand's clock; each found object's
    estimate becomes its slot for every later step."""

    row = {} if env.table == "disc" else {"aims": [(SLOT_X, y) for y in LONG_TABLE_AIMS], # the long table, panned along the row
                                          "box": ((SLOT_X - LONG_TABLE_DEPTH, SLOT_X + LONG_TABLE_DEPTH), (-env.half_width, env.half_width))}
    seen = table_look(env, np.random.default_rng([seed, TABLE_LOOK_STREAM]), DISC["centre"], DISC["radius"],
                      yaw_starts="quarter" if config.yaw_fix else "box", catalog=known_objects(), **row)
    for k, xy in seen.found.items():
        env.slots[k] = xy
    return seen


def plain(value):
    """``json.dumps``'s fallback: what the simulator's reports leave in NumPy."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def run(args):
    jobs = [{"family": args.family, "config": c, "layout": layout, "seed": seed, "setting": setting, "distance": args.distance,
             **({"spacing": args.round_spacing} if args.round_spacing is not None else {})}
            for c in args.configs for layout in args.layouts for seed in args.seeds for setting in args.settings]
    if getattr(args, "instructions", None): # each instruction on its own layout; --layouts is not used
        jobs = [{"family": args.family, "config": c, "layout": instruction_row(ins)["layout"], "seed": seed, "setting": setting,
                 "distance": args.distance, "instruction": ins}
                for c in args.configs for ins in args.instructions for seed in args.seeds for setting in args.settings]
    index, count = args.shard
    args.out.mkdir(parents=True, exist_ok=True)
    full = not args.bare
    files = (*RUN_FILES, ANY_PAIR_LAYOUTS, ANY_PAIR_PLANS) if any(CONFIGS[c].any_pair or CONFIGS[c].rule == "E" for c in args.configs) else RUN_FILES
    if any(CONFIGS[c].round_table for c in args.configs):
        files = (*files, ERRAND_PLANS, ROOT / "dexcarry" / "perception" / "table_look.py", ROOT / "dexcarry" / "envs" / "round_table.py")
    manifest = begin_run(args.out / "provenance", f"errands-{index}of{count}", {k: str(v) for k, v in vars(args).items()}, seeds=args.seeds, files=files) if full else None
    done = 0
    for job in jobs[index::count]:
        path = args.out / f"{job_name(job)}.json"
        if path.exists():
            continue
        result = run_job(job, args.out, args.record, args.poses, full, args.close)
        if manifest is not None:
            result["provenance"] = manifest.name
        path.write_text(json.dumps(result, default=plain), encoding="utf-8")
        done += 1
        print(job_name(job), f"{len(result['delivered'])} of {result['objects']} delivered ({len(result['upright'])} upright) in {result['seconds']:.0f} s, "
              f"{sum('refused' not in t for t in result['trips'])} trips", flush=True)
    if manifest is not None:
        end_run(manifest, errands=done)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--family", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--layouts", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[])
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=["standard"])
    parser.add_argument("--distance", type=float, default=FAR)
    parser.add_argument("--shard", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--record", type=Path, help="save a GIF of one errand")
    parser.add_argument("--close", action="store_true", help="--record's camera follows the robot, close up")
    parser.add_argument("--poses", action="store_true", help="also save each errand's pose log, poses-*.npz")
    parser.add_argument("--bare", action="store_true", help="outcomes only, no trip records or provenance")
    parser.add_argument("--any-pair-plans", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--round-spacing", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--instructions", nargs="+", help="request errands: the requests' ids, each on its own layout")
    args = parser.parse_args(argv)
    if args.any_pair_plans is not None:
        global ANY_PAIR_PLANS
        ANY_PAIR_PLANS = args.any_pair_plans
    run(args)


if __name__ == "__main__":
    main()
