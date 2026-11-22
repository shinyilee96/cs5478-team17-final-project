"""Two-object pickups on the robot: runs a pair's stored plans and records each pickup.

    python -m dexcarry.experiments.pickups robot --pair tall ndgn-6 --out FOLDER --accepted --repeats 20 --seed 9100 --base-steps --symmetry none --tag=-labels

``OBJECTS`` lists every object the experiments use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dexcarry.data.catalog import ObjectSpec, load_catalog
from dexcarry.envs.two_object import TwoObjectEnv
from dexcarry.paths import ROOT
from dexcarry.pipeline.two_object import BASE_STEPS, MORE_SECOND_RETREATS, SECOND_RETREATS, ReachOptions, run_two_object
from dexcarry.sim.loads import set_contact_friction
from dexcarry.sim.two_object import TwoObjectPlan

OBJECTS = {
    "bar-2cm": ObjectSpec("bar-2cm", half_extents=(.01, .01, .07), mass=.04),
    "bar-3cm": ObjectSpec("bar-3cm", half_extents=(.015, .015, .07), mass=.06),
    "bar-4cm": ObjectSpec("bar-4cm", half_extents=(.02, .02, .07), mass=.08),
    "bar-heavy": ObjectSpec("bar-heavy", half_extents=(.015, .015, .07), mass=.25),
    "stick": ObjectSpec("stick", half_extents=(.015, .015, .06), mass=.06),
    "slim": ObjectSpec("slim", half_extents=(.02, .02, .08), mass=.08),
    "tall": ObjectSpec("tall", half_extents=(.025, .025, .06), mass=.12),
    "tall-heavy": ObjectSpec("tall-heavy", half_extents=(.025, .025, .06), mass=.4),
    "can": ObjectSpec("can", half_extents=(.03, .03, .05), mass=.15),
    "cube-s": ObjectSpec("cube-s", half_extents=(.025, .025, .035), mass=.1),
    "slab": ObjectSpec("slab", half_extents=(.045, .02, .04), mass=.1),
}
# Real mesh objects prepared to stand on the table; absent until the preparation script has run.
MESH_CATALOG = ROOT / "data" / "two_object" / "objects.json"
if MESH_CATALOG.exists():
    OBJECTS.update(load_catalog(MESH_CATALOG))
# The new boxes and meshes of the plan-choice experiment; their names never clash with the above.
PLAN_CHOICE_CATALOG = MESH_CATALOG.parents[1] / "choose" / "objects.json"
if PLAN_CHOICE_CATALOG.exists():
    _new = load_catalog(PLAN_CHOICE_CATALOG)
    if _new.keys() & OBJECTS.keys():
        raise SystemExit(f"{PLAN_CHOICE_CATALOG} redefines {sorted(_new.keys() & OBJECTS.keys())}")
    OBJECTS.update(_new)
# the plan-choice experiment batch 2: ndgn-4 onwards.
PLAN_CHOICE_CATALOG_2 = PLAN_CHOICE_CATALOG.with_name("objects-b2.json")
if PLAN_CHOICE_CATALOG_2.exists():
    _new = load_catalog(PLAN_CHOICE_CATALOG_2)
    if _new.keys() & OBJECTS.keys():
        raise SystemExit(f"{PLAN_CHOICE_CATALOG_2} redefines {sorted(_new.keys() & OBJECTS.keys())}")
    OBJECTS.update(_new)
# The primitive-shape library's shapes: atlas-w-* (wrapped role), atlas-t-* (three-finger).
PRIMITIVE_CATALOG = MESH_CATALOG.parents[1] / "atlas" / "objects.json"
if PRIMITIVE_CATALOG.exists():
    _new = load_catalog(PRIMITIVE_CATALOG)
    if _new.keys() & OBJECTS.keys():
        raise SystemExit(f"{PRIMITIVE_CATALOG} redefines {sorted(_new.keys() & OBJECTS.keys())}")
    OBJECTS.update(_new)
# The unseen objects of the primitive-shape library: unseen-1 onwards. They are here for
# the physics and the privileged reference search only; every catalogue the method sees is ``known_objects``.
UNSEEN_CATALOG = MESH_CATALOG.parents[1] / "unseen" / "objects.json"
if UNSEEN_CATALOG.exists():
    _new = load_catalog(UNSEEN_CATALOG)
    if _new.keys() & OBJECTS.keys():
        raise SystemExit(f"{UNSEEN_CATALOG} redefines {sorted(_new.keys() & OBJECTS.keys())}")
    OBJECTS.update(_new)


def known_objects() -> dict:
    """The robot's catalogue: every object except the primitive-shape library primitives and the unseen objects."""
    return {k: v for k, v in OBJECTS.items() if not k.startswith(("atlas-", "unseen-"))}


# Test 4, fixed before the robot data was read: the best plan of a pair is chosen on the
# even-numbered repeats and must hold both objects in at least half of the odd-numbered ones,
# on at least three pairs.
# Robust plans, fixed on before any robust replay was run. A plan from the search is
# kept only if, under every contact friction at every time step below (six settings), it holds
# both objects in at least ROBUST_MIN_HELD of ROBUST_REPEATS floating replays with small hand-pose
# error (2 mm, 1 degree), and a second grasp that starts with the hand on the table counts as a
# failure. Kept plans are ranked by their worst setting, then by their total over the settings,
# then by the order the search found them; the first ROBUST_TOP are the pair's plans. The replay
# executes the plan on the true objects: a privileged filter, like the search itself.
ROBUST_FRICTIONS, ROBUST_STEPS = (.8, .4, 1.2), (1 / 240, 1 / 480)
ROBUST_REPEATS, ROBUST_MIN_HELD, ROBUST_TOP = 5, 4, 3
# Reach, fixed the same day before any dry run of a new plan: a plan stays only if the arm
# planner accepts both of its arm motions (``pipeline.two_object.dry_run``, nothing executed) in at
# least REACH_MIN of REACH_SCENES scenes with redrawn object poses. The planner is asked as the trips will ask it: base
# moves allowed, no box symmetry.
# Ranking: worst floating setting, then scenes reached, then total held, then search order.
REACH_SCENES, REACH_MIN, REACH_SEED = 5, 4, 9000
# Feasibility of a stage-wise degradation study, fixed on before either test was run.
# Margin: every plan that holds both without pose error is tilt-tested in 14 pull directions
# at each load below, in g. A drive at the robot's limits adds at most 0.2 g, so if at least
# MARGIN_DEAD of the plans keep both objects in every direction at MARGIN_LEVEL, the carry has no
# margin to lose for these objects.
MARGIN_LEVELS, MARGIN_LEVEL, MARGIN_DEAD = (1., 1.5, 2., 3.), 1.5, .9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["robot"])
    parser.add_argument("--pair", nargs=2, metavar=("FIRST", "SECOND"), help=f"objects from {sorted(OBJECTS)}")
    parser.add_argument("--out", type=Path, required=True, help="folder with the pair's plans-PAIR.json and reach-PAIR.json; results go here")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-plans", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold", type=float, default=5., help="seconds both objects must stay held after the second lift")
    parser.add_argument("--contact-friction", type=float, default=None,
                        help="friction between hand and object (default 0.4: 0.5 on the hand times 0.8 on the object)")
    parser.add_argument("--tag", default="", help="suffix of the output file: robot{tag}-PAIR.json")
    parser.add_argument("--min-replay", type=float, default=.8,
                        help="run only plans that held both in this share of floating replays with small error")
    parser.add_argument("--shard", nargs=2, type=int, metavar=("INDEX", "COUNT"),
                        help="run only this slice of the plans; start one process per slice")
    parser.add_argument("--spacing", type=float, default=.24, help="distance between the two objects on the table (m)")
    parser.add_argument("--rrt-iters", type=int, default=500, help="planner iterations per arm motion")
    parser.add_argument("--more-retreats", action="store_true", help="six retreat directions for the second approach, not two")
    parser.add_argument("--base-steps", action="store_true",
                        help="let the base move between the grasps when the second object is out of reach")
    parser.add_argument("--symmetry", choices=["both", "second", "none"], default="both",
                        help="whose box symmetry may turn a plan into reach")
    parser.add_argument("--probe", action="store_true",
                        help="also plan the second arm motion under every planner setting, from the same lifted state")
    parser.add_argument("--record", type=Path, help="save the first trial as a GIF at this path and stop")
    parser.add_argument("--plan", type=int, help="run only the stored plan with this index")
    parser.add_argument("--halve-dt", action="store_true", help="physics at half the time step, twice the substeps")
    parser.add_argument("--dt-scaled-gains", action="store_true",
                        help="scale motor gains with the time step (PyBullet's stiffness is gain / time step)")
    parser.add_argument("--accepted", action="store_true",
                        help="only plans the planner's dry run accepts (needs reach), up to --max-plans, in search order")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.shard and not 0 <= args.shard[0] < args.shard[1]:
        parser.error("--shard needs 0 <= INDEX < COUNT")
    if not args.pair or not set(args.pair) <= OBJECTS.keys():
        parser.error("--pair needs two known objects")
    first, second = (OBJECTS[name] for name in args.pair)
    return robot(args, first, second, f"{first.object_id}+{second.object_id}")


def reached(out: Path, name):
    """Scenes in which the planner's dry run accepted each plan; None if the dry run has not been made."""
    path = out / f"reach-{name}.json"
    if not path.exists():
        return None
    return {row["plan"]: sum(r is None for r in row["refusals"]) for row in json.loads(path.read_text(encoding="utf-8"))}


def robust_rows(out: Path, name):
    """Records of the robust filter for one pair, shards together; None if it has not been run."""
    paths = [path for path in sorted(out.glob(f"robust-{name}*.json"))
             if path.stem == f"robust-{name}" or path.stem.removeprefix(f"robust-{name}-").replace("of", "").isdigit()]
    return [row for path in paths for row in json.loads(path.read_text(encoding="utf-8")) if row["pair"] == name] if paths else None


def reach_options():
    """How the planner is asked in the dry run and must be asked in the trips of chosen plans."""
    return ReachOptions(base_steps=BASE_STEPS, symmetry="none")


def robust_choice(out: Path, name, top=ROBUST_TOP):
    """Indices of the pair's plans, chosen in advance: accepted by the planner's dry run and kept
    by the robust filter, most robust first."""
    accepted, rows = reached(out, name), robust_rows(out, name)
    if accepted is None or rows is None:
        raise SystemExit(f"{name}: run the reach and robust actions first; plans must pass both")
    kept = [row for row in rows if row["kept"] and len(row["settings"]) == len(ROBUST_FRICTIONS) * len(ROBUST_STEPS)]
    held = {row["plan"]: [sum(s["both_held"]) for s in row["settings"]] for row in kept}
    return sorted((k for k in held if accepted.get(k, 0) >= REACH_MIN),
                  key=lambda k: (-min(held[k]), -accepted[k], -sum(held[k]), k))[:top]


def planner_options(args):
    return ReachOptions(rrt_iters=args.rrt_iters, second_retreats=MORE_SECOND_RETREATS if args.more_retreats else SECOND_RETREATS,
                        base_steps=BASE_STEPS if args.base_steps else (), symmetry=args.symmetry)


def probe_options():
    """Planner settings compared on the same lifted state: each cheap fix alone, then all together."""
    return {"baseline": ReachOptions(), "iters-2000": ReachOptions(rrt_iters=2000),
            "more-retreats": ReachOptions(second_retreats=MORE_SECOND_RETREATS),
            "base-steps": ReachOptions(base_steps=BASE_STEPS),
            "all": ReachOptions(rrt_iters=2000, second_retreats=MORE_SECOND_RETREATS, base_steps=BASE_STEPS)}


def passed_replay(out: Path, name, min_replay, max_plans):
    """Indices of the plans that passed the floating replay filter, in stored order."""
    replay = out / f"replay-{name}.json"
    if not replay.exists():
        raise SystemExit(f"{name}: run the replay action first; plans must pass the replay filter")
    # About one plan in six does not reproduce on the floating hand; those never reach the robot.
    return [row["plan"] for row in json.loads(replay.read_text(encoding="utf-8"))
            if np.mean(row["levels"]["small"]["both_held"]) >= min_replay][:max_plans]


def robot(args, first, second, name):
    """Run replay-filtered plans on the Fetch arm; one record per plan with every repeat's outcome."""

    stored = json.loads((args.out / f"plans-{name}.json").read_text(encoding="utf-8"))["plans"]
    if args.accepted:
        # Labels for the plan predictor: every plan the planner's dry run accepts, in search order.
        accepted = reached(args.out, name)
        if accepted is None:
            raise SystemExit(f"{name}: run the reach action first; --accepted runs only plans the planner accepts")
        passed = sorted(k for k, scenes in accepted.items() if scenes >= REACH_MIN)[:args.max_plans]
    else:
        passed = passed_replay(args.out, name, args.min_replay, args.max_plans)
    index, count = args.shard or (0, 1)
    options, rows = planner_options(args), []
    half = args.spacing / 2
    with TwoObjectEnv(first, second, seed=args.seed, first_xy=(.45, -half), second_xy=(.45, half)) as env:
        # Simulator settings; the next reset puts the time step in force.
        if args.contact_friction is not None:
            if first.friction != second.friction:
                raise SystemExit("one hand coefficient cannot give the same contact friction on objects that differ")
            set_contact_friction(env.p, env.robot_id, args.contact_friction, first.friction, links=sorted(env.hand_link_indices))
        if args.halve_dt:
            env.scenario = replace(env.scenario, timestep=env.scenario.timestep / 2, substeps=env.scenario.substeps * 2)
        if args.dt_scaled_gains:
            env.scenario = replace(env.scenario, gain_reference_dt=1 / 240)
        for k in passed[index::count] if args.plan is None else [args.plan]:
            plan = TwoObjectPlan.from_dict(stored[k])
            results = []
            for repeat in range(args.repeats):
                env.reset(seed=args.seed + repeat)
                if args.record:
                    # Close on the pick table, from across it and above, facing the robot.
                    env.start_recording(args.record, eye=(1.1, -.6, 1.2), target=(.42, 0., .76))
                results.append(run_two_object(env, plan, seed=args.seed + repeat, hold_seconds=args.hold, options=options,
                                              probe=probe_options() if args.probe else None))
                if args.record:
                    env.stop_recording()
                    print(name, k, results[-1]["first_failure"] or "BOTH_HELD", "recorded in", args.record)
                    return
            rows.append({"pair": name, "plan": k, "seeds": [args.seed + repeat for repeat in range(args.repeats)],
                         "both_held": [r["both_held"] for r in results],
                         "clean": [r["clean"] for r in results], "first_failure": [r["first_failure"] for r in results],
                         "diagnostics": [r["diagnostics"] for r in results]})
            print(name, k, f"both held {np.mean(rows[-1]['both_held']):.0%}", flush=True)
    suffix = f"-{index}of{count}" if args.shard else ""
    (args.out / f"robot{args.tag}-{name}{suffix}.json").write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
