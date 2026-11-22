"""Known objects: the same plans picked up on poses estimated from the depth image and on exact poses
(21 plans, 20 pickups each).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexcarry.envs.two_object import TwoObjectEnv
from dexcarry.experiments.pickups import OBJECTS, REACH_MIN, reach_options, reached, robust_choice
from dexcarry.experiments.recovery_helpers import library_pairs
from dexcarry.paths import INPUTS, RUNS
from dexcarry.perception.perceive import perceive, perturbed
from dexcarry.pipeline.two_object import run_two_object
from dexcarry.sim.two_object import TwoObjectPlan

PERCEPTION_SEED = 25000
PERCEPTION_PICKUPS = 20
PERCEPTION_MIN_ACCEPTED = 5 # plans: the robust-rank top plan of every old library pair with at least 5 accepted plans
ARMS = ("exact", "estimated")
LIBRARY = INPUTS / "plan_library"
OUT = RUNS / "perception" / "known_objects"


def plans() -> list[tuple[str, int]]:
    out = []
    for name in library_pairs(LIBRARY):
        accepted = reached(LIBRARY, name) or {}
        if sum(v >= REACH_MIN for v in accepted.values()) < PERCEPTION_MIN_ACCEPTED:
            continue
        top = robust_choice(LIBRARY, name, top=1)
        if top:
            out.append((name, top[0]))
    return out


def run(args):

    index, count = args.shard
    args.out.mkdir(parents=True, exist_ok=True)
    for name, k in plans()[index::count]:
        first, second = name.split("+")
        path = args.out / f"pickups-{name}-p{k}.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        done = {r["arm"] for r in rows}
        plan = TwoObjectPlan.from_dict(json.loads((LIBRARY / f"plans-{name}.json").read_text(encoding="utf-8"))["plans"][k])
        with TwoObjectEnv(OBJECTS[first], OBJECTS[second], seed=PERCEPTION_SEED) as env:
            for arm in args.arms.split(";"):
                if arm in done:
                    continue
                results = []
                for repeat in range(args.pickups):
                    seed = PERCEPTION_SEED + repeat
                    env.reset(seed=seed)
                    env.perceived = None
                    look = perceive(env, np.random.default_rng([seed, 7]))
                    if arm == "estimated":
                        env.perceived = look.poses
                    elif arm.startswith("curve:"):
                        position, yaw = (float(v) for v in arm.removeprefix("curve:").split(","))
                        env.perceived = perturbed(env, (env.object_id, env.second_id), position, yaw,
                                                  np.random.default_rng([seed, 8]))
                    seen_both = arm != "estimated" or len(look.poses) == 2
                    r = run_two_object(env, plan, seed=seed, hold_seconds=5., options=reach_options()) if seen_both else None
                    env.perceived = None
                    results.append({"seed": seed, "both_held": bool(r and r["both_held"]),
                                    "first_failure": r["first_failure"] if r else "NOT_PERCEIVED",
                                    "look_seconds": look.seconds, "posture": look.posture,
                                    "errors": [list(look.errors.get(b, (None, None))) for b in (env.object_id, env.second_id)]})
                rows.append({"pair": name, "plan": k, "arm": arm, "results": results})
                path.write_text(json.dumps(rows), encoding="utf-8")
                print(name, k, arm, sum(x["both_held"] for x in results), "of", args.pickups, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("INDEX", "COUNT"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pickups", type=int, default=PERCEPTION_PICKUPS, help="pickups per plan")
    parser.add_argument("--arms", default=";".join(ARMS), help="pose sources, separated by ;")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
