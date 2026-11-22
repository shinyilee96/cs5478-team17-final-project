"""Unseen objects: pickups planned from the primitive-shape library for shapes fitted to the depth image, and
(privileged) from the true shape.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from dexcarry.envs.two_object import TwoObjectEnv
from dexcarry.experiments.pickups import OBJECTS, reach_options
from dexcarry.paths import INPUTS, RUNS
from dexcarry.perception.observe import observe
from dexcarry.perception.perceive import perceive
from dexcarry.pipeline.observed import ObservedPlanning
from dexcarry.pipeline.two_object import run_two_object
from dexcarry.sim.two_object import TwoObjectPlan

UNSEEN_PICKUPS = 20
UNSEEN_SEED = 9300
PRIMITIVE_MASS, PRIMITIVE_FRICTION = .15, .8 # the primitive-shape library's; an unseen object's mass is not known
# Candidates: the shape pair's plans that its own library's dry run accepted, in search order, kept if the arm planner
# accepts them at the unseen pair's estimated poses in REACH_MIN of REACH_SCENES scenes (one look per scene, planning on
# the fitted shapes only); at most 25.
# Library order: the re-checked plans by held-both count (re-check: the mapping look's fits, standard setting, held
# RECHECK_HOLD s), ties by the replay ranking R; then the other candidates in R order.
# True shape (privileged): the unseen pair's own library, candidates as in plan choice, its top three by R. A pair with no
# library candidate counts as 0 held and has no random plan to compare.
RECHECK_FRICTION, RECHECK_DT, RECHECK_HOLD = .4, 1 / 240, 5.
LOOK_SALT = 7
PRIMITIVE_LIBRARY = INPUTS / "primitive_shape_library"
UNSEEN = INPUTS / "unseen_true_shape_library"
OUT = RUNS / "perception" / "unseen_objects"
ROLES = {"first": "w", "second": "t"}


def primitive_cells() -> dict:
    return {k: v for k, v in OBJECTS.items() if k.startswith("atlas-")}


def fit_record(fit) -> dict:
    return {"kind": fit.kind, "sizes": [float(v) for v in fit.sizes], "centre": [float(v) for v in fit.centre],
            "yaw": float(fit.yaw), "residual": float(fit.residual), "other_residual": float(fit.other_residual)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stored_plans(folder: Path, name: str) -> list:
    return load(folder / f"plans-{name}.json")["plans"]


def relabel(data: dict, first: str, second: str):
    return dataclasses.replace(TwoObjectPlan.from_dict(data), first_id=first, second_id=second)


def jobs(args) -> list[tuple[str, str, int]]:
    choices = load(args.out / "choices.json")
    out = []
    for name, c in choices.items():
        out += [(name, "atlas", k) for k in c["candidates"]]
        out += [(name, "reference", k) for k in c["reference_top"]]
    return out


def run(args):
    """Label pickups: each job (pair, arm, plan) gets UNSEEN_PICKUPS pickups, each after its own
    look; the primitive-shape library arm plans on fitted-primitive proxies, the reference arm on model-based estimates."""

    maps = load(args.out / "mapping.json")
    index, count = args.shard
    env, env_pair = None, None
    try:
        for name, arm, k in jobs(args)[index::count]:
            path = args.out / f"label-{name}-{arm}-p{k}.json"
            if path.exists() or (getattr(args, "pair", None) and name != args.pair):
                continue
            first, second = name.split("+")
            if env_pair != name:
                if env is not None:
                    env.close()
                env, env_pair = TwoObjectEnv(OBJECTS[first], OBJECTS[second], seed=UNSEEN_SEED), name
            source = PRIMITIVE_LIBRARY if arm == "atlas" else UNSEEN
            plan = relabel(stored_plans(source, maps[name]["cell_pair"] if arm == "atlas" else name)[k], first, second)
            kinds = {role: maps[name][role]["cell"].split("-")[2] for role in ROLES} if arm == "atlas" else None
            results = []
            for repeat in range(args.pickups):
                seed = UNSEEN_SEED + repeat
                env.reset(seed=seed)
                rng = np.random.default_rng([seed, LOOK_SALT])
                view = None
                if arm == "atlas":
                    look = observe(env, rng)
                    seen = set(look.fits) == {"first", "second"}
                    if seen:
                        view = ObservedPlanning.build(env, look.fits, kinds)
                        env.observed = view
                    extra = {"fits": {r: fit_record(f) for r, f in look.fits.items()}, "fit_errors": look.errors}
                else:
                    look = perceive(env, rng)
                    seen = len(look.poses) == 2
                    env.perceived = look.poses
                    extra = {"errors": [list(look.errors.get(b, (None, None))) for b in (env.object_id, env.second_id)]}
                try:
                    r = run_two_object(env, plan, seed=seed, hold_seconds=5., options=reach_options()) if seen else None
                finally:
                    env.observed, env.perceived = None, None
                    if view is not None:
                        view.remove()
                results.append({"seed": seed, "both_held": bool(r and r["both_held"]),
                                "first_failure": r["first_failure"] if r else "NOT_PERCEIVED",
                                "look_seconds": look.seconds, "posture": look.posture, **extra})
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"pair": name, "arm": arm, "plan": k, "seeds": [x["seed"] for x in results],
                                        "results": results}), encoding="utf-8")
            print(name, {"atlas": "library", "reference": "true shape"}.get(arm, arm), k, sum(x["both_held"] for x in results), "of", args.pickups, flush=True)
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("INDEX", "COUNT"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pickups", type=int, default=UNSEEN_PICKUPS, help="pickups per plan")
    parser.add_argument("--pair", help="only this pair's jobs (each shard keeps its order and starts a fresh scene)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    main()
