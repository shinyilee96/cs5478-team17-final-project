"""The RL grip extension: grip a fragile second object without knowing its mass or crush limit. Compares the PPO policy,
a slip reflex and the best fixed grip at motor caps of 1.5 and 1.0 N m.

    python -m dexcarry.experiments.rl_grip run --pairs test --seed 32600 --methods reflex:0.1:1.2:0.25 ppo:weights/grip_policy/checkpoints/ppo_5999994_steps.zip --shard K 14
    python -m dexcarry.experiments.rl_grip train --workers 13 --steps 6000000 --n-steps 512 --checkpoint-every 1000000 --start 0.1 --slip-cost 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from dexcarry.envs.benchmark import DexCarryEnv
from dexcarry.eval.provenance import begin_run, end_run
from dexcarry.experiments import grip_tasks as rl_check
from dexcarry.experiments.predictors import TEST_PAIRS
from dexcarry.paths import INPUTS, ROOT, RUNS
from dexcarry.rl.grip import (
    GRIP_RATE,
    GRIPS,
    MOTORS,
    OBS_NAMES,
    Grip,
    GripEnv,
    GripPhaseEnv,
    PolicyGrip,
    SlipReflex,
    carry_outcomes,
    draw_world,
    firm_hold,
    outcome,
    run_grip,
    sweep,
)
from dexcarry.sim.transport import MotionProfile

OUT = RUNS / "ext-grip"
PER_PAIR, STATES = 2, 10
SEEDS = {"catch": 35000, "dev": 35100, "valid": 35500, "train": 36000, "smoke": 35900, "test": 32500,
         "independence": 35600, "confirm": 32600, # these two:
         "carry": 32700, "obs_range": 36100} # the carry test's reading; the training-range rollouts (train pairs)


def tasks_for(pairs: str, per_pair=PER_PAIR):
    chosen = rl_check.TRAIN_PAIRS
    if pairs == "test":
        chosen = tuple(tuple(name.split("+")) for name in TEST_PAIRS)
    return rl_check.load_tasks(INPUTS / "plan_library", pairs=chosen, per_pair=per_pair)


def states(args, tasks):
    """Yield ``(t, task, k, env)`` for each start state of this shard, seeded as in the grip tests; unbuildable ones as ``env=None``."""
    index, count = args.shard
    env = GripPhaseEnv(tasks, seed=args.seed + index)
    try:
        for t, task in list(enumerate(tasks))[index::count][:args.tasks]:
            for k in range(args.states):
                env.rng = np.random.default_rng([args.seed, t, k])
                try:
                    env.new_start(kind="second", task=task)
                except RuntimeError as error:
                    yield t, task, k, None, str(error)
                    continue
                yield t, task, k, env, None
    finally:
        env.close()


def settings_for(args):
    every = rl_check.settings(True)
    return {name: every[name] for name in args.settings}


def physics(env, setting):
    env.randomisation.scaled_gains = setting.get("scaled", True)
    return {"contact_friction": setting["contact_friction"], "dt": setting["dt"]}


REFLEX_GRID = [(s, f, t) for s in (.1, .2, .3) for f in (1.2, 1.4, 1.6) for t in (.1, .25, .5)]
# Added after the first grid's partial reading on train pairs put the best step at its edge (1.2): smaller steps, two anchors from the first grid
REFLEX_GRID2 = [(s, f, t) for s in (.1, .2, .3) for f in (1.05, 1.1) for t in (.25, .5, 1.)] + [(.3, 1.2, .25), (.1, 1.2, .25)]


def method(spec: str, top: float):
    """``reflex:START:STEP:THRESHOLD``, ``ppo:MODEL.zip`` (its start grip from the run's ``config.json``) or ``fixed:G``."""
    kind, _, rest = spec.partition(":")
    if kind == "fixed":
        return Grip(float(rest), top=float(rest))
    if kind == "reflex":
        start, step, threshold = (float(x) for x in rest.split(":"))
        return SlipReflex(start, step, threshold, top=top)
    if kind == "ppo":
        path = ROOT / rest
        model = _models.get(path)
        if model is None:
            # Loaded on first use, after the phase's physics has run, as in the recorded runs.
            import torch
            from stable_baselines3 import PPO
            torch.set_num_threads(1)
            model = _models[path] = PPO.load(path, device="cpu")
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8")) if (path.parent / "config.json").exists() else \
            json.loads((path.parent.parent / "config.json").read_text(encoding="utf-8"))
        return PolicyGrip(model, config["start"], top=top)
    raise ValueError(spec)


_models: dict = {}


def carry_profile(args):
    """the trip's ``MotionProfile`` (0.25 m/s, 0.3 m/s^2, 0.5 m/s^2 braking) over ``--carry`` metres, or None (no carry)."""
    return None if getattr(args, "carry", None) is None else MotionProfile(distance=args.carry)


def run(args):
    """Per start state and setting: the sweep (crushing ignored), then each method under each motor cap at which some grip holds.

    With ``--carry M``: the sweep stays the still phase; then each method, and the frozen
    fixed grip as a run of its own, does the phase, a drive of M metres on the trip's profile and 1 s still. ``--rerun`` runs each
    of those twice (an independence check). The gentlest grip's repeat is not run."""
    carry = carry_profile(args)
    tasks = tasks_for(args.pairs)
    grids = {"reflex-grid": REFLEX_GRID, "reflex-grid2": REFLEX_GRID2}
    specs = [f"reflex:{s:g}:{f:g}:{t:g}" for s, f, t in grids[args.methods[0]]] if args.methods[0:1] and args.methods[0] in grids else args.methods
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"grip-{args.shard[0]}of{args.shard[1]}.json"
    manifest = begin_run(args.out / "provenance", path.stem, {k: str(v) for k, v in vars(args).items()}, seeds=[args.seed],
                         files=(Path(__file__), ROOT / "dexcarry" / "rl" / "grip.py", ROOT / "dexcarry" / "rl" / "grasp_env.py"))
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    done = {(r["task"], r["state"]) for r in rows}
    settings = settings_for(args)
    for t, task, k, env, error in states(args, tasks):
        if (task.name, k) in done:
            continue
        if env is None:
            rows.append({"task": task.name, "state": k, "unbuildable": error})
            path.write_text(json.dumps(rows), encoding="utf-8")
            continue
        tick = time.perf_counter()
        for j, (name, setting) in enumerate(settings.items()):
            draw = {**physics(env, setting), **draw_world(env, setting["contact_friction"], np.random.default_rng([args.seed, t, k, j]))}
            env.carry = None
            runs = sweep(env, draw)
            row = {"task": task.name, "state": k, "setting": name, "mass": draw["mass_second"], "margin": draw["margin"],
                   "sweep": {f"{g:g}": runs[g] for g in GRIPS}, "methods": {}}
            if carry is not None:
                carried(env, args, specs, runs, row, draw, carry)
                rows.append(row)
                continue
            for cap, top in MOTORS.items():
                if cap not in args.caps or firm_hold(runs, top)[0] is None:
                    continue
                for spec in specs:
                    grip = method(spec, top).bind(env)
                    row["methods"].setdefault(spec, {})[cap] = run_grip(env, grip, draw)
            g, _ = firm_hold(runs, 1.5)
            if g is not None: # the gentlest holding grip once more, after the methods: how well a run reproduces
                row["repeat"] = run_grip(env, Grip(g, top=g).bind(env), draw)
            rows.append(row)
        path.write_text(json.dumps(rows), encoding="utf-8")
        print(task.name, k, f"{time.perf_counter() - tick:.0f}s", flush=True)
    end_run(manifest, states=len({(r["task"], r["state"]) for r in rows}))


def carried(env, args, specs, runs, row, draw, carry):
    """each method under each cap at which some constant grip holds, and the fixed grip once (it is the same run at both caps),
    through the carry; ``row["methods"]`` as ever, the fixed grip under ``fixed:G`` and cap "any"; with ``--rerun`` the same again in
    ``row["again"]``. Each run records its wall time."""
    fixed = f"fixed:{FROZEN['fixed']['1.5']:g}"
    jobs = [(spec, cap, top) for cap, top in MOTORS.items() if cap in args.caps and firm_hold(runs, top)[0] is not None for spec in specs]
    if jobs:
        jobs.append((fixed, "any", None))
    env.carry = carry
    try:
        for key in ("methods", "again") if args.rerun else ("methods",):
            row.setdefault(key, {})
            for spec, cap, top in jobs:
                grip = method(spec, top).bind(env)
                if hasattr(grip, "model"):
                    grip.observed = []
                tick = time.perf_counter()
                row[key].setdefault(spec, {})[cap] = {**run_grip(env, grip, draw), "seconds": time.perf_counter() - tick}
    finally:
        env.carry = None
    row["carry"] = {"distance": carry.distance, "speed": carry.speed, "acceleration": carry.acceleration, "braking": carry.braking,
                    "still_seconds": env.still_seconds}


def load(folder: Path) -> tuple[list[dict], int]:
    rows = [r for p in sorted(folder.glob("grip-*of*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
    return [r for r in rows if "unbuildable" not in r], sum("unbuildable" in r for r in rows)


def world(row: dict, cap: str):
    """``(g*, F, limit)`` of a phase under a motor cap, or None when no grip up to the cap holds (left out, counted)."""
    runs = {float(g): r for g, r in row["sweep"].items()}
    g, force = firm_hold(runs, MOTORS[cap])
    return None if g is None else (g, force, row["margin"] * force)


def results(rows: list[dict], cap: str, fixed: float | None = None, specs=(), told=None) -> dict:
    """Per arm, one outcome per phase in this cap's world: ``{arm: [(row, outcome, run)]}``."""
    out: dict = {}
    for r in rows:
        w = world(r, cap)
        if w is None:
            continue
        g, force, limit = w
        sweep_runs = {float(x): v for x, v in r["sweep"].items()}
        if fixed is not None:
            out.setdefault("fixed", []).append((r, outcome(sweep_runs[fixed], limit), sweep_runs[fixed]))
        for spec in specs:
            run_ = r["methods"][spec][cap]
            out.setdefault(spec, []).append((r, outcome(run_, limit), run_))
        if told is not None:
            allowed = [x for x, f in told.items() if x <= MOTORS[cap] + 1e-9 and f <= .95 * limit]
            pick = max(allowed) if allowed else min(told)
            out.setdefault("told", []).append((r, outcome(sweep_runs[pick], limit), sweep_runs[pick]))
        out.setdefault("any", []).append((r, "held" if any(outcome(v, limit) == "held" for x, v in sweep_runs.items() if x <= MOTORS[cap] + 1e-9) else "dropped", None))
    return out


# Frozen from train pairs only, before any test-pair run. None until set.
FROZEN = {"fixed": {"1.5": .7, "1.0": .7},
          "reflex": "reflex:0.1:1.2:0.25",
          "ppo": "ppo:weights/grip_policy/checkpoints/ppo_5999994_steps.zip"}


TRAINING_WORLDS = {SEEDS["dev"]: INPUTS / "grip_training" / "dev", 35200: INPUTS / "grip_training" / "dev2"} # train pairs only; their sweeps give F


def training_states(tasks, settings: list) -> list:
    """``(world_seed, t, k, [(setting, j, g*, F),...])`` for every stored train-pair start state with some holding grip."""
    index = {task.name: t for t, task in enumerate(tasks)}
    found: dict = {}
    for seed, folder in TRAINING_WORLDS.items():
        rows, _ = load(folder)
        for r in rows:
            w = world(r, "1.5")
            if w is None:
                continue
            found.setdefault((seed, index[r["task"]], r["state"]), []).append((r["setting"], settings.index(r["setting"]), w[0], w[1]))
    return [(seed, t, k, sorted(e, key=lambda x: x[1])) for (seed, t, k), e in sorted(found.items())]


def train(args):
    """PPO on train pairs only: one GripEnv per worker process (SubprocVecEnv, spawn), checkpoints and a monitor CSV."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # torch reads the thread count when it loads, so it is imported here, after setting it.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks, settings = tasks_for("train"), settings_for(args)
    DexCarryEnv().close()
    manifest = begin_run(args.out / "provenance", "train", {k: str(v) for k, v in vars(args).items()}, seeds=[SEEDS["train"] + k for k in range(args.workers)],
                         files=(Path(__file__), ROOT / "dexcarry" / "rl" / "grip.py", ROOT / "dexcarry" / "rl" / "grasp_env.py"))

    states = training_states(tasks, list(settings))
    order = np.random.default_rng(SEEDS["train"]).permutation(len(states))
    shares_ = [[states[i] for i in order[w::args.workers]] for w in range(args.workers)]

    def factory(k):
        def make():
            return GripEnv(tasks, settings, seed=SEEDS["train"] + k, start=args.start, states=shares_[k], slip_cost=args.slip_cost)
        return make
    venv = VecMonitor(SubprocVecEnv([factory(k) for k in range(args.workers)], start_method="spawn"),
                      filename=str(args.out / "monitor"), info_keywords=("success", "crushed", "held"))
    model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.n_steps * args.workers // 4, n_epochs=10, gamma=.995, gae_lambda=.95,
                learning_rate=lambda remaining: 3e-4 * remaining, clip_range=.2, ent_coef=0., target_kl=.05,
                policy_kwargs={"net_arch": [64, 64], "activation_fn": torch.nn.Tanh, "log_std_init": -1.5},
                seed=args.ppo_seed, device="cpu", verbose=0)
    config = {"start": args.start, "slip_cost": args.slip_cost, "steps": args.steps, "workers": args.workers, "n_steps": args.n_steps, "ppo_seed": args.ppo_seed,
              "env_seeds": [SEEDS["train"] + k for k in range(args.workers)], "grip_rate": GRIP_RATE, "observation": list(OBS_NAMES),
              "settings": list(settings), "tasks": [t.name for t in tasks], "training_states": len(states),
              "training_worlds": {str(seed): str(folder.relative_to(ROOT)) for seed, folder in TRAINING_WORLDS.items()},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (args.out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    saver = CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.workers), save_path=str(args.out / "checkpoints"), name_prefix="ppo")
    tick = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=saver, progress_bar=False)
    model.save(args.out / "checkpoints" / f"ppo_{model.num_timesteps}_steps.zip")
    venv.close()
    end_run(manifest, minutes=(time.perf_counter() - tick) / 60, steps=int(model.num_timesteps))
    print(f"trained {model.num_timesteps} steps in {(time.perf_counter() - tick) / 60:.0f} min", flush=True)


def carry_results(rows: list[dict], cap: str, key="methods") -> dict:
    """Per arm, one ``(row, carry_outcomes, run)`` per phase with a holding constant grip under ``cap`` (the fixed grip's one run serves
    both caps, each scored against the cap's own phases and limits)."""
    specs = {"ppo": (FROZEN["ppo"], cap), "reflex": (FROZEN["reflex"], cap), "fixed": (f"fixed:{FROZEN['fixed'][cap]:g}", "any")}
    out: dict = {arm: [] for arm in specs}
    for r in rows:
        w = world(r, cap)
        if w is None:
            continue
        for arm, (spec, c) in specs.items():
            run_ = r[key][spec][c]
            out[arm].append((r, carry_outcomes(run_, w[2]), run_))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run", "train"])
    parser.add_argument("--start", type=float, default=.1, help="train: the start grip, before the lift")
    parser.add_argument("--steps", type=int, default=4_000_000, help="train: policy decisions in all")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--ppo-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--slip-cost", type=float, default=.1, help="train: reward taken off per cm of new slip (0 for the frozen policy)")
    parser.add_argument("--pairs", choices=["train", "test"], default="train")
    parser.add_argument("--methods", nargs="+", default=[], help="reflex:START:STEP:THRESHOLD or ppo:MODEL.zip")
    parser.add_argument("--shard", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", type=int, default=None, help="at most this many plans per shard (a quick check)")
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--caps", nargs="+", default=list(MOTORS), help="run: the motor caps the methods run under")
    parser.add_argument("--carry", type=float, default=None, help="run: after the still hold, drive this many metres on the trip's profile, then 1 s still")
    parser.add_argument("--rerun", action="store_true", help="run --carry: every carry run twice (an independence check)")
    args = parser.parse_args()
    if args.action == "train":
        args.out = args.out or OUT / "ppo"
        args.settings = args.settings or ["standard", "mu0.8", "finestep"]
    else:
        args.out = args.out or OUT / "dev"
        args.seed = SEEDS["dev"] if args.seed is None else args.seed
        args.states = STATES if args.states is None else args.states
        args.settings = args.settings or ["standard", "mu0.8", "finestep"]
    {"run": run, "train": train}[args.action](args)


if __name__ == "__main__":
    main()
