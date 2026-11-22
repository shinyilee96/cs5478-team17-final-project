"""Replays one recorded trial of each part of the report in PyBullet's viewer.

    python run_simulation.py grasp | ranking | perception | unseen | recovery | rl-grip | llm
    python run_simulation.py errand --planner single|greedy|failure-aware [--table round|long|scarcity] [--layout N] [--seed S]
                                    [--distance 1|2|4|8|12|16]

--speed 2 plays twice as fast and --headless skips the viewer. The viewer waits for Enter after each scene.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pybullet

from dexcarry.config import Paths
from dexcarry.envs.benchmark import DexCarryEnv
from dexcarry.envs.two_object import TwoObjectEnv
from dexcarry.experiments import errands, rl_grip, tables
from dexcarry.experiments.perception import PERCEPTION_SEED, plans
from dexcarry.experiments.pickups import OBJECTS, reach_options
from dexcarry.experiments.plan_choice import candidate_rows
from dexcarry.experiments.unseen_objects import jobs
from dexcarry.experiments.unseen_objects import run as run_unseen
from dexcarry.paths import INPUTS, RESULTS, ROOT, RUNS
from dexcarry.perception.perceive import perceive
from dexcarry.pipeline.two_object import run_two_object
from dexcarry.rl.grip import draw_world, outcome, run_grip
from dexcarry.sim.two_object import TwoObjectPlan

OUT = RUNS / "watch"
PAIR = ("tall", "ndgn-6") # a plan-choice pair with 5 candidate plans
CAMERAS = {"table": (.9, 45, -35, (.45, 0., .75))}
DISTANCES = (1., 2., 4., 8., 12., 16.) # the drive distances of the recorded errands (m)


def errand_camera(drive: float) -> tuple:
    """A side view of the errand. Up to 4 m it keeps both tables in the frame; for longer drives it starts at the pick table
    and the viewer follows the robot (``follow``)."""
    return max(2.6, .9 * min(drive, 4.)), 90, -45, (.3, .45 * min(drive, 4.), .5)


def viewer(camera: tuple, speed: float, follow: bool = False) -> None:
    """Open PyBullet's viewer for every scene this process builds, aim the camera, and play the physics at ``speed`` times real
    time; with ``follow`` the camera keeps the robot in view (your angle and zoom are kept). While the arm planners try poses
    (they set the joints directly, then put them back), drawing pauses until the next physics step. Only what is shown
    changes; the physics is the same."""
    DexCarryEnv.__init__.__kwdefaults__["gui"] = True
    distance, yaw, pitch, target = camera
    connect, step, disconnect, reset_joint = pybullet.connect, pybullet.stepSimulation, pybullet.disconnect, pybullet.resetJointState
    clock = {}

    def connect_and_aim(method, *args, **kwargs):
        client = connect(method, *args, **kwargs)
        clock.clear()
        clock["gui"] = client if method == pybullet.GUI else None # the floating-hand replays connect without a viewer, at full speed
        if clock["gui"] is not None:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=client)
            pybullet.resetDebugVisualizerCamera(distance, yaw, pitch, target, physicsClientId=client)
        return client

    def step_in_time(*args, **kwargs):
        out = step(*args, **kwargs)
        if clock.get("gui") is not None:
            show()
            if "start" not in clock:
                clock.update(start=time.perf_counter(), sim=0., dt=pybullet.getPhysicsEngineParameters()["fixedTimeStep"])
            clock["sim"] += clock["dt"]
            ahead = clock["start"] + clock["sim"] / speed - time.perf_counter()
            if ahead > .01:
                time.sleep(ahead)
            elif ahead < 0: # behind (planning between motions takes time): carry on from here rather than rush to catch up
                clock["start"] -= ahead
            clock["steps"] = clock.get("steps", 0) + 1
            if follow and clock["steps"] % 30 == 0:
                follow_robot()
        return out

    def follow_robot():
        client = clock["gui"]
        if "robot" not in clock:
            for body in range(pybullet.getNumBodies(physicsClientId=client)):
                if pybullet.getBodyInfo(body, physicsClientId=client)[1] == b"fetch_hand":
                    links = [j for j in range(pybullet.getNumJoints(body, physicsClientId=client))
                             if pybullet.getJointInfo(body, j, physicsClientId=client)[12] == b"torso_lift_link"]
                    clock["robot"] = (body, links[0]) if links else None
        if clock.get("robot"):
            x, y, _ = pybullet.getLinkState(*clock["robot"], physicsClientId=client)[0]
            yaw, pitch, zoom = pybullet.getDebugVisualizerCamera(physicsClientId=client)[8:11]
            pybullet.resetDebugVisualizerCamera(zoom, yaw, pitch, (x, y, .5), physicsClientId=client)

    def hide_and_reset(*args, **kwargs):
        if clock.get("gui") is not None and not clock.get("hidden"):
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0, physicsClientId=clock["gui"])
            clock["hidden"] = True
        return reset_joint(*args, **kwargs)

    def show():
        if clock.get("hidden"):
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1, physicsClientId=clock["gui"])
            clock["hidden"] = False

    def hold_and_disconnect(*args, **kwargs):
        client = args[0] if args else kwargs.get("physicsClientId", 0)
        if clock.get("gui") is not None and client == clock["gui"]:
            show()
            try:
                input("End of this scene. Press Enter to go on. ")
            except EOFError:
                pass
        return disconnect(*args, **kwargs)

    pybullet.connect, pybullet.stepSimulation, pybullet.disconnect = connect_and_aim, step_in_time, hold_and_disconnect
    pybullet.resetJointState = hide_and_reset


def plan_of(pair: tuple[str, str], k: int, library: Path = INPUTS / "plan_choice_library") -> TwoObjectPlan:
    stored = json.loads((library / f"plans-{'+'.join(pair)}.json").read_text(encoding="utf-8"))["plans"]
    return TwoObjectPlan.from_dict(stored[k])


def recorded(name: str, **match) -> list[dict]:
    return [r for r in tables.read_csv(RESULTS / name) if all(r[k] == str(v) for k, v in match.items())]


def held(result) -> str:
    return "both objects held" if result["both_held"] else f"failed: {result['first_failure']}"


def pickup(pair, k, seed=9100, label="") -> None:
    """One robot pickup of plan ``k`` of ``pair`` at ``seed``, as the plan-choice labels ran it."""
    rows = recorded("plan_choice_pickups.csv", pair="+".join(pair), plan=k)
    with TwoObjectEnv(OBJECTS[pair[0]], OBJECTS[pair[1]], seed=seed, first_xy=(.45, -.12), second_xy=(.45, .12)) as env:
        env.reset(seed=seed)
        result = run_two_object(env, plan_of(pair, k), seed=seed, hold_seconds=5., options=reach_options())
        row = next(r for r in rows if r["seed"] == str(seed))
        print(f"{label}plan {k}, seed {seed}: {held(result)}. Recorded: {'both objects held' if row['both_held'] == '1' else 'failed: ' + row['first_failure']};"
              f" this plan held both in {sum(r['both_held'] == '1' for r in rows)} of {len(rows)} recorded pickups.")


def grasp(args) -> None:
    first = int(next(r for r in tables.read_csv(RESULTS / "plan_choice_rankings.csv") if r["pair"] == "+".join(PAIR))["candidates"].split()[0])
    pickup(PAIR, first, label=f"{PAIR[0]} wrapped against the palm, {PAIR[1]} in a tripod grasp: ")


def ranking(args) -> None:
    """The replay ranking R orders a pair's candidate plans; its first and last choice are picked up in turn. The pair is one where
    the choice matters (R's first choice held more often than its last in 31 of the report's 47 pairs)."""
    pair = ("nbox-2", "can")
    rows = candidate_rows(INPUTS / "plan_choice_library", "+".join(pair))
    order = [r["plan"] for r in sorted(rows, key=lambda r: (r["robust_key"], r["plan"]))]
    print(f"Replay ranking R of the {len(order)} candidate plans for {'+'.join(pair)}: {' '.join(map(str, order))}")
    pickup(pair, order[0], label="R's first choice: ")
    pickup(pair, order[-1], label="R's last choice: ")


def perception(args) -> None:
    """The robot turns its head, estimates both poses from one depth image and plans the pickup on them; first with exact poses for
    comparison. The plan is the first of the perception experiment's plans that held both objects on estimated poses in its first recorded pickup."""
    seed = PERCEPTION_SEED
    rows = tables.read_csv(RESULTS / "perception_pickups.csv")
    name, k = next((n, k) for n, k in plans() if any(r["pair"] == n and r["plan"] == str(k) and r["poses"] == "estimated" and r["seed"] == str(seed)
                                                     and r["both_held"] == "1" for r in rows))
    first, second = name.split("+")
    plan = plan_of((first, second), k, INPUTS / "plan_library")
    with TwoObjectEnv(OBJECTS[first], OBJECTS[second], seed=seed) as env:
        for poses in ("exact", "estimated"):
            env.reset(seed=seed)
            env.perceived = None
            look = perceive(env, np.random.default_rng([seed, 7]))
            if poses == "estimated":
                env.perceived = look.poses
                for body, label in ((env.object_id, first), (env.second_id, second)):
                    position, yaw = look.errors.get(body, (None, None))
                    print(f"  {label}: estimated {1000 * position:.1f} mm and {yaw:.1f} deg from its true pose" if position is not None else f"  {label}: not seen")
            result = run_two_object(env, plan, seed=seed, hold_seconds=5., options=reach_options())
            env.perceived = None
            row = next(r for r in rows if r["pair"] == name and r["plan"] == str(k) and r["poses"] == poses and r["seed"] == str(seed))
            print(f"{name}, plan {k}, {poses} poses: {held(result)}. Recorded: {'both objects held' if row['both_held'] == '1' else 'failed: ' + row['first_failure']}")


def unseen(args) -> None:
    """An unseen pair: shapes are fitted to the depth image, and the primitive-shape plan library's first choice is picked up."""
    pair, out = "unseen-8+unseen-6", OUT / "unseen"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("choices.json", "mapping.json"):
        (out / name).write_bytes((INPUTS / "unseen_objects" / name).read_bytes())
    k = json.loads((out / "choices.json").read_text(encoding="utf-8"))[pair]["atlas_order"][0]
    every = jobs(Namespace(out=out))
    # the recorded runs split the jobs over 10 processes; run the one where this plan was the pair's only job
    shard = next(i for i in range(10) if [j for j in every[i::10] if j[0] == pair] == [(pair, "atlas", k)])
    for path in out.glob("label-*.json"):
        path.unlink()
    with redirect_stdout(io.StringIO()): # the experiment's own progress line
        run_unseen(Namespace(shard=(shard, 10), out=out, pickups=1, pair=pair))
    new = json.loads(next(out.glob("label-*.json")).read_text(encoding="utf-8"))["results"][0]
    rows = recorded("unseen_pickups.csv", pair=pair, plans="primitive_library", plan=k)
    before = next(r for r in rows if r["seed"] == str(new["seed"]))
    print(f"{pair}, the library's first choice (plan {k}), seed {new['seed']}: "
          f"{'both objects held' if new['both_held'] else 'failed: ' + new['first_failure']}. Recorded: "
          f"{'both objects held' if before['both_held'] == '1' else 'failed: ' + before['first_failure']}; this plan held both in "
          f"{sum(r['both_held'] == '1' for r in rows)} of {len(rows)} recorded pickups.")


def errand(family: str, config: str, layout: int, seed: int, instruction=None, distance: float = 4.) -> dict:
    """One errand in the standard setting, as the errand experiments ran it. Prints each trip and the totals."""
    job = {"family": family, "config": config, "layout": layout, "seed": seed, "setting": "standard", "distance": distance,
           **({"instruction": instruction} if instruction else {})}
    result = errands.run_job(job, OUT)
    for k, trip in enumerate(t for t in result["trips"] if "refused" not in t):
        objects = [result["names"][trip["i"]]] + ([result["names"][trip["j"]]] if trip.get("j") is not None else [])
        steps = ", ".join(s["action"] for s in trip.get("recovery") or [])
        print(f"trip {k + 1}: {' + '.join(objects)}: {trip.get('failure') or 'delivered'}" + (f"; recovery: {steps}" if steps else ""))
    print(f"{len(result['delivered'])} of {result['objects']} objects delivered in {result['seconds']:.1f} s.")
    return result


# planner and table -> errand configuration, and the columns that find its recorded row in results/errands.csv
PLANNERS = {("round", "single"): ("round-baseline", "round", "random", "single_object"),
            ("round", "greedy"): ("round-greedy", "round", "random", "greedy"),
            ("round", "failure-aware"): ("round-dexcarry", "round", "random", "dexcarry"),
            ("long", "single"): ("long-baseline", "long", "random", "single_object"),
            ("long", "failure-aware"): ("long-dexcarry", "long", "random", "dexcarry"),
            ("scarcity", "greedy"): ("scarcity-greedy", "round", "partner_scarcity", "greedy"),
            ("scarcity", "failure-aware"): ("scarcity-dexcarry", "round", "partner_scarcity", "dexcarry")}


def errand_planner(table: str, planner: str, layout: int, seed: int | None, distance: float = 4.) -> None:
    if (table, planner) not in PLANNERS:
        raise SystemExit(f"no recorded {planner} errands on the {table} table; choose from {sorted(PLANNERS)}")
    config, family, layouts, method = PLANNERS[table, planner]
    seed = seed if seed is not None else (26500 if table == "scarcity" else 26300)
    rows = recorded("errands.csv", table=family, layouts=layouts, method=method, layout=layout, seed=seed, setting="standard",
                    distance=repr(float(distance)))
    if not rows:
        raise SystemExit(f"no recorded errand at layout {layout}, seed {seed}, {distance:g} m; layouts are 0-11, seeds 26300-26301 "
                         f"(scarcity: 26500-26505); drives other than 4 m exist only for the single and failure-aware planners "
                         f"on the round and long tables")
    errand(table, config, layout, seed, distance=distance)
    row = rows[0]
    print(f"Recorded: {row['delivered']} of {row['objects']} objects delivered in {float(row['seconds']):.1f} s.")


def recovery(args) -> None:
    """Round table, layout 0: on the second trip the wrapped object is lost while the robot drives, and recovery
    still delivers the other one."""
    errand_planner("round", "failure-aware", 0, 26301)


def errand_part(args) -> None:
    errand_planner(args.table, args.planner, args.layout, args.seed, args.distance)


def grip(args) -> None:
    """The second object of plan tall-heavy+cube-s#1 (first start state, standard setting) has a crush limit and an unknown mass. The
    RL grip, the slip reflex and the fixed grip each take it at a motor cap of 1.5 N m. This phase ran first in its process when
    recorded, so it replays exactly; on it the RL grip held and the other two dropped the object."""
    plan, seed, shards = "tall-heavy+cube-s#1", 32600, 14
    tasks = rl_grip.tasks_for("test")
    index = next(t for t, task in enumerate(tasks) if task.name == plan)
    rows = recorded("grip_stationary.csv", plan=plan, state=0, setting="standard", cap_Nm="1.5")
    limit = float(rows[0]["limit_N"])
    specs = {"rl": rl_grip.FROZEN["ppo"], "reflex": rl_grip.FROZEN["reflex"], "fixed": f"fixed:{rl_grip.FROZEN['fixed']['1.5']}"}
    setting = rl_grip.settings_for(Namespace(settings=["standard"]))["standard"]
    for t, task, k, env, error in rl_grip.states(Namespace(shard=(index, shards), seed=seed, tasks=1, states=1), tasks):
        draw = {**rl_grip.physics(env, setting), **draw_world(env, setting["contact_friction"], np.random.default_rng([seed, t, k, 0]))}
        print(f"{task.name}: hidden mass {draw['mass_second']:.2f} kg, crush limit {limit:.1f} N. Each grip runs in a fresh scene "
              f"rebuilt from this start state: rl, then reflex, then fixed.")
        for name, spec in specs.items():
            run = run_grip(env, rl_grip.method(spec, 1.5).bind(env), draw)
            before = next(r for r in rows if r["method"] == name)
            print(f"  {name}: {outcome(run, limit)} (peak {run['peak']:.1f} N). Recorded: {before['outcome']} (peak {float(before['peak_N']):.1f} N)")


def llm(args) -> None:
    request = errands.instruction_row("E03")
    print(f"Request: \"{request['text']}\"")
    errand("round", "request-planner", request["layout"], 26300, instruction="E03")
    row = recorded("llm_errands.csv", request="E03", seed=26300, method="planner")[0]
    print(f"Recorded: {float(row['share']):.0%} of the requested objects delivered in {float(row['seconds']):.1f} s, "
          f"{int(float(row['violations']))} instruction violations.")


PARTS = {"grasp": grasp, "ranking": ranking, "perception": perception, "unseen": unseen, "recovery": recovery, "errand": errand_part,
         "rl-grip": grip, "llm": llm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=sorted(PARTS))
    parser.add_argument("--planner", choices=["single", "greedy", "failure-aware"], default="failure-aware", help="errand: which planner")
    parser.add_argument("--table", choices=["round", "long", "scarcity"], default="round", help="errand: which table (scarcity: partner-scarcity layouts)")
    parser.add_argument("--layout", type=int, default=0, help="errand: layout 0-11")
    parser.add_argument("--seed", type=int, default=None, help="errand: 26300 or 26301 (scarcity: 26500-26505)")
    parser.add_argument("--distance", type=float, choices=DISTANCES, default=4., help="errand: drive in m (other than 4 only for "
                        "the single and failure-aware planners on the round and long tables)")
    parser.add_argument("--speed", type=float, default=1., help="playback speed against real time")
    parser.add_argument("--headless", action="store_true", help="no viewer (prints the outcome only)")
    args = parser.parse_args()
    paths = Paths.load()
    if not (ROOT / "data" / "two_object" / "objects.json").exists() or not paths.fetch_urdf.exists() or not paths.menagerie_dir.exists():
        raise SystemExit("prepared objects (data/) or robot assets (assets/) not found; see README, Setup")
    if not args.headless:
        drive = args.distance if args.part == "errand" else 4.
        viewer(errand_camera(drive) if args.part in ("recovery", "errand", "llm") else CAMERAS["table"], args.speed, follow=drive > 4.)
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS[args.part](args)


if __name__ == "__main__":
    main()
