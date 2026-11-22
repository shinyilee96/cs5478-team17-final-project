"""The RL grip: grips that change during the lift from finger forces and slip, and the PPO environment that trains one."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np

from ..control.grasp_controller import ClosingParams
from ..pipeline.two_object import second_phase
from ..sim.loads import rotation_between_deg
from .grasp_env import ForceLimits, GraspPhaseEnv, Randomisation

GRIPPING = ("thumb", "index", "middle")
GRIPS = (.1, .15, .2, .3, .45, .7, 1., 1.5) # the constant grips tried
MOTORS = {"1.5": 1.5, "1.0": 1.} # the top grip under each motor setting, N.m per joint
FLOOR = .1 # no grip goes below the grid's gentlest
MASS = (.08, .45)
MARGIN = (1.25, 2.5)
NARROW, WIDE = (1.25, 1.6), (2.0, 2.5) # margins reported apart
OUT_OF_REACH = 1e9 # runs record force; crushing is scored against the limit afterwards
ACTING = ("lift", "hold") # the grip changes from the first lift step on
CARRYING = ("drive", "still") # the carry after the hold (off unless ``GripPhaseEnv.carry`` is set)
SENSED = ACTING + CARRYING # every arm senses and acts at every control step from the lift to the end


class GripPhaseEnv(GraspPhaseEnv):
    """``GraspPhaseEnv`` whose draw's ``grip`` sets the gripping fingers' motor force limit at the phase's start.

    ``independent``: every ``run_rule`` runs in a fresh PyBullet client on a start state rebuilt from its seed, so
    each run starts from the same physical state whatever ran before. On one client, a run on a restored state depended on the runs
    before it: ``restoreState`` restores poses and velocities, not the contact manifolds (friction anchors, warm
    start) nor all of the engine's state. Dropping the manifolds (every body apart, then collision passes) and setting the run's friction
    before the restore removed most of it, not all. A fresh client costs about 4.4 s a run. ``independent=False`` is
    the behaviour, kept so its readings can be described as they ran."""

    def __init__(self, tasks, independent=True, **kwargs):
        kwargs.setdefault("randomisation", Randomisation(scaled_gains=True))
        kwargs.setdefault("force_limits", ForceLimits(mass=MASS))
        self.independent, self._recipe = bool(independent), None
        self.hold_peak = self.pickup = None
        super().__init__(tasks, **kwargs)

    def new_start(self, kind=None, task=None):
        """As ever, and it keeps what built the state (the stream's position, kind and task), so a run can rebuild it."""
        self._recipe = (copy.deepcopy(self.rng.bit_generator.state), kind, task)
        return super().new_start(kind, task)

    def rebuild(self):
        """A fresh client and the current start state built again from its recipe: the same state, with no history."""
        state, kind, task = self._recipe
        if self.env is not None:
            self.env.close()
        self.env, self.start = None, None
        self.rng.bit_generator.state = copy.deepcopy(state)
        self.start = self._build(kind, task)

    def run_rule(self, controller, draw: dict):
        """As ``GraspPhaseEnv.run_rule``; when ``independent``, in a fresh client on the rebuilt start state."""
        if self.independent and self._recipe is not None:
            self.rebuild()
        return super().run_rule(controller, draw)

    def _watched(self, steps, draw: dict):
        """As ever; with a carry, also the force peaks at the end of the hold (``hold_peak``, taken when the first drive point
        arrives, so after the hold's last step) and the run's ``Pickup`` (``pickup``, whose ``result`` holds the carry's record)."""
        watched = super()._watched(steps, draw)
        self.hold_peak = self.pickup = None
        return watched if self.carry is None else self._mark_hold_end(watched)

    def _mark_hold_end(self, steps):
        point = next(steps)
        try:
            while True:
                self.pickup = point["pickup"]
                if self.hold_peak is None and point["stage"] in CARRYING:
                    self.hold_peak = dict(self.watch.peak)
                point = steps.send((yield point))
        except StopIteration as stop:
            return stop.value

    def gripping(self):
        groups = self.env.hand_config["finger_groups"]
        dofs = self.env.dofs(["leap_" + j for finger in GRIPPING for j in groups[finger]])
        return dofs, self._base_forces[dofs].copy()

    def _physics(self, randomisation, draw):
        super()._physics(randomisation, draw)
        if "grip" in draw and self.start is not None and self.start.kind == "second":
            dofs, forces = self.gripping()
            self.env.max_forces[dofs] = forces * draw["grip"]


class Sense:
    """Touch and slip, as the hand would sense them. ``slip`` (cm) and ``turn`` (deg) are the object's motion in the palm since the
    first lift step (the pipeline's own reference for "held"); ``speed`` (mm) and ``spin`` (deg) are the last control step's."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.reference = self.last = None
        self.slip = self.turn = self.speed = self.spin = 0.
        self.forces = dict.fromkeys(GRIPPING, 0.)

    def __call__(self, point):
        env, body = point["env"], point["object"]
        touch = env.finger_contacts(body)
        self.forces = {f: float(touch[f]) for f in GRIPPING}
        if point["stage"] in SENSED:
            now = point["pickup"].in_palm(body)
            if self.reference is None:
                self.reference = self.last = now
            self.speed = float(np.linalg.norm(now[:3, 3] - self.last[:3, 3])) * 1000
            self.spin = float(rotation_between_deg(self.last, now))
            self.slip = float(np.linalg.norm(now[:3, 3] - self.reference[:3, 3])) * 100
            self.turn = float(rotation_between_deg(self.reference, now))
            self.last = now
        return self


class Grip:
    """A grip that may change during the lift. Subclasses give ``decide``; this one holds its start. ``params`` and ``reset`` make it a
    controller for ``GraspPhaseEnv.run_rule``, which sets the start grip from the draw's ``grip`` before the phase."""

    name = "fixed"
    params = ClosingParams()

    def __init__(self, start: float, top: float = 1.5):
        self.start, self.top, self.sense = float(start), float(top), Sense()
        self.dofs = self.forces = None
        self.reset()

    def bind(self, env: GripPhaseEnv):
        self.dofs, self.forces = env.gripping()
        return self

    def reset(self):
        self.g = self.highest = self.start
        self.sense.reset()
        self.steps_up = 0

    def decide(self, sense: Sense) -> float:
        return self.g

    def __call__(self, point):
        sense = self.sense(point)
        if point["stage"] in SENSED:
            g = float(np.clip(self.decide(sense), FLOOR, self.top))
            self.steps_up += g > self.g
            self.g, self.highest = g, max(self.highest, g)
            point["env"].max_forces[self.dofs] = self.forces * self.g
        return point["scripted"]


class CatchGrip(Grip):
    """A slip check: start at ``start``; the first time slip passes ``trigger`` mm in the palm during the lift, jump to ``to``."""

    name = "catch"

    def __init__(self, start, to, trigger=2., top=1.5):
        self.to, self.trigger = float(to), float(trigger)
        super().__init__(start, top)

    def reset(self):
        super().reset()
        self.caught_at = None

    def decide(self, sense):
        if self.caught_at is None and sense.slip * 10 > self.trigger:
            self.caught_at = sense.slip * 10
            return self.to
        return self.g


class SlipReflex(Grip):
    """The hand-written slip reflex: start gentle; while slip is sensed (the object moved more than ``threshold`` mm in
    the palm, or turned more than ``threshold`` degrees, in the last control step), raise g by the factor ``step``; else hold it."""

    name = "reflex"

    def __init__(self, start=.1, step=1.15, threshold=.3, top=1.5):
        self.step, self.threshold = float(step), float(threshold)
        super().__init__(start, top)

    def decide(self, sense):
        if sense.speed > self.threshold or sense.spin > self.threshold:
            return self.g * self.step
        return self.g


def draw_world(env: GripPhaseEnv, contact_friction: float, rng: np.random.Generator) -> dict:
    """One grasp's hidden masses (``limit_draw``'s, both objects, as drawn for the grip tests) and its margin. Limits stay out of reach in
    the run; each run is scored against margin x F afterwards."""
    limits = env.limit_draw(contact_friction, rng)
    return {"mass_first": limits["mass_first"], "mass_second": limits["mass_second"], "margin": float(rng.uniform(*MARGIN)),
            "limit_first": OUT_OF_REACH, "limit_second": OUT_OF_REACH}


def run_grip(env: GripPhaseEnv, grip: Grip, draw: dict) -> dict:
    """One phase from the current start state under ``grip``. Force is recorded with limits out of reach."""
    held_first, held_second, failure = env.run_rule(grip, {**draw, "grip": grip.start})
    second = env.env.second_id
    out = {"held": bool(held_second), "first_held": bool(held_first), "failure": failure, "peak": float(env.watch.peak[second]),
           "final_g": grip.g, "highest_g": grip.highest, "slip": grip.sense.slip, "turn": grip.sense.turn}
    if env.carry is not None:
        # "held" and "peak" are the whole run's (to the end of the carry); the hold's are kept beside them.
        result = env.pickup.result
        out["hold"] = {"held": result["hold"]["second_kept"], "first_held": result["hold"]["first_kept"],
                       "peak": float(env.hold_peak[second])}
        out["carry"] = result["carry"]
        observed = getattr(grip, "observed", None)
        if observed:
            out["obs"] = {part: [np.min(rows, 0).round(5).tolist(), np.max(rows, 0).round(5).tolist()]
                          for part, stages in (("still", ACTING), ("carry", CARRYING))
                          if len(rows := np.array([o for s, o in observed if s in stages]))}
    return out


def carry_outcomes(run: dict, limit: float) -> dict:
    """the per-run record against a crush limit: the outcome at the end of the hold and at the end of the carry (crushed over the
    whole run), lost in the carry (held at the end of the hold, not at the end), crushed in the carry (not over the limit by the end
    of the hold, over it by the end), and the wrapped object held at the end."""
    hold, end = outcome(run["hold"], limit), outcome(run, limit)
    return {"hold": hold, "end": end, "lost_in_carry": bool(run["hold"]["held"] and not run["held"]),
            "crushed_in_carry": bool(run["hold"]["peak"] <= limit < run["peak"]), "first_held": bool(run["first_held"])}


def sweep(env: GripPhaseEnv, draw: dict, grips=GRIPS, stop_at_hold=False) -> dict:
    """The constant-grip sweep, crushing ignored: ``{g: run}``. With ``stop_at_hold``, ascending until the first grip that holds."""
    out = {}
    for g in grips:
        out[g] = run_grip(env, Grip(g, top=g).bind(env), draw)
        if stop_at_hold and out[g]["held"]:
            break
    return out


def firm_hold(runs: dict, top: float):
    """The gentlest grip up to ``top`` that holds, and its peak force F; ``(None, None)`` if none holds."""
    for g in sorted(runs):
        if g <= top + 1e-9 and runs[g]["held"]:
            return g, runs[g]["peak"]
    return None, None


def outcome(run: dict, limit: float) -> str:
    """held, crushed or dropped: crushed if the peak went over the limit, whatever else happened."""
    if run["peak"] > limit:
        return "crushed"
    return "held" if run["held"] else "dropped"


OBS_NAMES = ("force_thumb", "force_index", "force_middle", "force_sum", "slip_cm", "turn_deg", "speed_mm", "spin_deg",
             "grip", "top", "stage_lift", "stage_hold", "progress", "last_action")
FORCE_SCALE = 20. # N
GRIP_RATE = .5 # one decision changes g by at most a factor exp(0.5), above the reflex grid's largest step


def observe(sense: Sense, g: float, top: float, point: dict, last_action: float) -> np.ndarray:
    """What the policy sees: touch, slip, its own grip and motor cap, the stage. Never mass, friction, F or the limit."""
    f = [sense.forces[x] / FORCE_SCALE for x in GRIPPING]
    return np.array([*f, sum(f), sense.slip, sense.turn / 10, sense.speed, sense.spin, g, top,
                     point["stage"] == "lift", point["stage"] == "hold", point["k"] / max(point["n"], 1), last_action], np.float32).clip(-10, 10)


class PolicyGrip(Grip):
    """A trained policy as a grip, for evaluation: deterministic actions, the same observation as training."""

    name = "ppo"

    def __init__(self, model, start, top=1.5, observed: list | None = None):
        self.model, self.observed = model, observed
        super().__init__(start, top)

    def reset(self):
        super().reset()
        self.last_action, self._hold_point = 0., None
        if self.observed is not None:
            self.observed.clear()

    def __call__(self, point):
        # through the carry the policy's stage input stays "hold", the last stage it was trained on, and its progress input
        # stays at the hold's last value (inside its training range); everything it senses is live.
        if point["stage"] == "hold":
            self._hold_point = {"stage": "hold", "k": point["k"], "n": point["n"]}
        self._point = self._hold_point if point["stage"] in CARRYING else point
        self._stage = point["stage"]
        return super().__call__(point)

    def decide(self, sense):
        obs = observe(sense, self.g, self.top, self._point, self.last_action)
        if self.observed is not None:
            self.observed.append((self._stage, obs))
        action, _ = self.model.predict(obs[None], deterministic=True)
        self.last_action = float(np.clip(action[0][0], -1, 1))
        return self.g * float(np.exp(GRIP_RATE * self.last_action))


class GripEnv(gym.Env):
    """PPO's environment: train pairs only. ``states`` are this worker's start states from stored train-pair worlds,
    ``(world_seed, t, k, [(setting, j, g*, F),...])``: each is rebuilt from its seed (a build follows ``reset``, which recreates the
    bodies, so it is the stored state) and serves ``per_setting`` episodes per setting, each with the stored hidden masses, a fresh
    margin and a motor cap among those at which some grip holds. No sweep runs during training: in ``SubprocVecEnv`` every worker waits
    for the slowest, so a sweep in one worker's reset stalls them all. The policy acts at every control step
    of the lift and hold; before that the grip is ``start``. Reward: +1 if the object is held and not crushed at the end; a crush ends
    the episode with nothing; 0.1 is taken off per centimetre of new slip (the slip the policy also senses)."""

    metadata = {"render_modes": []}

    def __init__(self, tasks, settings: dict, *, seed: int, start: float, states: list, per_setting=24, slip_cost=.1):
        self.phase_env = GripPhaseEnv(tasks, seed=seed)
        self.tasks, self.settings, self.rng = list(tasks), settings, np.random.default_rng([seed, 2])
        self.start, self.per_setting, self.slip_cost = float(start), int(per_setting), float(slip_cost)
        self.states, self.next_state, self.queue = list(states), 0, []
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(OBS_NAMES),), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        self.entry, self.stats = None, {"states": 0, "unbuildable": 0}

    def _new_entry(self):
        env = self.phase_env
        while not self.queue:
            world_seed, t, k, entries = self.states[self.next_state % len(self.states)]
            self.next_state += 1
            env.rng = np.random.default_rng([world_seed, t, k])
            try:
                env.new_start(kind="second", task=self.tasks[t])
            except RuntimeError:
                self.stats["unbuildable"] += 1
                continue
            self.stats["states"] += 1
            self.queue = [(world_seed, t, k, *e) for e in entries for _ in range(self.per_setting)]
        world_seed, t, k, name, j, g, force = self.queue.pop(0)
        setting = self.settings[name]
        env.randomisation.scaled_gains = setting.get("scaled", True)
        draw = {"contact_friction": setting["contact_friction"], "dt": setting["dt"],
                **draw_world(env, setting["contact_friction"], np.random.default_rng([world_seed, t, k, j]))}
        self.entry = {"draw": draw, "g": g, "force": force, "setting": name}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._new_entry()
        entry, env = self.entry, self.phase_env
        tops = [t for t in MOTORS.values() if t >= entry["g"] - 1e-9]
        self.top = float(tops[self.rng.integers(len(tops))])
        self.limit = float(self.rng.uniform(*MARGIN)) * entry["force"]
        start, sim = env.start, env.env
        sim.p.restoreState(start.state)
        sim.step_count = start.step_count
        draw = {"strength": 1., "mass_scale": 1., "shift": np.zeros(2), "turn": 0., **entry["draw"], "grip": self.start}
        env._physics(env.randomisation, draw)
        pk, context = start.pickup.snapshot(), copy.deepcopy(start.context)
        self.grip = Grip(self.start, self.top).bind(env)
        self.steps = env._watched(second_phase(pk, context, env.hold_seconds, ClosingParams()), draw)
        # The ramp and squeeze run one control step per ``step`` with the action ignored, never inside ``reset``: in a SubprocVecEnv
        # every worker waits for the slowest, so 90 steps in one worker's reset would stall all of them (found 3 Nov).
        self.point, self.last_action, self.slip_seen = next(self.steps), 0., 0.
        self.grip.sense(self.point)
        return self._obs(), {"setting": entry["setting"]}

    def _obs(self):
        return observe(self.grip.sense, self.grip.g, self.top, self.point, self.last_action)

    def step(self, action):
        grip, env = self.grip, self.phase_env
        self.last_action = 0.
        if self.point["stage"] in ACTING:
            self.last_action = float(np.clip(np.asarray(action, float).ravel()[0], -1., 1.))
            grip.g = float(np.clip(grip.g * np.exp(GRIP_RATE * self.last_action), FLOOR, self.top))
            grip.highest = max(grip.highest, grip.g)
            env.env.max_forces[grip.dofs] = grip.forces * grip.g
        second = env.env.second_id
        try:
            self.point = self.steps.send(self.point["scripted"])
        except StopIteration as stop:
            held_first, held_second, failure = env._outcome("second", stop.value)
            crushed = env.watch.peak[second] > self.limit
            ok = bool(held_second) and not crushed
            return self._obs(), float(ok), True, False, {"held": bool(held_second), "crushed": crushed, "success": ok, "failure": failure}
        if env.watch.peak[second] > self.limit:
            self.steps.close()
            return self._obs(), 0., True, False, {"held": False, "crushed": True, "success": False, "failure": "CRUSHED"}
        grip.sense(self.point)
        cost = self.slip_cost * max(0., grip.sense.slip - self.slip_seen)
        self.slip_seen = max(self.slip_seen, grip.sense.slip)
        return self._obs(), -cost, False, False, {}

    def close(self):
        self.phase_env.close()
