"""A gymnasium environment for learning finger control during the two-object pickup on the robot."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ..control.grasp_controller import STAGES, ClosingParams, ScriptedClose, drive
from ..data.catalog import ObjectSpec
from ..envs.benchmark import Scenario
from ..envs.two_object import TwoObjectEnv
from ..pipeline.two_object import BASE_STEPS, Failed, Pickup, ReachOptions, first_phase, second_approach, second_phase
from ..sim.loads import set_contact_friction
from ..sim.two_object import TwoObjectPlan

FINGERS = ("index", "middle", "ring", "thumb")
ACTIVE = {"first": ("ring",), "second": FINGERS}
FLEX_RANGE = (-.10, .30)
SIDE_RANGE = .10
FORCE_SCALE = 5. # N; forces are observed divided by this
PHASES = ("first", "second")


@dataclass(frozen=True)
class Task:
    """A plan and the two objects it was made for."""

    first: ObjectSpec
    second: ObjectSpec
    plan: TwoObjectPlan
    name: str = ""


@dataclass
class Randomisation:
    """Ranges drawn once per episode. ``contact_friction`` is drawn log-uniformly."""

    contact_friction: tuple = (.3, 1.3)
    mass_scale: tuple = (.7, 1.5)
    finger_strength: tuple = (.8, 1.2) # scales the hand's force limit and gain together
    second_shift: float = .003 # m, the second object moved on the table before the close
    second_turn_deg: float = 2.
    joint_noise: float = .005 # rad, on observed joint positions
    force_noise: tuple = (.8, 1.2) # scale on observed forces
    timesteps: tuple = (1 / 240,) # physics steps drawn from (control period stays 1/60 s)
    scaled_gains: bool = True # keep motor stiffness when the step changes (P-DT)

    @classmethod
    def none(cls):
        return cls(contact_friction=(.4, .4), mass_scale=(1., 1.), finger_strength=(1., 1.), second_shift=0.,
                   second_turn_deg=0., joint_noise=0., force_noise=(1., 1.), timesteps=(1 / 240,))


@dataclass(frozen=True)
class ForceLimits:
    """The grip tests' world. Per episode and object: a hidden mass
    drawn uniformly in ``mass``, replacing the catalogue mass and ``mass_scale``, and a limit on the summed normal force
    of the hand on the object, ``factor`` (drawn uniformly) x m g / mu with mu the episode's contact friction. A phase
    fails as "crushed" if that force, averaged over ``window`` seconds, ever exceeds the limit. PyBullet bodies are
    rigid, so "crushed" is this rule, stated with every result. Both objects have limits."""

    mass: tuple = (.08, .45)
    factor: tuple = (1.3, 3.0)
    window: float = .05
    gravity: float = 9.81


def hand_force(env, body) -> float:
    """Summed normal force (N) of every hand link (palm and fingers) on ``body``, at this instant."""
    return float(sum(c[9] for c in env.p.getContactPoints(env.robot_id, body) if c[3] in env.hand_link_indices))


class ForceWatch:
    """The moving-average hand force on each limited object over one phase, and whether a limit was exceeded."""

    def __init__(self, env, limits: dict, window_steps: int):
        self.env, self.limits, self.window = env, dict(limits), max(1, int(window_steps))
        self.recent = {body: [] for body in self.limits}
        self.peak = {body: 0. for body in self.limits}
        self.crushed: set = set()

    def record(self):
        for body, limit in self.limits.items():
            recent = self.recent[body]
            recent.append(hand_force(self.env, body))
            del recent[:-self.window]
            mean = float(np.mean(recent))
            self.peak[body] = max(self.peak[body], mean)
            if mean > limit:
                self.crushed.add(body)

    def watch(self, steps):
        """Pass a pickup generator through, recording the force after every control step."""
        try:
            point = next(steps)
            while True:
                self.record()
                target = yield point
                point = steps.send(target)
        except StopIteration as stop:
            self.record()
            return stop.value


class Residual:
    """Maps the 8-number action to a 16-joint offset of the hand target."""

    def __init__(self, cfg):
        order, groups = cfg["joint_order"], cfg["finger_groups"]
        flexion = set(cfg["flexion_joints"])
        self.flex = {f: [order.index(j) for j in groups[f] if j in flexion] for f in FINGERS}
        self.side = {f: [order.index(j) for j in groups[f] if j not in flexion] for f in FINGERS}

    def __call__(self, action, phase) -> np.ndarray:
        a = np.clip(np.asarray(action, float), -1., 1.)
        out = np.zeros(16)
        for i, finger in enumerate(FINGERS):
            if finger not in ACTIVE[phase]:
                continue
            flex = a[2 * i] * (FLEX_RANGE[1] if a[2 * i] > 0 else -FLEX_RANGE[0])
            out[self.flex[finger]] += flex
            out[self.side[finger]] += a[2 * i + 1] * SIDE_RANGE
        return out


OBS_NAMES = ([f"q{i}" for i in range(16)] + [f"target{i}" for i in range(16)] + [f"lag{i}" for i in range(16)]
             + [f"force_first_{f}" for f in FINGERS] + [f"force_second_{f}" for f in FINGERS]
             + [f"phase_{p}" for p in PHASES] + [f"stage_{s}" for s in STAGES] + ["progress"]
             + [f"plan_closed{i}" for i in range(16)] + [f"first_size{i}" for i in range(3)]
             + [f"second_size{i}" for i in range(3)])
PRIVILEGED_NAMES = ([f"first_slip{i}" for i in range(6)] + [f"second_slip{i}" for i in range(6)]
                    + ["contact_friction", "mass_scale"])


@dataclass
class _Start:
    kind: str
    state: int
    pickup: object
    context: dict | None
    step_count: int
    task: Task
    seed: int
    reference_outcome: object = None
    uses: int = 0


@dataclass
class _Episode:
    kind: str
    steps: object
    point: dict
    draw: dict
    references: dict = field(default_factory=dict)
    slip: dict = field(default_factory=dict)
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(8))
    last_target: np.ndarray | None = None


class GraspPhaseEnv(gym.Env):
    """Residual finger control during one grasp phase of a two-object pickup on the robot."""

    metadata = {"render_modes": []}
    # a ``MotionProfile`` here makes ``run_rule``'s second phase carry after the hold (``pipeline.two_object.carry_steps``).
    # None (the default) changes nothing.
    carry, still_seconds = None, 1.

    def __init__(self, tasks: list[Task], *, seed=0, second_share=.7, episodes_per_state=8, hold_seconds=2.,
                 decision_every=2, privileged=False, randomisation: Randomisation | None = None, options=None,
                 max_build_tries=20, force_limits: ForceLimits | None = None):
        if not tasks:
            raise ValueError("at least one task (plan and objects) is needed")
        self.tasks, self.rng = list(tasks), np.random.default_rng(seed)
        # Observation noise has its own random stream, so start states and physics draws do not depend on the actions:
        # a policy and the zero residual can be compared on the same episodes.
        self.noise = np.random.default_rng([seed, 1])
        self.second_share, self.per_state, self.hold_seconds = float(second_share), int(episodes_per_state), float(hold_seconds)
        self.every, self.privileged = int(decision_every), bool(privileged)
        self.randomisation = randomisation or Randomisation()
        self.max_build_tries = int(max_build_tries)
        # the grip tests' force limits. Off by default: then nothing here draws a number or changes a step.
        self.force_limits, self.watch = force_limits, None
        if options is None:
            options = ReachOptions(base_steps=BASE_STEPS, symmetry="none")
        self.options = options
        size = len(OBS_NAMES) + (len(PRIVILEGED_NAMES) if self.privileged else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (8,), np.float32)
        self.env, self.start, self.episode = None, None, None
        self.built = {"tries": 0, "refused": {}}


    def _scene(self, task: Task, seed: int):
        if self.env is None:
            self.env = TwoObjectEnv(task.first, task.second, seed=seed)
            self._base_forces = self.env.max_forces.copy()
            self._base_gains = self.env.position_gains.copy()
            self.residual = Residual(self.env.hand_config)
        elif (self.env.first_spec, self.env.second_spec) != (task.first, task.second):
            self.env.first_spec, self.env.second_spec = task.first, task.second
            self.env.scenario = Scenario.for_object(task.first, seed=seed, target_xy=self.env.scenario.target_xy)
        self._physics(Randomisation.none(), {"contact_friction": .4, "mass_scale": 1., "strength": 1., "dt": 1 / 240})
        self.env.reset(seed=seed)
        self._physics(Randomisation.none(), {"contact_friction": .4, "mass_scale": 1., "strength": 1., "dt": 1 / 240})

    def _physics(self, randomisation: Randomisation, draw: dict):
        env = self.env
        p = env.p
        hand = env.dofs(env.hand_joints)
        env.max_forces[hand] = self._base_forces[hand] * draw["strength"]
        env.position_gains[hand] = self._base_gains[hand] * draw["strength"]
        if hasattr(env, "second_id"):
            set_contact_friction(p, env.robot_id, draw["contact_friction"], env.first_spec.friction,
                                 links=sorted(env.hand_link_indices))
            # the grip tests' hidden masses replace the catalogue mass and its scale.
            p.changeDynamics(env.object_id, -1, mass=draw.get("mass_first", env.first_spec.mass * draw["mass_scale"]))
            p.changeDynamics(env.second_id, -1, mass=draw.get("mass_second", env.second_spec.mass * draw["mass_scale"]))
        dt = draw["dt"]
        p.setTimeStep(dt)
        env.substeps = max(1, round(env.dt / dt))
        env.gain_scale = dt / (1 / 240) if randomisation.scaled_gains else 1.

    def _build(self, kind=None, task=None) -> _Start:
        fixed_kind, fixed_task = kind, task
        for _ in range(self.max_build_tries):
            self.built["tries"] += 1
            kind = fixed_kind or ("second" if self.rng.random() < self.second_share else "first")
            task = fixed_task or self.tasks[self.rng.integers(len(self.tasks))]
            seed = int(self.rng.integers(1 << 30))
            self._scene(task, seed)
            pk = Pickup(self.env, task.plan, seed, self.options)
            failure = pk.prepare()
            context = None
            if not failure and kind == "second":
                context = drive(first_phase(pk, ClosingParams()), ScriptedClose())
                if isinstance(context, Failed):
                    failure = context[0]["first_failure"]
                else:
                    context = second_approach(pk, context)
                    if isinstance(context, Failed):
                        failure = context[0]["first_failure"]
            if failure:
                self.built["refused"][failure] = self.built["refused"].get(failure, 0) + 1
                continue
            state = self.env.p.saveState()
            start = _Start(kind, state, pk.snapshot(), copy.deepcopy(context), self.env.step_count, task, seed)
            # What the scripted rule does from here, once, without restoring: the zero-residual check.
            steps = (second_phase(pk, context, self.hold_seconds, ClosingParams()) if kind == "second"
                     else first_phase(pk, ClosingParams()))
            start.reference_outcome = self._outcome(kind, drive(steps, ScriptedClose()))
            return start
        raise RuntimeError(f"no start state after {self.max_build_tries} tries: {self.built['refused']}")

    @staticmethod
    def _outcome(kind, value):
        """``(held_first, held_second, failure)`` of a finished phase."""
        if kind == "first":
            if isinstance(value, Failed):
                return False, False, value[0]["first_failure"]
            return True, False, None
        result = value[0]
        return bool(result.get("first_kept")), bool(result.get("second_kept")), result["first_failure"]

    def new_start(self, kind=None, task=None) -> _Start:
        """Replace the current start state with a new one, of this kind and task if given."""
        if self.start is not None:
            self.env.p.removeState(self.start.state)
            self.start = None
        self.start = self._build(kind, task)
        return self.start

    def run_rule(self, controller, draw: dict):
        """Run the current start state's phase once under a closing rule (``control.grasp_controller``)
        and the physics of ``draw`` (keys as ``_draw`` returns). Returns ``(held_first, held_second,
        failure)``. The start state is left as it was. With force limits, ``draw`` must hold the hidden masses and
        limits (``limit_draw``), so that rules compared on one state share them."""
        start, env = self.start, self.env
        env.p.restoreState(start.state)
        env.step_count = start.step_count
        draw = {"strength": 1., "mass_scale": 1., "shift": np.zeros(2), "turn": 0., **draw}
        self._physics(self.randomisation, draw)
        pk, context = start.pickup.snapshot(), copy.deepcopy(start.context)
        controller.reset()
        steps = (second_phase(pk, context, self.hold_seconds, controller.params, carry=self.carry, still_seconds=self.still_seconds)
                 if start.kind == "second" else first_phase(pk, controller.params))
        return self._crush(self._outcome(start.kind, drive(self._watched(steps, draw), controller)))


    def limit_draw(self, contact_friction: float, rng=None) -> dict:
        """Hidden masses and force limits of one episode, for both objects (empty when force limits are off)."""
        fl, rng = self.force_limits, rng if rng is not None else self.rng
        if fl is None:
            return {}
        out = {}
        for name in ("first", "second"):
            mass, factor = float(rng.uniform(*fl.mass)), float(rng.uniform(*fl.factor))
            out[f"mass_{name}"], out[f"limit_{name}"] = mass, factor * mass * fl.gravity / contact_friction
        return out

    def _watched(self, steps, draw: dict):
        """With force limits, a ``ForceWatch`` over both objects for this phase; otherwise ``steps`` unchanged."""
        if self.force_limits is None:
            self.watch = None
            return steps
        if "limit_first" not in draw:
            raise ValueError("force limits are on: the draw needs the hidden masses and limits (limit_draw)")
        env = self.env
        self.watch = ForceWatch(env, {env.object_id: draw["limit_first"], env.second_id: draw["limit_second"]},
                                round(self.force_limits.window / env.dt))
        return self.watch.watch(steps)

    def _crush(self, outcome):
        """A crushed object is not held; the phase fails with the pipeline's code if it has one, else CRUSHED."""
        if self.watch is None or not self.watch.crushed:
            return outcome
        held_first, held_second, failure = outcome
        first, second = self.env.object_id in self.watch.crushed, self.env.second_id in self.watch.crushed
        code = failure or ("FIRST_CRUSHED" if first else "SECOND_CRUSHED")
        return held_first and not first, held_second and not second, code

    def _draw(self) -> dict:
        r, rng = self.randomisation, self.rng
        low, high = np.log(r.contact_friction[0]), np.log(r.contact_friction[1])
        return {"contact_friction": float(np.exp(rng.uniform(low, high))), "mass_scale": float(rng.uniform(*r.mass_scale)),
                "strength": float(rng.uniform(*r.finger_strength)), "dt": float(r.timesteps[rng.integers(len(r.timesteps))]),
                "shift": rng.uniform(-r.second_shift, r.second_shift, 2), "turn": float(np.radians(rng.uniform(-1, 1) * r.second_turn_deg)),
                "force_scale": float(rng.uniform(*r.force_noise))}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng, self.noise = np.random.default_rng(seed), np.random.default_rng([seed, 1])
        if self.start is None or self.start.uses >= self.per_state:
            if self.start is not None:
                self.env.p.removeState(self.start.state)
            self.start = self._build()
        start, env = self.start, self.env
        start.uses += 1
        env.p.restoreState(start.state)
        env.step_count = start.step_count
        draw = self._draw() if not (options or {}).get("no_randomisation") else {
            "contact_friction": .4, "mass_scale": 1., "strength": 1., "dt": 1 / 240, "shift": np.zeros(2), "turn": 0.,
            "force_scale": 1.}
        draw.update(self.limit_draw(draw["contact_friction"]))
        self._physics(self.randomisation, draw)
        if start.kind == "second" and (np.any(draw["shift"]) or draw["turn"]):
            pos, quat = env.p.getBasePositionAndOrientation(env.second_id)
            turned = (Rotation.from_euler("z", draw["turn"]) * Rotation.from_quat(quat)).as_quat()
            env.p.resetBasePositionAndOrientation(env.second_id, [pos[0] + draw["shift"][0], pos[1] + draw["shift"][1], pos[2]], turned)
        pk = start.pickup.snapshot()
        context = copy.deepcopy(start.context)
        steps = self._watched(second_phase(pk, context, self.hold_seconds, ClosingParams()) if start.kind == "second"
                              else first_phase(pk, ClosingParams()), draw)
        self.episode = _Episode(start.kind, steps, next(steps), draw)
        self.episode.last_target = np.asarray(self.episode.point["scripted"], float)
        return self._observe(), {"kind": start.kind, "task": start.task.name, "draw": {k: v for k, v in draw.items() if k != "shift"}}

    def step(self, action):
        ep = self.episode
        action = np.clip(np.asarray(action, float), -1., 1.)
        reward = -.01 * float(np.mean(action ** 2)) - .01 * float(np.mean((action - ep.last_action) ** 2))
        ep.last_action = action
        done, value = False, None
        for _ in range(self.every):
            point = ep.point
            lo, hi = point["limits"]
            target = np.clip(np.asarray(point["scripted"], float) + self.residual(action, point["phase"]), lo, hi)
            ep.last_target = target
            try:
                ep.point = ep.steps.send(target)
            except StopIteration as stop:
                done, value = True, stop.value
                break
            reward -= self._slip_cost()
        info = {"kind": ep.kind}
        if done:
            held_first, held_second, failure = self._crush(self._outcome(ep.kind, value))
            if self.watch is not None:
                info.update(crushed=sorted(self.watch.crushed), peak_force=dict(self.watch.peak))
            reward += (held_first if ep.kind == "first" else held_first + held_second + (held_first and held_second))
            info.update(held_first=held_first, held_second=held_second, failure=failure,
                        reference=self.start.reference_outcome, task=self.start.task.name)
        return self._observe(), float(reward), done, False, info

    def _slip_cost(self) -> float:
        """0.1 per centimetre of new slip in the palm, from the lift on (privileged; reward only)."""
        ep, env = self.episode, self.env
        point = ep.point
        if point["stage"] in ("ramp", "squeeze"):
            return 0.
        pk, cost = point["pickup"], 0.
        bodies = [env.object_id] + ([env.second_id] if point["phase"] == "second" else [])
        for body in bodies:
            now = pk.in_palm(body)
            reference = ep.references.setdefault((point["phase"], body), now)
            slip = float(np.linalg.norm(now[:3, 3] - reference[:3, 3]))
            cost += .1 * max(0., slip - ep.slip.get(body, 0.)) * 100
            ep.slip[body] = max(ep.slip.get(body, 0.), slip)
        return cost

    def _observe(self) -> np.ndarray:
        ep, env = self.episode, self.env
        point, draw, r = ep.point, ep.draw, self.randomisation
        q = env.hand_joint_values() + self.noise.normal(0., r.joint_noise, 16)
        target = ep.last_target
        forces = [env.finger_contacts(body) for body in (env.object_id, env.second_id)]
        plan = self.start.task.plan
        phase = np.array([point["phase"] == p for p in PHASES], float)
        stage = np.array([point["stage"] == s for s in STAGES], float)
        progress = point["k"] / max(point["n"], 1)
        obs = [q, target, target - q,
               [min(f[x] * draw["force_scale"] / FORCE_SCALE, 5.) for f in forces for x in FINGERS],
               phase, stage, [progress], np.asarray(plan.second_closed, float),
               np.asarray(self.start.task.first.half_extents) * 10, np.asarray(self.start.task.second.half_extents) * 10]
        if self.privileged:
            pk = point["pickup"]
            for body in (env.object_id, env.second_id):
                now = pk.in_palm(body)
                reference = ep.references.get((point["phase"], body), now)
                obs.append(np.r_[(now[:3, 3] - reference[:3, 3]) * 100,
                                 Rotation.from_matrix(reference[:3, :3].T @ now[:3, :3]).as_rotvec()])
            obs.append([draw["contact_friction"], draw["mass_scale"]])
        return np.concatenate([np.asarray(o, float).ravel() for o in obs]).astype(np.float32)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
