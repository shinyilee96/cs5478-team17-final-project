"""Two objects in one hand on the Fetch robot: side wrap, three-finger grasp, carry and set down, executing plans
found on the floating hand.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..control.grasp_controller import ClosingParams, ScriptedClose, drive
from ..envs.fetch import FETCH_BASE_JOINTS
from ..grasp.grasp_vector import invert_transform, make_transform, matrix_to_quat_xyzw
from ..kin.collision import PyBulletCollisionChecker
from ..kin.ik import DLSIK
from ..kin.rrt import rrt_connect, shortcut
from ..sim.closing import ContactStopClose
from ..sim.loads import SLIP_ROTATION_DEG, SLIP_TRANSLATION, rotation_between_deg
from ..sim.transport import MotionProfile, base_trajectory, return_route
from ..sim.two_object import LIFT, SETTLE_SECONDS, TwoObjectPlan
from .common import ARM_AND_TORSO, ArmExecutor
from .execution import APPROACH_DISPLACEMENT_LIMIT, STANDING_SPEED
from .feasibility import _drive_clear
from .postures import posture_candidates

# First failure of a two-object pickup on the robot, in the order the stages run. The codes the
# floating harness shares keep their names; the others are what an arm adds.
FAILURES = ("FIRST_UNREACHABLE", "FIRST_APPROACH_COLLISION", "FIRST_DISTURBED", "FIRST_NO_CONTACT", "FIRST_NOT_LIFTED",
            "FIRST_OTHER_FINGER", "SECOND_UNREACHABLE", "SECOND_PATH_COLLISION", "FIRST_LOST_MOVING", "SECOND_PUSHED",
            "SECOND_NOT_LIFTED", "FIRST_LOST_ON_SECOND_GRASP", "BOTH_LOST")
# Failures of the planner alone: known before the arm makes the motion, so they cost no object.
PLANNER_FAILURES = ("FIRST_UNREACHABLE", "FIRST_APPROACH_COLLISION", "SECOND_UNREACHABLE", "SECOND_PATH_COLLISION",
                    "CARRY_BLOCKED", "SECOND_PLACE_UNREACHABLE", "SECOND_PLACE_COLLISION", "FIRST_PLACE_UNREACHABLE",
                    "FIRST_PLACE_COLLISION")
# Where the hand waits before the straight move onto a grasp pose: (metres along the back of
# the hand, metres up), tried in order.
FIRST_RETREATS = ((.08, 0.), (0., .10))
SECOND_RETREATS = ((0., .12), (.08, 0.))
MORE_SECOND_RETREATS = (*SECOND_RETREATS, (.05, .08), (.12, 0.), (0., .18), (.10, .10))
# Base moves (metres forward, metres to the left) tried between the grasps when the second object
# cannot be reached from where the base stood for the first. The second object stands to the left.
BASE_STEPS = ((0., .12), (0., .24), (.10, 0.), (-.10, 0.), (.10, .24), (-.10, .24))
# Carry posture that leaves the arm where the second lift ended, so gravity keeps the direction
# in the palm that both grasps have just held.
HOLD = "hold"
# The object is released a little above the table; if the hand would touch the table there, higher.
PLACE_RAISES = (0., .01, .02, .03)
RELEASE_SECONDS = .5
UPRIGHT_DEG = 20.
# the any-pair counter: the drive between the grasps, with ``ReachOptions(step_to_second=True)``. Each option's own step
# puts the second object where the plan library's scene has it, ``LIBRARY_SECOND_Y`` to the base's left; the hand first rises so
# that it and the held object clear every standing object by ``STEP_CLEARANCE``, the base drives at the carry's speed
# (``MotionProfile``'s 0.25 m/s), then the arm comes down onto the second. The drive is collision-checked every ``STEP_CHECK``.
LIBRARY_SECOND_Y = .12
STEP_CLEARANCE = .05
STEP_CHECK = .02
STEP_RISE_SPEED = .1 # m/s of the hand's rise (as ``back_to_lift``'s 0.10 m in 1 s)
STEP_SETTLE_STEPS = 30 # held still after the drive, as ``step_base``
# the round table: with ``ReachOptions.step_forward`` the step also puts the second object as far ahead of the base
# as the plan library's scene has it: the slots' x (0.45 m) from the start pose's x (-0.35 m).
LIBRARY_AHEAD = .80


@dataclass(frozen=True)
class ReachOptions:
    """How hard the planner tries for the arm motions of a two-object pickup."""

    rrt_iters: int = 500
    second_retreats: tuple = SECOND_RETREATS
    # Base moves tried, in order, after the second reach has failed from where the base stands.
    base_steps: tuple = ()
    # Whose box symmetry may turn a plan about the vertical: "both", "second" or "none".
    symmetry: str = "both"
    # the any-pair counter: the base steps are measured from where the second object stands (each option's own step), and a
    # step is made with the hand raised clear of standing objects, at the carry's speed. False is every earlier run's.
    step_to_second: bool = False
    # the round table: with ``step_to_second``, the step also moves the base along its heading so that the second stands
    # ``LIBRARY_AHEAD`` in front of it, tried before the sideways-only steps. False is every earlier run's.
    step_forward: bool = False

    def __post_init__(self):
        if self.symmetry not in ("both", "second", "none"):
            raise ValueError("symmetry is 'both', 'second' or 'none'")


def symmetries(spec, allowed=True) -> list[np.ndarray]:
    """Turns about the vertical that map a standing box onto itself; a mesh object has only the identity."""
    if spec.visual_mesh or not allowed:
        return [np.eye(4)]
    square = np.isclose(spec.half_extents[0], spec.half_extents[1])
    return [make_transform(Rotation.from_euler("z", angle, degrees=True).as_matrix(), np.zeros(3))
            for angle in ((0, 90, 180, 270) if square else (0, 180))]


def _touched(env, checker) -> str:
    """What the checker's last collision involved, then the robot link and the depth, for diagnostics."""
    if checker.last_collision is None:
        return "unknown"
    kind, a, b, depth = checker.last_collision
    names = {env.robot_id: "robot", env.table_id: "pick_table", env.target_table_id: "target_table",
             env.plane_id: "floor", env.object_id: "first", getattr(env, "second_id", None): "second"}
    if kind == "held":
        return f"held_{names.get(a, '?')}-{names.get(b, '?')} {depth * 1000:.1f}mm"
    link = next((name for name, index in env.link_names.items() if index == b), "?")
    return f"{'hand' if b in env.hand_link_indices else 'arm'}-{names.get(a, '?')} {link} {depth * 1000:.1f}mm"


def _reach(env, T, q_start, rng, retreats, *, touch=(), attached=None, rrt_iters=500):
    """Arm motion onto palm pose ``T``: a contact-free transit to a backed-off pose, then straight in.

    Only the straight move may bring the hand into contact with the bodies in ``touch``.
    Returns ``(status, plan, reason)`` where status is None, "UNREACHABLE" or "COLLISION" and
    reason says which step refused.
    """
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    goal = ik.solve_multi(T[:3, 3], T[:3, :3], [q_start, ik.rest, rng.uniform(ik.lower, ik.upper), rng.uniform(ik.lower, ik.upper)])
    lift = ik.solve(T[:3, 3] + [0, 0, LIFT], T[:3, :3], q0=goal.q)
    if not goal.success or not lift.success:
        return "UNREACHABLE", None, "IK_GOAL" if not goal.success else "IK_LIFT"
    transit = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=attached)
    final = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=attached)
    for body in touch:
        final.allowed_contacts[body] = env.hand_link_indices
    if final(goal.q):
        return "COLLISION", None, "GOAL:" + _touched(env, final)
    reason = "NO_APPROACH"
    for back, up in retreats:
        pre = ik.solve(T[:3, 3] + back * T[:3, 2] + [0, 0, up], T[:3, :3], q0=goal.q)
        if not pre.success:
            continue
        approach = [pre.q + a * (goal.q - pre.q) for a in np.linspace(0, 1, 40)]
        if any(final(q) for q in approach):
            continue
        reason = "NO_PATH"
        path = rrt_connect(q_start, pre.q, transit, ik.lower, ik.upper, max_iters=rrt_iters, rng=rng)
        if path is None:
            continue
        path = shortcut(path, transit, n_iters=60, rng=rng)
        return None, {"path": path + approach, "q_goal": goal.q, "q_lift": lift.q, "retreat": (back, up)}, None
    return "COLLISION", None, reason


def perceive_for_planning(env, bodies) -> None:
    """Estimated poses: if ``env.perceived`` holds ``{body: 4x4 world pose}``, put those of ``bodies`` there, so
    that planning (targets and collision checks) sees the scene as the robot estimated it. Only the planners call it,
    between their own save and restore of the simulator state, so the physics always runs on the true poses. Without
    estimates (the default) nothing changes.

    Observation-only planning: if ``env.observed`` holds a ``pipeline.observed.ObservedPlanning``, the planners see
    fitted-primitive proxies instead of the bodies (``bodies`` standing where the look saw them); the planners give the
    bodies back with ``_leave_observed`` before their own restore."""
    poses = getattr(env, "perceived", None)
    for body in bodies:
        if poses and body in poses:
            env.p.resetBasePositionAndOrientation(body, poses[body][:3, 3], matrix_to_quat_xyzw(poses[body][:3, :3]))
    observed = getattr(env, "observed", None)
    if observed is not None:
        observed.enter(env, bodies)


def _leave_observed(env) -> None:
    """End observation-only planning's view; nothing without ``env.observed``."""
    observed = getattr(env, "observed", None)
    if observed is not None:
        observed.leave(env)


def plan_first(env, plan: TwoObjectPlan, seed=0, options: ReachOptions | None = None):
    """Standing pose and arm motion for the side wrap of the first object; nothing is executed.

    Tries each standing offset and each symmetric turn of the plan about the object. Returns
    ``(failure, plan)``; of several failures the one that got furthest is reported.
    """
    options = options or ReachOptions()
    p, rng = env.p, np.random.default_rng(seed)
    lo, hi = env.hand_joint_limits()
    state = p.saveState()
    try:
        start = env.get_base_pose()
        heading = np.array([np.cos(start[2]), np.sin(start[2]), 0.])
        worst = "UNREACHABLE"
        for offset in env.scenario.standing_offsets:
            p.restoreState(state)
            perceive_for_planning(env, (env.object_id, env.second_id))
            base = start + offset * heading
            if offset and not _drive_clear(env, start, base):
                continue
            env.set_joints(dict(zip(FETCH_BASE_JOINTS, base, strict=True)))
            env.set_joints(dict(zip(env.hand_joints, np.clip(plan.first_pre, lo, hi), strict=True)))
            q_now = env.get_joints(ARM_AND_TORSO)
            for turn in symmetries(env.first_spec, options.symmetry == "both"):
                T = env.get_object_transform() @ turn @ np.asarray(plan.first_pose, float)
                status, reach, _ = _reach(env, T, q_now, rng, FIRST_RETREATS, touch=(env.object_id,),
                                          rrt_iters=options.rrt_iters)
                if status is None:
                    return None, {**reach, "base": base, "standing_offset": float(offset), "turn": turn}
                if status == "COLLISION":
                    worst = status
        return "FIRST_" + ("APPROACH_COLLISION" if worst == "COLLISION" else worst), None
    finally:
        _leave_observed(env)
        p.restoreState(state)
        p.removeState(state)


def _plan_second(env, plan: TwoObjectPlan, rng, options: ReachOptions):
    """Arm motion from the first lift onto the second grasp, with the first object attached as held.

    Returns ``(failure, plan, reasons)``. The base stays where it is unless that fails and
    ``options.base_steps`` offers moves; a move must be clear with the first object in hand.
    ``options.step_to_second`` (the any-pair counter) plans the steps with ``_plan_second_stepped`` instead.
    """
    if options.step_to_second:
        return _plan_second_stepped(env, plan, rng, options)
    p = env.p
    state = p.saveState()
    try:
        here = env.get_base_pose()
        heading, left = np.array([np.cos(here[2]), np.sin(here[2]), 0.]), np.array([-np.sin(here[2]), np.cos(here[2]), 0.])
        q_arm = env.get_joints(ARM_AND_TORSO)
        worst, reasons = "UNREACHABLE", []
        for forward, sideways in ((0., 0.), *options.base_steps):
            p.restoreState(state)
            # The first object is in the hand by now; only the second is still where perception put it.
            perceive_for_planning(env, (env.second_id,))
            base = here + forward * heading + sideways * left
            if forward or sideways:
                drive = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS, attached_body=env.object_id)
                # The last pose checked leaves the base and the held object where the reach starts.
                if any(drive(here + a * (base - here)) for a in np.linspace(0, 1, 8)[1:]):
                    reasons.append("BASE_BLOCKED")
                    continue
            for turn in symmetries(env.second_spec, options.symmetry != "none"):
                T = env.body_transform(env.second_id) @ turn @ np.asarray(plan.second_pose, float)
                status, reach, reason = _reach(env, T, q_arm, rng, options.second_retreats, touch=(env.second_id,),
                                               attached=env.object_id, rrt_iters=options.rrt_iters)
                if status is None:
                    return None, {**reach, "base": base, "base_step": (float(forward), float(sideways))}, reasons
                reasons.append(reason)
                if status == "COLLISION":
                    worst = status
        return "SECOND_" + ("PATH_COLLISION" if worst == "COLLISION" else worst), None, reasons
    finally:
        _leave_observed(env)
        p.restoreState(state)
        p.removeState(state)


def step_candidates(env, here, steps, forward_too: bool = False) -> tuple[list[tuple[float, float]], float]:
    """the any-pair counter: the base moves (metres forward, metres to the left) tried between the grasps, in order: none; each of ``steps``
    shifted by the option's own step, which puts the second object where the plan library's scene has it (``LIBRARY_SECOND_Y``
    to the base's left, from the second's pose as planning sees it); then ``steps`` as they are. Duplicates dropped. Also
    returns the shift (metres to the left). the round table (``forward_too``): first of all after none, the same shifts with the base also
    moved along its heading so that the second stands ``LIBRARY_AHEAD`` in front of it."""
    left = np.array([-np.sin(here[2]), np.cos(here[2])])
    shift = float((env.body_transform(env.second_id)[:2, 3] - np.asarray(here[:2], float)) @ left) - LIBRARY_SECOND_Y
    ahead = ()
    if forward_too:
        heading = np.array([np.cos(here[2]), np.sin(here[2])])
        forward = float((env.body_transform(env.second_id)[:2, 3] - np.asarray(here[:2], float)) @ heading) - LIBRARY_AHEAD
        ahead = tuple((f + forward, s + shift) for f, s in ((0., 0.), *steps))
    out: list[tuple[float, float]] = []
    for forward, sideways in ((0., 0.), *ahead, *((f, s + shift) for f, s in ((0., 0.), *steps)), *steps):
        move = (float(forward), float(sideways))
        if all(abs(move[0] - f) > 1e-3 or abs(move[1] - s) > 1e-3 for f, s in out):
            out.append(move)
    return out, shift


def clear_rise(env, held) -> float:
    """the any-pair counter: how far the hand must rise so that it and ``held`` clear the tops of the scene's other objects (``env.bodies``,
    else the two of a two-object scene) by ``STEP_CLEARANCE``. Their collision poses are as planning sees them."""
    p = env.p
    others = [b for b in (getattr(env, "bodies", None) or (env.object_id, env.second_id)) if b != held]
    top = max(p.getAABB(b)[1][2] for b in others)
    low = min([p.getAABB(held)[0][2], *(p.getAABB(env.robot_id, link)[0][2] for link in env.hand_link_indices)])
    return max(0., float(top + STEP_CLEARANCE - low))


def _plan_second_stepped(env, plan: TwoObjectPlan, rng, options: ReachOptions):
    """``_plan_second`` for the any-pair counter: each option's own step (``step_candidates``). A move is made with the hand
    raised: first the hand rises straight up by ``clear_rise`` (a straight joint move, checked with the first object in hand),
    then the base drives (checked every ``STEP_CHECK``), then the arm comes down onto the second from the raised pose. Returns
    ``(failure, plan, reasons)`` as ``_plan_second``; a moving plan also holds ``q_high``, ``rise`` and ``shift``."""
    p = env.p
    state = p.saveState()
    try:
        here = env.get_base_pose()
        heading, left = np.array([np.cos(here[2]), np.sin(here[2]), 0.]), np.array([-np.sin(here[2]), np.cos(here[2]), 0.])
        q_arm = env.get_joints(ARM_AND_TORSO)
        perceive_for_planning(env, (env.second_id,))
        moves, shift = step_candidates(env, here, options.base_steps, options.step_forward)
        worst, reasons, high = "UNREACHABLE", [], None
        for forward, sideways in moves:
            p.restoreState(state)
            # The first object is in the hand by now; only the second is still where perception put it.
            perceive_for_planning(env, (env.second_id,))
            base, q_start, rise = here + forward * heading + sideways * left, q_arm, 0.
            if forward or sideways:
                if high is None:
                    high = _raised(env, q_arm)
                if isinstance(high, str):
                    reasons.append(high)
                    continue
                q_start, rise = high
                arm = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=env.object_id)
                arm(q_start) # leaves the arm raised, the first object with it
                drive = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS, attached_body=env.object_id)
                n = max(2, int(np.ceil(np.linalg.norm(base[:2] - here[:2]) / STEP_CHECK)) + 1)
                # The last pose checked leaves the base and the held object where the reach starts.
                if any(drive(here + a * (base - here)) for a in np.linspace(0, 1, n)[1:]):
                    reasons.append("BASE_BLOCKED")
                    continue
            for turn in symmetries(env.second_spec, options.symmetry != "none"):
                T = env.body_transform(env.second_id) @ turn @ np.asarray(plan.second_pose, float)
                status, reach, reason = _reach(env, T, q_start, rng, options.second_retreats, touch=(env.second_id,),
                                               attached=env.object_id, rrt_iters=options.rrt_iters)
                if status is None:
                    moved = bool(forward or sideways)
                    return None, {**reach, "base": base, "base_step": (float(forward), float(sideways)), "shift": shift,
                                  "q_high": q_start if moved else None, "rise": rise}, reasons
                reasons.append(reason)
                if status == "COLLISION":
                    worst = status
        return "SECOND_" + ("PATH_COLLISION" if worst == "COLLISION" else worst), None, reasons
    finally:
        _leave_observed(env)
        p.restoreState(state)
        p.removeState(state)


def _raised(env, q_arm):
    """The arm with the hand ``clear_rise`` straight up, palm turned as it is, and the rise; or the reason it cannot."""
    rise = clear_rise(env, env.object_id)
    if rise < 1e-3:
        return q_arm, 0.
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    palm = env.get_link_transform(env.hand_root_link)
    up = ik.solve(palm[:3, 3] + [0, 0, rise], palm[:3, :3], q0=q_arm)
    if not up.success:
        return "RISE_UNREACHABLE"
    check = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=env.object_id)
    if any(check(q_arm + a * (up.q - q_arm)) for a in np.linspace(0, 1, 20)[1:]):
        return "RISE_BLOCKED"
    return up.q, rise


def dry_run(env, plan: TwoObjectPlan, seed=0, options: ReachOptions | None = None) -> str | None:
    """Would the planner accept both arm motions of the pickup in this scene? Nothing is executed.

    The first motion is planned as in a trial. For the second, the arm is set where the first lift
    would end and the first object is put in the hand where it settled when the search lifted it
    on the floating hand (``plan.first_held``; without it, where the plan grasps it, a few
    centimetres further out). In a trial it settles where it settles, so this is an estimate.
    Returns the planner's failure or None. It uses the true object poses, as execution does, but
    no outcome of any grasp on the robot."""
    options = options or ReachOptions()
    failure, first = plan_first(env, plan, seed, options)
    if failure:
        return failure
    p = env.p
    lo, hi = env.hand_joint_limits()
    order, groups = env.hand_config["joint_order"], env.hand_config["finger_groups"]
    q_hand = np.clip(np.asarray(plan.first_closed, float), lo, hi)
    for finger in plan.second_fingers:
        idx = [order.index(j) for j in groups[finger]]
        q_hand[idx] = np.clip(np.asarray(plan.second_pre, float), lo, hi)[idx]
    state = p.saveState()
    try:
        env.set_joints(dict(zip(FETCH_BASE_JOINTS, first["base"], strict=True)))
        env.set_joints(dict(zip(ARM_AND_TORSO, first["q_lift"], strict=True)))
        env.set_joints(dict(zip(env.hand_joints, q_hand, strict=True)))
        in_palm = (invert_transform(first["turn"] @ np.asarray(plan.first_pose, float)) if plan.first_held is None
                   else np.asarray(plan.first_held, float))
        held = env.get_link_transform(env.hand_root_link) @ in_palm
        p.resetBasePositionAndOrientation(env.object_id, held[:3, 3], matrix_to_quat_xyzw(held[:3, :3]))
        return _plan_second(env, plan, np.random.default_rng([seed, 1]), options)[0]
    finally:
        p.restoreState(state)
        p.removeState(state)


def posed(env, command, hand_q):
    """The action: every joint at its target in ``command``, the hand at ``hand_q``."""
    # One commanded target per joint (see execute_plan): the arm must not be re-commanded to its measured pose.
    return env.action_with_hand(hand_q, command)


def in_palm(env, body):
    """Pose of ``body`` in the palm frame."""
    return invert_transform(env.get_link_transform(env.hand_root_link)) @ env.body_transform(body)


def step_base(env, command, goal, hand_q):
    """Move the base to ``goal`` at walking pace, the arm as ``command`` holds it; ``command`` is updated."""
    base = env.dofs(FETCH_BASE_JOINTS)
    here = command[base].copy()
    n = int(np.linalg.norm(goal[:2] - here[:2]) / (STANDING_SPEED * env.dt))
    for k in range(1, n + 1):
        command[base] = here + (goal - here) * k / n
        env.step(posed(env, command, hand_q))
    for _ in range(30 if n else 0):
        env.step(posed(env, command, hand_q))


class Pickup:
    """State of one two-object pickup on the robot, shared by its stages and by a trip after it.

    ``prepare`` plans and executes everything up to the first close; ``pickup_steps`` is the rest,
    as a generator that yields a decision point at every control step of the two grasp phases (see
    ``dexcarry/control/grasp_controller.py``). ``snapshot`` copies the state a later episode can
    restart from (with the simulator's own saved state), which is how the RL environment reuses
    one approach for several episodes.
    """

    def __init__(self, env, plan: TwoObjectPlan, seed, options: ReachOptions):
        if (plan.first_id, plan.second_id) != (env.first_spec.object_id, env.second_spec.object_id):
            raise ValueError("plan was made for other objects")
        if plan.support != "table":
            raise ValueError("on the robot the first object is taken from the table")
        if set(plan.first_fingers) & set(plan.second_fingers):
            raise ValueError("a finger cannot hold both objects")
        self.env, self.plan, self.seed, self.options = env, plan, seed, options
        self.cfg = env.hand_config
        self.lo, self.hi = env.hand_joint_limits()
        self.order, self.groups = self.cfg["joint_order"], self.cfg["finger_groups"]
        self.result = {"both_held": False, "clean": False, "first_failure": None, "diagnostics": {"timeline": {}}}
        self.executor = ArmExecutor(env)
        self.arm, self.base = env.dofs(ARM_AND_TORSO), env.dofs(FETCH_BASE_JOINTS)
        self.command = None

    @property
    def notes(self):
        return self.result["diagnostics"]

    def failed(self, code):
        self.result["first_failure"] = code
        if code in PLANNER_FAILURES:
            self.notes["planner_rejected"] = True
        return self.result, None

    def steps(self, seconds):
        return round(seconds / self.env.dt)

    def in_palm(self, body):
        return in_palm(self.env, body)

    def kept(self, reference, body):
        now = self.in_palm(body)
        return bool(np.linalg.norm(now[:3, 3] - reference[:3, 3]) <= SLIP_TRANSLATION
                    and rotation_between_deg(reference, now) <= SLIP_ROTATION_DEG)

    def posed(self, hand_q):
        # The last hand target sent, so recovery can keep holding what a failed pickup left in the hand.
        self.last_hand = hand_q
        return posed(self.env, self.command, hand_q)

    def step_base(self, goal, hand_q):
        step_base(self.env, self.command, goal, hand_q)

    def prepare(self) -> str | None:
        """Plan the side wrap, step the base and bring the open hand onto the first object."""
        env, plan = self.env, self.plan
        failure, first = plan_first(env, plan, self.seed, self.options)
        if failure:
            return failure
        self.first = first
        self.notes.update(standing_offset=first["standing_offset"], first_retreat=first["retreat"])
        self.command = env.hold_action()
        self.pre, self.closed = (np.clip(np.asarray(v, float), self.lo, self.hi) for v in (plan.first_pre, plan.first_closed))
        for _ in range(40):
            env.step(self.posed(self.pre))
        rest = [env.body_transform(b)[:3, 3] for b in (env.object_id, env.second_id)]
        self.step_base(first["base"], self.pre)
        self.executor.follow_path(first["path"], base=self.posed(self.pre))
        self.command[self.arm] = first["q_goal"]
        moved = max(float(np.linalg.norm(env.body_transform(b)[:3, 3] - r)) for b, r in zip((env.object_id, env.second_id), rest, strict=True))
        self.notes["first_approach_displacement"] = moved
        if moved > APPROACH_DISPLACEMENT_LIMIT:
            return "FIRST_DISTURBED"
        self.first_grasp = env.get_link_transform(env.hand_root_link)
        self.step_count = env.step_count
        return None

    def snapshot(self):
        """A copy that can run ``pickup_steps`` again from where ``prepare`` left off, once the
        simulator is restored to the state saved at that moment."""
        twin = copy.copy(self)
        twin.command = self.command.copy()
        twin.result = copy.deepcopy(self.result)
        return twin

    def point(self, phase, stage, k, n, scripted):
        env, plan = self.env, self.plan
        body, fingers, other = ((env.object_id, tuple(plan.first_fingers), None) if phase == "first" else
                                (env.second_id, tuple(plan.second_fingers), env.object_id))
        return {"phase": phase, "stage": stage, "k": k, "n": n, "scripted": scripted, "env": env, "object": body,
                "fingers": fingers, "held": other, "limits": (self.lo, self.hi), "pickup": self}

    def _hold(self, phase, stage, n, scripted):
        hand = scripted
        for k in range(n):
            hand = yield self.point(phase, stage, k, n, scripted)
            self.env.step(self.posed(hand))
        return hand

    def _lift(self, phase, q_target, n, scripted):
        """``ArmExecutor.move_to`` with the hand decided at every step."""
        env = self.env
        q0, q_target, hand = env.get_joints(ARM_AND_TORSO), np.asarray(q_target, float), scripted
        for k in range(1, n + 1):
            q = q0 + (q_target - q0) * (k / n)
            hand = yield self.point(phase, "lift", k - 1, n, scripted)
            env.step(self.executor.action_for(q, self.posed(hand)))
        return hand


class Failed(tuple):
    """What a stage returns when the pickup has failed: ``(result, None)`` as ``_pick_both`` returns."""


def first_phase(pk: Pickup, params: ClosingParams):
    """Side wrap of the first object, lift, settle and hold. A generator of decision points; returns
    ``Failed`` or the context the rest of the pickup needs."""
    env, plan, notes = pk.env, pk.plan, pk.notes
    cfg, lo, hi, order, groups = params.hand_config(pk.cfg), pk.lo, pk.hi, pk.order, pk.groups
    command, arm, first = pk.command, pk.arm, pk.first
    pre, closed = pk.pre, pk.closed.copy()
    if params.wrap_overdrive:
        flex = np.array([j in cfg["flexion_joints"] for j in order])
        for finger in plan.first_fingers:
            idx = [order.index(j) for j in groups[finger]]
            closed[idx] += params.wrap_overdrive * flex[idx]
        closed = np.clip(closed, lo, hi)
    # Side wrap: the wrapping fingers never halt at first touch.
    close = ContactStopClose(cfg, pre, closed, lo, hi, envelop=plan.first_fingers)
    n = pk.steps(params.ramp_seconds)
    for k, alpha in enumerate(np.linspace(0, 1, n)):
        hand = yield pk.point("first", "ramp", k, n, close.command(alpha))
        env.step(pk.posed(hand))
    last = yield from pk._hold("first", "squeeze", pk.steps(params.squeeze_seconds), closed)
    if not env.fingers_touching(env.object_id) & set(plan.first_fingers):
        return Failed(pk.failed("FIRST_NO_CONTACT"))
    start_z = env.object_position()[2]
    last = yield from pk._lift("first", first["q_lift"], pk.steps(1.), closed)
    command[arm] = first["q_lift"]
    last = yield from pk._hold("first", "settle", pk.steps(SETTLE_SECONDS), closed)
    reference = pk.in_palm(env.object_id)
    last = yield from pk._hold("first", "hold", pk.steps(2.), closed)
    notes["first_lift_height"] = float(env.object_position()[2] - start_z)
    if notes["first_lift_height"] < LIFT - .04 or not pk.kept(reference, env.object_id):
        return Failed(pk.failed("FIRST_NOT_LIFTED"))
    if not env.fingers_touching(env.object_id) <= set(plan.first_fingers):
        return Failed(pk.failed("FIRST_OTHER_FINGER"))
    notes["timeline"]["first_lifted"] = env.step_count * env.dt
    # The wrapping fingers stay where the first phase last commanded them (the scripted rule: closed).
    q_pre, q_closed = closed.copy(), closed.copy()
    for finger in plan.first_fingers:
        idx = [order.index(j) for j in groups[finger]]
        q_pre[idx] = q_closed[idx] = np.asarray(last, float)[idx]
    for finger in plan.second_fingers:
        idx = [order.index(j) for j in groups[finger]]
        q_pre[idx], q_closed[idx] = np.asarray(plan.second_pre, float)[idx], np.asarray(plan.second_closed, float)[idx]
    return {"reference": reference, "q_pre": np.clip(q_pre, lo, hi), "q_closed": np.clip(q_closed, lo, hi)}


def second_approach(pk: Pickup, ctx: dict, probe=None):
    """Open the second grasp's fingers, plan and make the arm motion onto the second object. No
    decision points. Returns ``Failed`` or the context, now with the plan of the second reach."""
    env, plan, notes = pk.env, pk.plan, pk.notes
    q_pre = ctx["q_pre"]
    for _ in range(pk.steps(.5)):
        env.step(pk.posed(q_pre))
    if probe:
        # The same lifted state under several planner settings, each with the same random stream.
        notes["second_probe"] = {}
        for name, variant in probe.items():
            outcome, _, why = _plan_second(env, plan, np.random.default_rng([pk.seed, 1]), variant)
            notes["second_probe"][name] = {"failure": outcome, "reasons": why}
    failure, second, why = _plan_second(env, plan, np.random.default_rng(pk.seed), pk.options)
    notes["second_reasons"] = why
    if failure:
        return Failed(pk.failed(failure))
    notes.update(second_retreat=second["retreat"], second_base_step=second["base_step"])
    if second.get("q_high") is not None:
        # the any-pair counter: the hand rises, the base drives at the carry's speed, then the path below brings the arm down.
        notes["second_step"] = _raised_step(pk, second, q_pre, ctx["reference"])
        if not pk.kept(ctx["reference"], env.object_id):
            return Failed(pk.failed("FIRST_LOST_MOVING"))
    else:
        pk.step_base(second["base"], q_pre)
    home = env.body_transform(env.second_id)[:3, 3].copy()
    pk.executor.follow_path(second["path"], base=pk.posed(q_pre))
    pk.command[pk.arm] = second["q_goal"]
    if not pk.kept(ctx["reference"], env.object_id):
        return Failed(pk.failed("FIRST_LOST_MOVING"))
    notes["second_approach_displacement"] = float(np.linalg.norm(env.body_transform(env.second_id)[:3, 3] - home))
    if notes["second_approach_displacement"] > APPROACH_DISPLACEMENT_LIMIT:
        return Failed(pk.failed("SECOND_PUSHED"))
    return {**ctx, "second": second, "second_grasp": env.get_link_transform(env.hand_root_link)}


def _raised_step(pk: Pickup, second: dict, hand_q, reference) -> dict:
    """the any-pair counter's move between the grasps (``_plan_second_stepped``'s plan): the arm to ``q_high`` at ``STEP_RISE_SPEED``, the base
    to ``second["base"]`` on the carry's speed limits (``MotionProfile``), held still ``STEP_SETTLE_STEPS``. Returns its record:
    rise, metres, seconds, and how far the wrapped object moved in the palm meanwhile (metres, degrees)."""
    env, began = pk.env, pk.env.step_count
    rise = float(second["rise"])
    if rise > 0:
        pk.executor.move_to(second["q_high"], steps=max(1, round(max(.5, rise / STEP_RISE_SPEED) / env.dt)), base=pk.posed(hand_q))
        pk.command[pk.arm] = second["q_high"]
    here = pk.command[pk.base].copy()
    _, route = return_route(here, np.asarray(second["base"], float), MotionProfile(), env.dt)
    for pose in route[1:]:
        pk.command[pk.base] = pose
        env.step(pk.posed(hand_q))
    for _ in range(STEP_SETTLE_STEPS):
        env.step(pk.posed(hand_q))
    now = pk.in_palm(env.object_id)
    return {"rise": rise, "metres": float(np.linalg.norm(np.asarray(second["base"][:2], float) - here[:2])), "shift": second.get("shift"),
            "seconds": (env.step_count - began) * env.dt,
            "slip": [float(np.linalg.norm(now[:3, 3] - reference[:3, 3])), float(rotation_between_deg(reference, now))]}


def carry_steps(pk: Pickup, profile: MotionProfile, still_seconds: float, scripted):
    """the carry after the hold: the base drives ``profile`` along the scene's travel direction (as the trip's carry does), the
    arm and torso held at their commanded pose, then ``still_seconds`` with the base at the route's end. A generator of decision
    points (stages "drive" and "still") with ``scripted`` as the hand; returns the last hand target and the carry's record."""
    env = pk.env
    base, p = env.dofs(FETCH_BASE_JOINTS), env.p
    bodies, scene = (env.object_id, env.second_id), (env.table_id, env.target_table_id, env.plane_id)
    start = env.get_base_pose()
    _, route = base_trajectory(start, profile, env.dt, env.scenario.travel_yaw)
    record = {"object_scene_contact_steps": 0, "robot_table_contact_steps": 0}

    def touches():
        record["object_scene_contact_steps"] += any(p.getContactPoints(b, s) for b in bodies for s in scene)
        record["robot_table_contact_steps"] += any(p.getContactPoints(env.robot_id, s) for s in (env.table_id, env.target_table_id))

    hand, n = scripted, len(route) - 1
    for k, pose in enumerate(route[1:]):
        pk.command[base] = pose
        hand = yield pk.point("second", "drive", k, n, scripted)
        env.step(pk.posed(hand))
        touches()
    record["drive_steps"] = n
    record["drive_metres"] = float(np.linalg.norm(env.get_base_pose()[:2] - start[:2]))
    record["drive_second_kept"] = None # filled by the caller, which holds the references
    m = pk.steps(still_seconds)
    for k in range(m):
        hand = yield pk.point("second", "still", k, m, scripted)
        env.step(pk.posed(hand))
        touches()
    record["still_steps"] = m
    record["metres"] = float(np.linalg.norm(env.get_base_pose()[:2] - start[:2]))
    return hand, record


def second_phase(pk: Pickup, ctx: dict, hold_seconds, params: ClosingParams, carry: MotionProfile | None = None,
                 still_seconds: float = 1.):
    """Close on the second object, lift and hold both. A generator of decision points; returns
    ``(result, state)`` as ``_pick_both`` always has.

    ``carry``: after the hold, the base drives the profile with the arm held at
    its post-lift pose, then ``still_seconds`` still (``carry_steps``). Both objects' holds are judged at the end of the hold
    (``result["hold"]``) and again at the end of the carry (``first_kept``, ``second_kept``, against the same references); the
    carry runs whatever the hold left in the hand."""
    env, plan, notes = pk.env, pk.plan, pk.notes
    cfg, lo, hi, order, groups = params.hand_config(pk.cfg), pk.lo, pk.hi, pk.order, pk.groups
    command, arm, second, q_pre = pk.command, pk.arm, ctx["second"], ctx["q_pre"]
    # The fingers on the first object keep wrapping; the others stop where they touch the second.
    close = ContactStopClose(cfg, q_pre, ctx["q_closed"], lo, hi, envelop=plan.first_fingers)
    n = pk.steps(params.ramp_seconds)
    for k, alpha in enumerate(np.linspace(0, 1, n)):
        hand = yield pk.point("second", "ramp", k, n, close.command(alpha))
        env.step(pk.posed(hand))
        close.observe(env.finger_contacts(env.second_id), env.hand_joint_values())
    n = pk.steps(params.squeeze_seconds)
    for k, alpha in enumerate(np.linspace(0, 1, n)):
        hand = yield pk.point("second", "squeeze", k, n, close.squeeze(alpha))
        env.step(pk.posed(hand))
    grip = close.squeeze(1.)
    second_reference, start_z = pk.in_palm(env.second_id), env.body_transform(env.second_id)[2, 3]
    last = yield from pk._lift("second", second["q_lift"], pk.steps(1.), grip)
    command[arm] = second["q_lift"]
    last = yield from pk._hold("second", "hold", pk.steps(hold_seconds), grip)
    # What the carry holds on with: the last hand target commanded (the scripted rule: the grip).
    grip = np.asarray(last, float)

    def second_held():
        return bool(env.body_transform(env.second_id)[2, 3] - start_z >= LIFT - .04 and pk.kept(second_reference, env.second_id))

    first_kept = pk.kept(ctx["reference"], env.object_id)
    second_kept = second_held()
    lost_code = None
    if carry is not None:
        pk.result["hold"] = {"first_kept": bool(first_kept), "second_kept": bool(second_kept)}
        moving, drive_kept = carry_steps(pk, carry, still_seconds, last), None
        point = next(moving)
        try:
            while True:
                if point["stage"] == "still" and point["k"] == 0:
                    drive_kept = second_held()
                point = moving.send((yield point))
        except StopIteration as stop:
            last, record = stop.value
        record["drive_second_kept"] = drive_kept
        pk.result["carry"] = record
        grip = np.asarray(last, float)
        first_kept, second_kept = pk.kept(ctx["reference"], env.object_id), second_held()
        held_before = pk.result["hold"]
        gone = {name for name, before, now in (("first", held_before["first_kept"], first_kept),
                                               ("second", held_before["second_kept"], second_kept)) if before and not now}
        if gone:
            lost_code = "BOTH_LOST_CARRY" if len(gone) == 2 else f"{min(gone).upper()}_LOST_CARRY"
    pk.result.update(first_kept=bool(first_kept), second_kept=bool(second_kept), both_held=bool(first_kept and second_kept),
                     clean=bool(env.fingers_touching(env.object_id) <= set(plan.first_fingers)
                                and env.fingers_touching(env.second_id) <= set(plan.second_fingers)))
    notes["timeline"]["both_lifted"] = env.step_count * env.dt
    if not pk.result["both_held"]:
        return pk.failed(lost_code or ("FIRST_LOST_ON_SECOND_GRASP" if second_kept else "SECOND_NOT_LIFTED" if first_kept else "BOTH_LOST"))
    open_first = q_pre.copy()
    for finger in plan.first_fingers:
        idx = [order.index(j) for j in groups[finger]]
        open_first[idx] = pk.pre[idx]
    return pk.result, {"command": command, "executor": pk.executor, "grip": grip, "first": pk.first, "second": second,
                       "first_grasp": pk.first_grasp, "second_grasp": ctx["second_grasp"], "open_second": q_pre,
                       "open_both": open_first, "first_pick_open": pk.pre.copy()}


def pickup_steps(pk: Pickup, hold_seconds=5., probe=None, params: ClosingParams | None = None):
    """Both grasps, from the first close to the end of the hold after the second lift.

    A generator: yields a decision point at every control step of the grasp phases and expects the
    hand target back (``control.grasp_controller.drive``). Returns ``(result, state)`` as
    ``_pick_both`` always has; ``state`` is None unless both objects are held.
    """
    params = params or ClosingParams()
    ctx = yield from first_phase(pk, params)
    if isinstance(ctx, Failed):
        return tuple(ctx)
    pk.ctx = ctx
    ctx = second_approach(pk, ctx, probe)
    if isinstance(ctx, Failed):
        return tuple(ctx)
    pk.ctx = ctx
    return (yield from second_phase(pk, ctx, hold_seconds, params))


def _pick_both(env, plan: TwoObjectPlan, seed, hold_seconds, options: ReachOptions, probe, controller=None):
    """The pickup. Returns the result and, when both are held, what a trip needs to go on."""
    pk = Pickup(env, plan, seed, options)
    failure = pk.prepare()
    if failure:
        return pk.failed(failure)
    controller = controller or ScriptedClose()
    controller.reset()
    return drive(pickup_steps(pk, hold_seconds, probe, controller.params), controller)


def run_two_object(env, plan: TwoObjectPlan, *, seed=0, hold_seconds=5., options: ReachOptions | None = None,
                   probe: dict | None = None, controller=None) -> dict:
    """Execute a two-object plan on the robot up to the second lift. ``env`` is a freshly reset ``TwoObjectEnv``.

    ``probe`` maps names to ``ReachOptions``: the second arm motion is also planned, not executed,
    under each of them from the same lifted state, and the outcomes go into the diagnostics.
    ``controller`` decides the finger targets of the grasp phases (default: the scripted rule).
    """
    return _pick_both(env, plan, seed, hold_seconds, options or ReachOptions(), probe, controller)[0]


def carry_configuration(q_lift, posture):
    """Arm and torso configurations to try for a carry posture: where the lift ended, or a library shape."""
    if posture == HOLD:
        return [(HOLD, np.asarray(q_lift, float))]
    return [(name, np.r_[q_lift[0], arm]) for name, arm in posture_candidates(posture)]


def _plan_carry(env, q_lift, profile, posture, bodies=None):
    """Carry configuration that is collision-free with the held objects (default: both) in hand, on the
    way there and along the drive. Returns ``(failure, plan)``; the plan holds the palm-frame load
    direction at rest."""
    p = env.p
    lower, upper = env.joint_limits(ARM_AND_TORSO)
    bodies = (env.object_id, env.second_id) if bodies is None else tuple(bodies)
    state = p.saveState()
    try:
        held = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=bodies[0])
        for body in bodies[1:]:
            held.attach_also(body)
        start = env.get_base_pose()
        _, route = base_trajectory(start, profile, env.dt, env.scenario.travel_yaw)
        for name, q in carry_configuration(q_lift, posture):
            if np.any(q < lower) or np.any(q > upper):
                continue
            env.set_joints(dict(zip(FETCH_BASE_JOINTS, start, strict=True)))
            if any(held(q_lift + a * (q - q_lift)) for a in np.linspace(0, 1, 80)):
                continue
            load = env.get_link_transform(env.hand_root_link)[:3, :3].T @ np.array([0., 0., -1.])
            blocked = False
            for pose in route[::max(1, len(route) // 100)]:
                env.set_joints(dict(zip(FETCH_BASE_JOINTS, pose, strict=True)))
                if held(q):
                    blocked = True
                    break
            if not blocked:
                return None, {"q": q, "name": name, "load_direction": load.tolist(), "route": route}
        return "CARRY_BLOCKED", None
    finally:
        p.restoreState(state)
        p.removeState(state)


def end_state(env, body) -> dict:
    """Where a trip left an object: in the hand, standing or fallen on a table, or on the floor."""
    p = env.p
    T = env.body_transform(body)
    tilt = float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))))
    if p.getContactPoints(body, env.robot_id) and not any(p.getContactPoints(body, other) for other in
                                                         (env.table_id, env.target_table_id, env.plane_id)):
        where = "in_hand"
    else:
        table = "pick" if p.getContactPoints(body, env.table_id) else "target" if p.getContactPoints(body, env.target_table_id) else None
        where = "floor" if table is None else f"{'standing' if tilt <= UPRIGHT_DEG else 'fallen'}_{table}"
    return {"where": where, "tilt_deg": tilt, "position": T[:3, 3].tolist()}

