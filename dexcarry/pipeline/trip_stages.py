"""The two-object trip as a chain of stages (pick, carry, set down both, withdraw) and the steps recovery reuses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation

from ..control.grasp_controller import ScriptedClose, drive
from ..envs.fetch import FETCH_BASE_JOINTS
from ..grasp.grasp_vector import make_transform
from ..kin.collision import PyBulletCollisionChecker
from ..kin.ik import DLSIK
from ..kin.rrt import rrt_connect, shortcut
from ..sim.closing import ContactStopClose
from ..sim.loads import SLIP_ROTATION_DEG, SLIP_TRANSLATION, rotation_between_deg
from ..sim.transport import MotionProfile, return_route
from ..sim.two_object import LIFT, TwoObjectPlan
from .common import ARM_AND_TORSO, ArmExecutor
from .execution import APPROACH_DISPLACEMENT_LIMIT
from .feasibility import TUCK_STEPS, _drive_clear
from .two_object import (
    HOLD,
    MORE_SECOND_RETREATS,
    PLACE_RAISES,
    PLANNER_FAILURES,
    RELEASE_SECONDS,
    UPRIGHT_DEG,
    Failed,
    Pickup,
    ReachOptions,
    _plan_carry,
    _reach,
    end_state,
    first_phase,
    in_palm,
    pickup_steps,
    posed,
    step_base,
    symmetries,
)

NAMES = ("first", "second")
# Recovery executors. A first version stowed the arm with a straight joint move, which swept through objects already set
# down, and retook a wrapped object alone with the side wrap (21 falls, 1 delivery). The planned-stow recovery plans the stow
# with RRT-Connect and never retakes a wrapped object alone. Its known limit (development runs): with the delivered wrapped
# object left standing, the carry of the next object at lift height can graze it and is refused (14 of 142 failed trips).
# The posture recovery adds a fallback: when the carry of an object taken alone is refused at lift posture (its hand would
# reach over an object already delivered), the posture library's carry shapes, which hold it in front of the torso, are
# tried before refusing (``CARRY_POSTURES``).
# errand recovery: deliver what is held wherever the hand is
# (``deliver``), look again, retry the second grasp once with the wrapped object still held, retake every other standing
# object alone with the single-trip grasp. Its look and single grasps belong to the errand, so it runs only there
# (``pipeline.errand.recover_errand recovery``); ``run_trip_with_recovery`` refuses it.
RECOVERY = {"basic": {"planned_stow": False, "retake_wrapped": True},
            "planned": {"planned_stow": True, "retake_wrapped": False},
            "postures": {"planned_stow": True, "retake_wrapped": False, "carry_postures": True},
            "errand": {"planned_stow": True, "retake_wrapped": True, "carry_postures": True, "errand_only": True}}
CARRY_POSTURES = ("front_down_r0", "front_tilt_r0", "front_high_r0", "front_tilt_r90", "front_tilt_r270", "front_high_r90",
                  "front_high_r270", "front_high_r180")
RING_JOINTS = [8, 9, 10, 11] # rf_mcp, rf_rot, rf_pip, rf_dip in the LEAP hand's canonical joint order (configs/hands/leap.yaml)
RELEASE_V2 = {
    "upright": True, # turn the palm so the object, as it now sits in the palm, stands upright (v1: the palm as at the grasp)
    "fallback": True, # if no upright set-down can be planned, try the palm as at the grasp before refusing
    "partial": (.75, .5, .25), # before that fallback, palms turned this fraction of the way to upright
    "stances": ((.05, 0.), (-.05, 0.), (0., .05), (0., -.05), (.1, 0.), (-.1, 0.), (0., .1), (0., -.1)),
                             # before the partial palms: the upright palm from these base offsets (m, world x-y), each
                             # base move collision-checked with the load
    "clearance": .003, # height of the object's bottom above the table at the release (v1: 3 mm)
    "raises": (0., .0025, .005, .0075, .01, .015, .02, .03), # release heights tried above that, in order (v1: PLACE_RAISES)
    "straighten_deg": 3., # at the release pose, before opening: if the object leans more than this from vertical...
    "straighten_tries": 2, #... re-pose the palm to stand it upright on its spot, at most this many times (v1: 0)
    "straighten_seconds": .5, # duration of each re-posing move
    "straighten_undo": True, # a re-posing move that leaves the object leaning more is undone, and no more are tried
    "seat_max": .04, # then, before opening: lower the palm straight down by up to this until the object touches the
                     # table; 0 is off (v1)
    "seat_step": .001, # in steps of this
    "seat_press": .001, # and once it touches, this much further, if it had to be lowered
    "seat_hand_stop": True, # stop lowering once the hand touches the table
    "fallen_deg": 20., # after opening: an object leaning more than this has fallen; the palm then rises before backing off
    "open_seconds": .5, # how long the hand takes to open (v1: 0.5 s)
    "open": "ring", # "all": every finger to the first grasp's preshape (v1); "ring": only the wrapping ring
                    # finger opens, the others follow once the palm has backed off
    "settle_seconds": .3, # hold with the hand open before backing off (v1: 0.3 s)
    "withdraw": "normal", # "joint": v1's joint-space move to the retreat pose; "normal": a straight Cartesian back-off
    "withdraw_distance": .02, # metres along the palm normal, for "normal"
    "withdraw_up": .08, # then metres straight up, for "normal", the palm's orientation kept throughout
    "withdraw_to_above": False, # after the back-off, still move to v1's retreat pose
    "withdraw_seconds": 1., # duration of the final move to the retreat pose (v1: 1 s)
}


@dataclass
class TripState:
    """Everything a trip's stages share: the scene, the plan, the result being built and where each object is."""
    env: object
    plan: TwoObjectPlan
    profile: MotionProfile
    seed: int = 0
    posture: str = HOLD
    options: ReachOptions = field(default_factory=ReachOptions)
    release_fix: bool = True
    result: dict = field(default_factory=dict)
    # What the pickup hands on when both objects are held: command, executor, grip, grasps and open hands.
    pick: dict | None = None
    rng: np.random.Generator | None = None
    held: set = field(default_factory=set)
    where: dict = field(default_factory=dict)
    hand: np.ndarray | None = None
    carry: dict | None = None
    references: dict = field(default_factory=dict)
    worst: dict = field(default_factory=dict)
    travel: np.ndarray | None = None
    # The pickup that ran (kept for recovery), where the base started, and which table the robot is at.
    pickup: Pickup | None = None
    home: np.ndarray | None = None
    rest: np.ndarray | None = None
    spare: np.ndarray | None = None
    recovery: str = "planned"
    at: str = "pick"
    # Which release the wrapped object gets (``RELEASES``); "v1" is every earlier result's.
    release: str = "v1"

    @property
    def notes(self) -> dict:
        return self.result["diagnostics"]

    @property
    def bodies(self) -> dict:
        env = self.env
        return {"first": (env.object_id, env.first_spec, env.goal_spots[0]),
                "second": (env.second_id, env.second_spec, env.goal_spots[1])}

    def steps(self, seconds) -> int:
        return round(seconds / self.env.dt)

    def mark(self, phase):
        self.notes["timeline"][phase] = self.env.step_count * self.env.dt

    def observe(self):
        """Where each object is now. Reads contacts only; nothing is stepped."""
        self.where = {name: end_state(self.env, body)["where"] for name, (body, _, _) in self.bodies.items()}

    @property
    def command(self) -> np.ndarray:
        """The joint targets the robot is being held at."""
        if self.pick is not None:
            return self.pick["command"]
        if self.pickup is not None and self.pickup.command is not None:
            return self.pickup.command
        if self.spare is None:
            self.spare = self.env.hold_action()
        return self.spare

    @property
    def executor(self) -> ArmExecutor:
        if self.pick is not None:
            return self.pick["executor"]
        if self.pickup is None:
            self.pickup = Pickup(self.env, self.plan, self.seed, self.options)
        return self.pickup.executor


def finish(state: TripState, code=None) -> dict:
    """Record ``code`` as the first failure unless one is recorded, then the set-down tests and end states."""
    env, result, notes = state.env, state.result, state.notes
    if code and not result["first_failure"]:
        result["first_failure"] = code
        if code in PLANNER_FAILURES:
            notes["planner_rejected"] = True
    notes["set_down"] = {name: env.set_down(*entry) for name, entry in state.bodies.items()}
    notes["end_state"] = {name: end_state(env, entry[0]) for name, entry in state.bodies.items()}
    state.where = {name: notes["end_state"][name]["where"] for name in NAMES}
    result["delivered"] = [notes["set_down"][name]["success"] for name in NAMES]
    result["upright"] = [notes["set_down"][name]["tilt_deg"] <= UPRIGHT_DEG for name in NAMES]
    result["seconds"] = env.step_count * env.dt
    return result


def pick_pair(state: TripState, settle_seconds=1., controller=None) -> str | None:
    """Both grasps and the lift, held for ``settle_seconds``. On success the hand holds both objects."""
    env = state.env
    # The start pose, before anything moves: the commanded base and arm (``hold_action`` is the last command).
    state.home = env.hold_action()[env.dofs(FETCH_BASE_JOINTS)].copy()
    state.rest = env.hold_action()[env.dofs(ARM_AND_TORSO)].copy()
    # The calls of ``_pick_both``, keeping the pickup for recovery.
    pk = state.pickup = Pickup(env, state.plan, state.seed, state.options)
    failure = pk.prepare()
    if failure:
        result, pick = pk.failed(failure)
    else:
        controller = controller or ScriptedClose()
        controller.reset()
        result, pick = drive(pickup_steps(pk, settle_seconds, None, controller.params), controller)
    result.update(pickup=result["both_held"], carried=None)
    state.result, state.pick = result, pick
    state.observe()
    if pick is None:
        return result["first_failure"]
    state.rng = np.random.default_rng([state.seed, 2])
    state.held, state.hand = set(NAMES), pick["grip"]
    return None


def slipped(state: TripState) -> set:
    """Objects that have moved past the slip limits in the palm since the carry began; updates the worst slip."""
    gone = set()
    for name, (body, _, _) in state.bodies.items():
        if name not in state.references:
            continue
        now, reference = in_palm(state.env, body), state.references[name]
        worst = state.worst[name]
        state.worst[name] = [max(worst[0], float(np.linalg.norm(now[:3, 3] - reference[:3, 3]))),
                             max(worst[1], rotation_between_deg(reference, now))]
        if state.worst[name][0] > SLIP_TRANSLATION or state.worst[name][1] > SLIP_ROTATION_DEG:
            gone.add(name)
    return gone


def carry(state: TripState, postures=()) -> str | None:
    """Change to the carry posture and drive the profile to the target table, watching both objects for slip. If
    the carry is refused in ``state.posture``, each of ``postures`` is tried in turn (the posture recovery's fallback)."""
    env, notes, command = state.env, state.notes, state.pick["command"]
    arm, base = env.dofs(ARM_AND_TORSO), env.dofs(FETCH_BASE_JOINTS)
    q_lift = command[arm].copy()
    held = [name for name in NAMES if name in state.held]
    failure, plan = _plan_carry(env, q_lift, state.profile, state.posture, [state.bodies[name][0] for name in held])
    for posture in postures if failure == "CARRY_BLOCKED" else ():
        failure, plan = _plan_carry(env, q_lift, state.profile, posture, [state.bodies[name][0] for name in held])
        if failure is None:
            notes.setdefault("carry_postures_tried", []).append(posture)
            break
    if failure:
        return failure
    state.carry = plan
    notes["carry"] = {"posture": plan["name"], "load_direction": plan["load_direction"]}
    # One reference per held object, taken when the carry begins, covers the posture change and the drive.
    state.references = {name: in_palm(env, state.bodies[name][0]) for name in held}
    state.worst = {name: [0., 0.] for name in held}
    gone, phase = set(), "posture"
    for k in range(1, (TUCK_STEPS if plan["name"] != HOLD else 0) + 1):
        command[arm] = q_lift + (plan["q"] - q_lift) * k / TUCK_STEPS
        env.step(posed(env, command, state.hand))
        gone = slipped(state)
        if gone:
            break
    if not gone:
        phase = "drive"
        for pose in [*plan["route"][1:], *[plan["route"][-1]] * 30]:
            command[base] = pose
            env.step(posed(env, command, state.hand))
            gone = slipped(state)
            if gone:
                break
    state.result["carried"] = not gone
    notes["carry"].update(max_slip={name: {"displacement": v[0], "rotation_deg": v[1]} for name, v in state.worst.items()},
                          lost_in=phase if gone else None)
    state.held -= gone
    state.observe()
    if gone:
        return "BOTH_LOST_CARRY" if len(gone) == 2 else f"{min(gone).upper()}_LOST_CARRY"
    state.mark("arrived")
    state.at = "target"
    state.travel = plan["route"][-1] - plan["route"][0]
    return None


def place(state: TripState, name: str) -> str | None:
    """Bring ``name`` down on its spot by reversing its pick, open the hand, back off.

    The three-finger object is placed first, the hand opening to the second grasp's preshape; then the
    wrapped one, the hand opening to the first grasp's preshape (or, without the release fix of 15 Oct,
    to the second's with the wrapping fingers open)."""
    env, p, pick, notes, rng = state.env, state.env.p, state.pick, state.notes, state.rng
    command, executor = pick["command"], pick["executor"]
    arm, base = env.dofs(ARM_AND_TORSO), env.dofs(FETCH_BASE_JOINTS)
    reach, grasp, hand_q = pick[name], pick[f"{name}_grasp"], state.hand
    hand_open = pick["open_second"] if name == "second" else pick["first_pick_open"] if state.release_fix else pick["open_both"]
    also = tuple(other for other in NAMES if other != name and other in state.held)
    body, spec, spot = state.bodies[name]
    v2 = name == "first" and state.release == "v2"
    clearance = RELEASE_V2["clearance"] if v2 else .003
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    held, R = in_palm(env, body), grasp[:3, :3]
    orientations = [R]
    if v2 and RELEASE_V2["upright"]:
        # A hooked object turns in the palm as it settles, so the palm as at the grasp sets it down leaning (23-31
        # degrees in every fall seen in development). Turn the palm by the smallest rotation that stands it up;
        # if no set-down with it can be planned, fall back to the palm as at the grasp (v1's) rather than refuse.
        R_up = _uprighted(R, held[:3, :3])
        orientations = [R_up] + [_turned(R, R_up, f) for f in RELEASE_V2["partial"]]
        orientations += [grasp[:3, :3]] if RELEASE_V2["fallback"] else []
        R = R_up

    def centre_for(R_try):
        # The object as actually held, with the palm turned as ``R_try``, bottom on the table.
        lowest = float((spec.local_corners() @ (R_try @ held[:3, :3]).T)[:, 2].min())
        return np.array([spot[0], spot[1], env.table_top_z() - lowest + clearance])
    centre = centre_for(R)
    # Stand where the palm is as far from the base as it was at the grasp: one drive further on,
    # corrected for how the object sits in the hand and where its spot is. Many grasps are at
    # the edge of the arm's reach, and a few centimetres decide whether the release is in it.
    stand = reach["base"] + state.travel
    stand[:2] += (centre - R @ held[:3, 3])[:2] - (grasp[:2, 3] + state.travel[:2])
    if np.linalg.norm(stand - command[base]) > 1e-3:
        drive = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS, attached_body=body)
        saved, here = p.saveState(), command[base].copy()
        for other in also:
            drive.attach_also(state.bodies[other][0])
        clear = not any(drive(here + a * (stand - here)) for a in np.linspace(0, 1, 8)[1:])
        p.restoreState(saved)
        p.removeState(saved)
        if clear:
            step_base(env, command, stand, hand_q)
    q_now, (back, up), q_seed = command[arm].copy(), reach["retreat"], reach["q_goal"]
    # The palms to try, each from a base offset (None: where the base stands). v2: when the upright palm cannot be
    # planned from here, it is tried from nearby stances (``stances``) before a palm turned only part of the way.
    candidates = [(orientations[0], None)]
    if v2 and RELEASE_V2["upright"]:
        candidates += [(orientations[0], np.array([*o, 0.])) for o in RELEASE_V2["stances"]]
    candidates += [(R_try, None) for R_try in orientations[1:]]
    carried = [body, *(state.bodies[other][0] for other in also)]
    saved, found, reached = p.saveState(), None, False
    try:
        for R_try, offset in candidates:
            centre_try = centre if R_try is R else centre_for(R_try)
            if offset is not None:
                p.restoreState(saved)
                if not _base_clear(env, command[base], offset, carried):
                    continue
            for higher in RELEASE_V2["raises"] if v2 else PLACE_RAISES:
                p.restoreState(saved)
                if offset is not None:
                    _shift_robot(env, command[base] + offset, offset, carried)
                T = make_transform(R_try, centre_try + [0, 0, higher] - R_try @ held[:3, 3])
                goal = ik.solve_multi(T[:3, 3], R_try, [q_seed, q_now, ik.rest, rng.uniform(ik.lower, ik.upper)])
                above = ik.solve(T[:3, 3] + back * R_try[:, 2] + [0, 0, up], R_try, q0=goal.q)
                if not goal.success or not above.success:
                    continue
                reached = True
                checker = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=body)
                for other in also:
                    checker.attach_also(state.bodies[other][0])
                path = rrt_connect(q_now, above.q, checker, ik.lower, ik.upper, max_iters=state.options.rrt_iters, rng=rng)
                if path is None:
                    continue
                # Only the straight move down may bring the object onto the table.
                checker.allowed_support = {env.target_table_id}
                descent = [above.q + a * (goal.q - above.q) for a in np.linspace(0, 1, 40)]
                blocked = any(checker(q) for q in descent)
                checker.allowed_support = set()
                if blocked:
                    continue
                found = (shortcut(path, checker, n_iters=60, rng=rng) + descent, goal.q, above.q, higher, R_try, offset)
                break
            if found is not None:
                break
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    if found is None:
        state.observe()
        return f"{name.upper()}_PLACE_" + ("COLLISION" if reached else "UNREACHABLE")
    path, q_goal, q_above, higher, R_used, offset = found
    if v2:
        notes.setdefault("release_v2", {})["upright"] = R_used is R
        notes["release_v2"]["palm"] = next(k for k, R_try in enumerate(orientations) if R_try is R_used)
        notes["release_v2"]["stance"] = None if offset is None else offset[:2].round(3).tolist()
        R = R_used
    if offset is not None:
        step_base(env, command, command[base] + offset, hand_q)
    notes.setdefault("release_height", {})[name] = higher + clearance
    executor.follow_path(path, base=posed(env, command, hand_q))
    command[arm] = q_goal
    if "first" in slipped(state):
        state.held.discard("first")
        state.observe()
        return "FIRST_LOST_PLACING"
    if v2:
        return _release_v2(state, ik, R, q_goal, q_above, hand_q, hand_open)
    for alpha in np.linspace(0, 1, state.steps(RELEASE_SECONDS)):
        env.step(posed(env, command, hand_q + alpha * (hand_open - hand_q)))
    executor.hold(state.steps(.3), base=posed(env, command, hand_open))
    executor.move_to(q_above, steps=state.steps(1.), base=posed(env, command, hand_open))
    command[arm] = q_above
    executor.hold(state.steps(.5), base=posed(env, command, hand_open))
    state.mark(f"{name}_released")
    state.held.discard(name)
    state.hand = hand_open
    state.observe()
    return None if env.set_down(body, spec, spot)["success"] else f"{name.upper()}_PLACE_MISSED"


def _uprighted(R: np.ndarray, held_R: np.ndarray) -> np.ndarray:
    """The palm rotation nearest ``R`` that puts an object held at ``held_R`` (its rotation in the palm) upright:
    ``R`` turned by the smallest rotation taking the object's axis, ``(R @ held_R)[:, 2]``, to the vertical."""
    a, b = (R @ held_R)[:, 2], np.array([0., 0., 1.])
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        raise ValueError("the object is upside down in the palm")
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx / (1 + c)) @ R


def _base_clear(env, here, offset, carried) -> bool:
    """Can the base move by ``offset`` from ``here`` with ``carried`` in the hand, touching nothing? State restored."""
    p = env.p
    drive = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS, attached_body=carried[0])
    for other in carried[1:]:
        drive.attach_also(other)
    saved = p.saveState()
    try:
        return not any(drive(here + a * offset) for a in np.linspace(0, 1, 8)[1:])
    finally:
        p.restoreState(saved)
        p.removeState(saved)


def _shift_robot(env, base_pose, offset, carried) -> None:
    """Put the base at ``base_pose`` and move the held bodies by the same ``offset``, so that planning sees the robot
    there with its load (planning only: the caller restores the simulator state)."""
    env.set_joints(dict(zip(FETCH_BASE_JOINTS, base_pose, strict=True)))
    for other in carried:
        position, orientation = env.p.getBasePositionAndOrientation(other)
        env.p.resetBasePositionAndOrientation(other, np.asarray(position) + offset, orientation)


def _turned(R: np.ndarray, R_to: np.ndarray, fraction: float) -> np.ndarray:
    """``R`` turned ``fraction`` of the way to ``R_to``, about the axis of the rotation between them."""
    return Rotation.from_rotvec(fraction * Rotation.from_matrix(R_to @ R.T).as_rotvec()).as_matrix() @ R


def _lean(env, body) -> float:
    """Degrees between a body's own vertical axis and the world's."""
    return float(np.degrees(np.arccos(np.clip(env.body_transform(body)[2, 2], -1., 1.))))


def _release_v2(state: TripState, ik: DLSIK, R: np.ndarray, q_goal, q_above, hand_q, hand_open) -> str | None:
    """Release "v2" of the wrapped object, from the moment it is on its spot
    with the arm at ``q_goal``. Ends as v1 does: the hand open at the retreat pose ``q_above``, the object judged by
    ``set_down``. With ``withdraw="normal"`` the palm first backs straight off along its normal (the palm faces -z of
    its frame, so away from the object is its +z axis), then rises straight up, its orientation kept; an object that
    has fallen by then is cleared by rising first. Then, if ``withdraw_to_above``, it moves to the retreat pose.
    Waypoints the arm cannot reach end the straight part early.

    Before opening, the object's lean is measured again (it can turn further in the palm on the way down, which the
    pose planned at the start of ``place`` does not know). Above ``straighten_deg`` the palm is re-posed, by IK alone,
    so the object stands upright on its spot at the same release height."""
    env, V, notes = state.env, RELEASE_V2, state.notes.setdefault("release_v2", {})
    command, executor = state.pick["command"], state.pick["executor"]
    arm = env.dofs(ARM_AND_TORSO)
    body, spec, spot = state.bodies["first"]
    notes["lean"] = []
    for _ in range(V["straighten_tries"] + 1):
        lean = _lean(env, body)
        notes["lean"].append(round(lean, 2))
        if lean <= V["straighten_deg"] or len(notes["lean"]) > V["straighten_tries"]:
            break
        held, palm = in_palm(env, body), env.get_link_transform(env.hand_root_link)
        R_up = _uprighted(palm[:3, :3], held[:3, :3])
        lowest = float((spec.local_corners() @ (R_up @ held[:3, :3]).T)[:, 2].min())
        centre = np.array([spot[0], spot[1], env.table_top_z() - lowest + state.notes["release_height"]["first"]])
        fix = ik.solve(centre - R_up @ held[:3, 3], R_up, q0=command[arm])
        if not fix.success:
            notes["straighten_failed"] = True
            break
        before = command[arm].copy()
        executor.move_to(fix.q, steps=state.steps(V["straighten_seconds"]), base=posed(env, command, hand_q))
        command[arm] = fix.q
        executor.hold(state.steps(.2), base=posed(env, command, hand_q))
        if V["straighten_undo"] and _lean(env, body) > lean:
            # It leans more than before the move: move back and stop.
            executor.move_to(before, steps=state.steps(V["straighten_seconds"]), base=posed(env, command, hand_q))
            command[arm] = before
            notes["straighten_undone"] = round(_lean(env, body), 2)
            break
    # The release height can leave the object above the table (a raise of up to 3 cm when lower ones cannot be planned,
    # or the arm settling upward); opened there, it drops and tips. Lower it onto the table first, orientation kept.
    if V["seat_max"] > 0:
        table, palm = env.target_table_id, env.get_link_transform(env.hand_root_link)
        lowered, q = 0., command[arm].copy()
        touching = bool(env.p.getContactPoints(body, table))
        # With "seat_hand_stop", lowering stops when the hand reaches the table (if it did not touch it already): pressed
        # further, the fingers tilt the object in the grasp.
        hand_before, hand_down = bool(env.p.getContactPoints(env.robot_id, table)), False
        while not touching and not hand_down and lowered < V["seat_max"] - 1e-9:
            lowered += V["seat_step"]
            down = ik.solve(palm[:3, 3] - [0, 0, lowered], palm[:3, :3], q0=q)
            if not down.success:
                break
            q = command[arm] = down.q
            executor.move_to(q, steps=state.steps(.05), base=posed(env, command, hand_q))
            touching = bool(env.p.getContactPoints(body, table))
            hand_down = V["seat_hand_stop"] and not hand_before and bool(env.p.getContactPoints(env.robot_id, table))
        if touching and lowered > 0 and V["seat_press"] > 0:
            down = ik.solve(palm[:3, 3] - [0, 0, lowered + V["seat_press"]], palm[:3, :3], q0=q)
            if down.success:
                command[arm] = down.q
                executor.move_to(down.q, steps=state.steps(.1), base=posed(env, command, hand_q))
        notes["seated"] = {"lowered": round(lowered, 4), "touching": touching, "hand_down": hand_down}
    # Opening the hand to the first grasp's preshape also swings the thumb, index and middle fingers from the second
    # grasp's preshape; seen on video to drag the can over. "ring" opens the wrapping finger alone.
    final_open, hand_open = hand_open, hand_open.copy()
    if V["open"] == "ring":
        hand_open = hand_q.copy()
        hand_open[RING_JOINTS] = final_open[RING_JOINTS]
    for alpha in np.linspace(0, 1, state.steps(V["open_seconds"])):
        env.step(posed(env, command, hand_q + alpha * (hand_open - hand_q)))
    executor.hold(state.steps(V["settle_seconds"]), base=posed(env, command, hand_open))
    notes["lean_open"] = round(_lean(env, body), 2)
    ended = command[arm].copy()
    if V["withdraw"] == "normal" and V["withdraw_distance"] + V["withdraw_up"] > 0:
        palm = env.get_link_transform(env.hand_root_link)
        back, up = V["withdraw_distance"] * palm[:3, 2], np.array([0, 0, V["withdraw_up"]])
        legs = (up, back) if notes["lean_open"] > V["fallen_deg"] else (back, up)
        targets = [palm[:3, 3] + a * legs[0] for a in np.linspace(0, 1, 6)[1:]]
        targets += [palm[:3, 3] + legs[0] + a * legs[1] for a in np.linspace(0, 1, 6)[1:]]
        q, path = command[arm].copy(), [command[arm].copy()]
        for target in targets:
            back = ik.solve(target, palm[:3, :3], q0=q)
            if not back.success:
                break
            q = back.q
            path.append(q)
        notes["withdrawn"] = (len(path) - 1) / len(targets)
        if len(path) > 1:
            executor.follow_path(path, step=.01, base=posed(env, command, hand_open))
            command[arm] = ended = path[-1]
    if V["withdraw"] != "normal" or V["withdraw_to_above"]:
        executor.move_to(q_above, steps=state.steps(V["withdraw_seconds"]), base=posed(env, command, hand_open))
        ended = q_above
    command[arm] = ended
    if V["open"] == "ring":
        # Clear of the object now: the other fingers go to the preshape too, so the hand ends as v1's does.
        for alpha in np.linspace(0, 1, state.steps(.3)):
            env.step(posed(env, command, hand_open + alpha * (final_open - hand_open)))
        hand_open = final_open
    executor.hold(state.steps(.5), base=posed(env, command, hand_open))
    state.mark("first_released")
    state.held.discard("first")
    state.hand = hand_open
    state.observe()
    return None if env.set_down(body, spec, spot)["success"] else "FIRST_PLACE_MISSED"


def withdraw(state: TripState) -> str | None:
    """Hold the open hand still for a second after both releases; fails if the three-finger object was knocked."""
    state.pick["executor"].hold(state.steps(1.), base=posed(state.env, state.pick["command"], state.hand))
    state.mark("done")
    state.observe()
    return None if state.env.set_down(*state.bodies["second"])["success"] else "SECOND_KNOCKED"


def hand_on(state: TripState):
    """What ``carry`` and ``place`` need, from a pickup that failed with an object still in the hand."""
    pk = state.pickup
    ctx = getattr(pk, "ctx", None) or {}
    state.pick = {"command": pk.command, "executor": pk.executor, "first": pk.first, "first_grasp": pk.first_grasp,
                  "first_pick_open": pk.pre.copy(), "open_second": ctx.get("q_pre"), "second": ctx.get("second"),
                  "second_grasp": ctx.get("second_grasp")}
    state.hand = np.asarray(pk.last_hand, float).copy()
    if state.rng is None:
        state.rng = np.random.default_rng([state.seed, 2])


def settle(state: TripState):
    """Hold the open hand still for a second after a release, as ``withdraw`` does."""
    state.pick["executor"].hold(state.steps(1.), base=posed(state.env, state.command, state.hand))
    state.observe()


def drive_back(state: TripState) -> str | None:
    """Stow the arm, then drive the base straight back to where the trip started, on the trip's speed limits.
    Nothing in the hand is watched: recovery drives back empty."""
    env, command = state.env, state.command
    stow(state)
    base = env.dofs(FETCH_BASE_JOINTS)
    _, route = return_route(command[base], state.home, state.profile, env.dt)
    for pose in [*route[1:], *[route[-1]] * 30]:
        command[base] = pose
        env.step(posed(env, command, state.hand))
    state.at = "pick"
    state.mark("back")
    state.observe()
    return None


def stow(state: TripState):
    """Raise the hand 10 cm, then bring the arm to the posture it started the trip in; nothing if it is there.
    With the the planned-stow recovery executor the way down is planned (RRT-Connect against every body in the scene, so objects
    already set down are avoided); a straight joint move is used only if no path is found, and that is noted."""
    env, command = state.env, state.command
    arm = env.dofs(ARM_AND_TORSO)
    start = state.rest
    if np.linalg.norm(env.get_joints(ARM_AND_TORSO) - start) < 1e-2:
        return
    executor = state.executor
    hand = np.asarray(state.hand if state.hand is not None else env.hand_joint_values(), float)
    state.hand = hand
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    palm = env.get_link_transform(env.hand_root_link)
    up = ik.solve(palm[:3, 3] + [0, 0, .10], palm[:3, :3], q0=env.get_joints(ARM_AND_TORSO))
    if up.success:
        executor.move_to(up.q, steps=state.steps(1.), base=posed(env, command, state.hand))
        command[arm] = up.q
    path = _stow_path(state, start) if RECOVERY[state.recovery]["planned_stow"] else None
    if path is not None:
        executor.follow_path(path, base=posed(env, command, state.hand))
    else:
        if RECOVERY[state.recovery]["planned_stow"]:
            state.notes.setdefault("stow_unplanned", 0)
            state.notes["stow_unplanned"] += 1
        executor.move_to(start, steps=state.steps(2.), base=posed(env, command, state.hand))
    command[arm] = start


def _stow_path(state: TripState, goal):
    """A collision-free arm path from where the arm is to ``goal``, or None. Nothing is executed."""
    env, p = state.env, state.env.p
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    rng = np.random.default_rng([state.seed, 4, state.env.step_count])
    q_now = env.get_joints(ARM_AND_TORSO)
    saved = p.saveState()
    try:
        checker = PyBulletCollisionChecker(env, ARM_AND_TORSO)
        path = rrt_connect(q_now, np.asarray(goal, float), checker, ik.lower, ik.upper,
                           max_iters=state.options.rrt_iters, rng=rng)
        if path is not None:
            path = shortcut(path, checker, n_iters=60, rng=rng)
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    return path


def to_start(state: TripState, hand_q):
    """Empty hand: open it to ``hand_q``, stow the arm and bring the base to where the trip started, so a
    pickup can be planned as at the start of a trip."""
    env, command = state.env, state.command
    hand = np.asarray(state.hand if state.hand is not None else env.hand_joint_values(), float)
    for alpha in np.linspace(0, 1, state.steps(RELEASE_SECONDS)):
        env.step(posed(env, command, hand + alpha * (hand_q - hand)))
    state.hand = np.asarray(hand_q, float).copy()
    stow(state)
    step_base(env, command, state.home, state.hand)
    state.at = "pick"


def pick_single(state: TripState, name: str, controller=None, attempt=0) -> str | None:
    """Take ``name`` alone from the pick table with the plan's own grasp of it: the side wrap for the first
    object, the three-finger grasp for the second (the wrapping fingers curled as when they hold the
    first). Lift and hold. On success the hand holds it, and ``carry`` and ``place`` can follow. The
    planner's random stream is ``[seed, 3, attempt]``."""
    env, plan = state.env, state.plan
    lo, hi = env.hand_joint_limits()
    to_start(state, np.clip(np.asarray(plan.first_pre, float), lo, hi))
    pk = Pickup(env, plan, [state.seed, 3, attempt], state.options)
    controller = controller or ScriptedClose()
    controller.reset()
    if name == "first":
        failure = pk.prepare()
        if failure:
            return failure
        ctx = drive(first_phase(pk, controller.params), controller)
        if isinstance(ctx, Failed):
            state.observe()
            return ctx[0]["first_failure"]
        state.pickup = pk
        hand_on(state)
        state.held = {"first"}
        state.observe()
        return None
    return _second_alone(state, pk, controller.params, lo, hi, attempt)


def _second_alone(state: TripState, pk: Pickup, params, lo, hi, attempt) -> str | None:
    """The three-finger grasp of the second object with nothing else in the hand: the reach from where the
    trip started (each standing offset, then each base move of ``options.base_steps`` from it, each
    symmetric turn), the close of ``second_phase``, a lift and a second's hold."""
    env, plan = state.env, state.plan
    rng = np.random.default_rng([state.seed, 3, attempt, 1])
    order, groups = env.hand_config["joint_order"], env.hand_config["finger_groups"]
    q_pre = np.clip(np.asarray(plan.first_closed, float), lo, hi)
    q_closed = q_pre.copy()
    for finger in plan.second_fingers:
        idx = [order.index(j) for j in groups[finger]]
        q_pre[idx] = np.clip(np.asarray(plan.second_pre, float), lo, hi)[idx]
        q_closed[idx] = np.clip(np.asarray(plan.second_closed, float), lo, hi)[idx]
    command, arm = state.command, env.dofs(ARM_AND_TORSO)
    for alpha in np.linspace(0, 1, state.steps(.5)):
        env.step(posed(env, command, state.hand + alpha * (q_pre - state.hand)))
    state.hand = q_pre.copy()
    p = env.p
    saved, found = p.saveState(), None
    try:
        start = env.get_base_pose()
        heading = np.array([np.cos(start[2]), np.sin(start[2]), 0.])
        left = np.array([-np.sin(start[2]), np.cos(start[2]), 0.])
        bases = [start + (offset + forward) * heading + sideways * left
                 for forward, sideways in ((0., 0.), *state.options.base_steps)
                 for offset in env.scenario.standing_offsets]
        for base in bases:
            p.restoreState(saved)
            if np.linalg.norm(base - start) > 1e-9 and not _drive_clear(env, start, base):
                continue
            env.set_joints(dict(zip(FETCH_BASE_JOINTS, base, strict=True)))
            for turn in symmetries(env.second_spec, state.options.symmetry != "none"):
                T = env.body_transform(env.second_id) @ turn @ np.asarray(plan.second_pose, float)
                status, reach, _ = _reach(env, T, command[arm].copy(), rng, MORE_SECOND_RETREATS, touch=(env.second_id,),
                                          rrt_iters=state.options.rrt_iters)
                if status is None:
                    found = {**reach, "base": base}
                    break
            if found:
                break
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    if found is None:
        return "SECOND_UNREACHABLE"
    step_base(env, command, found["base"], q_pre)
    home = env.body_transform(env.second_id)[:3, 3].copy()
    pk.executor.follow_path(found["path"], base=posed(env, command, q_pre))
    command[arm] = found["q_goal"]
    if float(np.linalg.norm(env.body_transform(env.second_id)[:3, 3] - home)) > APPROACH_DISPLACEMENT_LIMIT:
        state.observe()
        return "SECOND_PUSHED"
    grasp = env.get_link_transform(env.hand_root_link)
    close = ContactStopClose(params.hand_config(env.hand_config), q_pre, q_closed, lo, hi, envelop=plan.first_fingers)
    hand = q_pre
    for alpha in np.linspace(0, 1, state.steps(params.ramp_seconds)):
        hand = close.command(alpha)
        env.step(posed(env, command, hand))
        close.observe(env.finger_contacts(env.second_id), env.hand_joint_values())
    for alpha in np.linspace(0, 1, state.steps(params.squeeze_seconds)):
        hand = close.squeeze(alpha)
        env.step(posed(env, command, hand))
    reference, start_z = in_palm(env, env.second_id), env.body_transform(env.second_id)[2, 3]
    pk.executor.move_to(found["q_lift"], steps=state.steps(1.), base=posed(env, command, hand))
    command[arm] = found["q_lift"]
    pk.executor.hold(state.steps(1.), base=posed(env, command, hand))
    now = in_palm(env, env.second_id)
    kept = (env.body_transform(env.second_id)[2, 3] - start_z >= LIFT - .04
            and np.linalg.norm(now[:3, 3] - reference[:3, 3]) <= SLIP_TRANSLATION
            and rotation_between_deg(reference, now) <= SLIP_ROTATION_DEG)
    state.pickup = pk
    if not kept:
        state.observe()
        return "SECOND_NOT_LIFTED"
    state.pick = {"command": command, "executor": pk.executor, "second": found, "second_grasp": grasp,
                  "open_second": q_pre, "first": None, "first_grasp": None,
                  "first_pick_open": np.clip(np.asarray(plan.first_pre, float), lo, hi)}
    state.hand = np.asarray(hand, float).copy()
    if state.rng is None:
        state.rng = np.random.default_rng([state.seed, 2])
    state.held = {"second"}
    state.observe()
    return None


def recover(state: TripState, controller=None) -> list:
    """After a failed trip, use what is left:
    1. an object still in the hand at the pick table is carried to the target table and set down;
    2. each object still standing on the pick table, the wrapped one first, is taken alone once
       (``pick_single``), carried and set down, driving back to the pick table first if needed. The the planned-stow recovery
       executor never retakes the wrapped object alone (recorded as ``LEFT_WRAPPED``);
    3. objects fallen, on the floor, or still held at the target table after a failed set-down are left.
    Every attempt runs in simulated time. Returns one record per attempt."""
    log = []
    state.observe()
    in_hand = [name for name in NAMES if state.where[name] == "in_hand"]
    if in_hand and state.at == "pick":
        if state.pick is None:
            hand_on(state)
        state.held = set(in_hand)
        code = carry(state)
        for name in ("second", "first"):
            if code is None and name in state.held:
                code = place(state, name)
        if code is None or code.endswith("PLACE_MISSED"):
            settle(state)
        log.append(_record(state, "deliver", in_hand, code))
    for k, name in enumerate(NAMES):
        state.observe()
        if state.where[name] != "standing_pick":
            continue
        if any(state.where[other] == "in_hand" for other in NAMES):
            log.append(_record(state, "single", [name], "HAND_NOT_EMPTY"))
            break
        if name == "first" and not RECOVERY[state.recovery]["retake_wrapped"]:
            log.append(_record(state, "skip", [name], "LEFT_WRAPPED"))
            continue
        if state.at == "target":
            drive_back(state)
        code = pick_single(state, name, controller, attempt=k)
        if code is None:
            postures = CARRY_POSTURES if RECOVERY[state.recovery].get("carry_postures") else ()
            code = carry(state, postures) or place(state, name)
        if code is None or code.endswith("PLACE_MISSED"):
            settle(state)
        log.append(_record(state, "single", [name], code))
    state.observe()
    return log


def _record(state, action, objects, code):
    return {"action": action, "objects": list(objects), "failure": code,
            "seconds": state.env.step_count * state.env.dt, "where": dict(state.where)}


def back_to_lift(state: TripState, hand, rise=.10) -> bool:
    """errand recovery: after a failed pickup, bring the arm back to where the first lift ended, whatever is in the hand carried along:
    the hand first rises ``rise`` (off the second object, if it is there), then a path planned with the held objects attached.
    ``hand`` is held throughout. False, with the arm where the rise left it, if no path is found; True if it is there."""
    env, pk, p = state.env, state.pickup, state.env.p
    if pk is None or getattr(pk, "first", None) is None or pk.command is None:
        return False
    arm, command = env.dofs(ARM_AND_TORSO), pk.command
    q_back = np.asarray(pk.first["q_lift"], float)
    if np.linalg.norm(env.get_joints(ARM_AND_TORSO) - q_back) <= 1e-2:
        return True
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    palm = env.get_link_transform(env.hand_root_link)
    up = ik.solve(palm[:3, 3] + [0, 0, rise], palm[:3, :3], q0=env.get_joints(ARM_AND_TORSO))
    if up.success:
        pk.executor.move_to(up.q, steps=state.steps(1.), base=posed(env, command, hand))
        command[arm] = up.q
    held = [state.bodies[name][0] for name in NAMES if state.where.get(name) == "in_hand"]
    saved = p.saveState()
    try:
        checker = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=held[0] if held else None)
        for body in held[1:]:
            checker.attach_also(body)
        rng = np.random.default_rng([state.seed, 5, 1])
        path = rrt_connect(env.get_joints(ARM_AND_TORSO), q_back, checker, ik.lower, ik.upper, max_iters=state.options.rrt_iters, rng=rng)
        path = None if path is None else shortcut(path, checker, n_iters=60, rng=rng)
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    if path is None:
        return False
    pk.executor.follow_path(path, base=posed(env, command, hand))
    command[arm] = q_back
    return True


def deliver(state: TripState) -> str | None:
    """Step 1 of errand recovery: set down on its spot whatever is still in the hand, wherever the hand is.
    - At the pick table it is carried over, the posture library's shapes tried if the lift's posture is refused (the posture recovery
      tried them only for a retake, and 5 of its deliveries on the long table ended ``CARRY_BLOCKED``).
    - Part-way along the drive, after the other object was lost, the drive goes on for what is left of it (the posture recovery drove
      the whole distance again from there).
    - At the target table, after a set-down that could not be planned or missed the other object, each object still
      held gets its set-down once more (fresh draws of the planner's random stream).
    Returns the first failure, or None; ``state.held`` and ``state.where`` say what is left."""
    env = state.env
    state.observe()
    in_hand = [name for name in NAMES if state.where[name] == "in_hand"]
    if not in_hand:
        return None
    if state.pick is None:
        hand_on(state)
    state.held = set(in_hand)
    code = None
    if state.at == "pick":
        profile, began = state.profile, None
        if state.carry is not None:
            # A carry began and lost an object on the way: go on from here for the rest of the drive.
            began = np.asarray(state.carry["route"][0], float)
            travelled = float(np.linalg.norm(env.get_base_pose()[:2] - began[:2]))
            if profile.distance - travelled > 1e-3:
                state.profile = replace(profile, distance=profile.distance - travelled)
        try:
            code = carry(state, CARRY_POSTURES)
            # A pickup that failed with the arm out at the second object: back to where the first lift ended, then carry.
            if code == "CARRY_BLOCKED" and state.carry is None and back_to_lift(state, state.hand):
                code = carry(state, CARRY_POSTURES)
        finally:
            state.profile = profile
        if code is None and began is not None:
            state.travel = state.carry["route"][-1] - began
    elif state.at == "target":
        # The set-down starts from the object as it sits in the palm now; its slip is judged from here, not from the carry's start
        # (a held object past those limits was already called lost, and would be again at once).
        state.references = {name: in_palm(env, state.bodies[name][0]) for name in in_hand}
        state.worst = {name: [0., 0.] for name in in_hand}
    for name in ("second", "first"):
        if code is None and name in state.held:
            code = place(state, name)
    if code is None or code.endswith("PLACE_MISSED"):
        settle(state)
    state.observe()
    return code

