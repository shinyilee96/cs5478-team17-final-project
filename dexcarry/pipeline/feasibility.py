"""Geometric checks before execution: reach, approach and lift, and each carry posture's route and placement."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..data.lineage import CandidateRecord
from ..envs.fetch import FETCH_BASE_JOINTS
from ..grasp.grasp_vector import grasp_in_world, invert_transform, make_transform, unpack_grasp
from ..kin.collision import PlanningSession, PyBulletCollisionChecker
from ..kin.ik import DLSIK
from ..kin.rrt import rrt_connect, shortcut
from ..sim.loads import load_summary, specific_load, subsample
from ..sim.transport import MotionProfile, base_trajectory
from .common import ARM_AND_TORSO
from .postures import DEFAULT, posture_candidates

TUCK_STEPS = 150
PLACEMENT_RRT_ITERS = 300


@dataclass
class FeasibilityPlan:
    execution: bool = False
    placement: bool = False
    failure: str | None = None
    path: list = field(default_factory=list)
    q_grasp: np.ndarray | None = None
    q_lift: np.ndarray | None = None
    q_carry: np.ndarray | None = None
    q_place_pre: np.ndarray | None = None
    q_place: np.ndarray | None = None
    T_palm_object: np.ndarray | None = None
    T_world_object: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    posture: str = DEFAULT
    # (T,3) predicted palm-frame load in g over reorientation and driving; None if never planned.
    predicted_load: np.ndarray | None = None
    # Base pose (x, y, yaw) to pick from; None means the pose the episode started in.
    base_standing: np.ndarray | None = None


def check_candidate(env, candidate: CandidateRecord, profile: MotionProfile, *, seed=0, rrt_iters=500,
                    posture: str = DEFAULT) -> FeasibilityPlan:
    return check_candidate_postures(env, candidate, profile, [posture], seed=seed, rrt_iters=rrt_iters)[posture]


def check_candidate_postures(env, candidate: CandidateRecord, profile: MotionProfile, postures, *, seed=0,
                             rrt_iters=500) -> dict[str, FeasibilityPlan]:
    postures = list(postures)
    if not postures or len(set(postures)) != len(postures):
        raise ValueError("unique carry postures required")
    p = env.p
    state = p.saveState()
    try:
        start = env.get_base_pose()
        heading = np.array([np.cos(start[2]), np.sin(start[2]), 0.])
        best, attached, base = None, None, start
        for offset in env.scenario.standing_offsets:
            p.restoreState(state)
            base = start + offset * heading
            if offset and not _drive_clear(env, start, base):
                continue
            env.set_joints(dict(zip(FETCH_BASE_JOINTS, base, strict=True)))
            pick, attached = _plan_pick(env, candidate, seed, rrt_iters)
            pick.diagnostics["standing_offset"] = float(offset)
            pick.base_standing = base
            # Keep the attempt that got furthest: a collision is closer to a plan than no reach.
            if attached is not None or best is None or (best.failure, pick.failure) == ("IK_UNREACHABLE", "APPROACH_COLLISION"):
                best = pick
            if attached is not None:
                break
        if attached is None:
            return {name: replace(best, posture=name, diagnostics=dict(best.diagnostics)) for name in postures}
        return {name: _plan_carry(env, replace(best, posture=name, diagnostics=dict(best.diagnostics)),
                                  attached, profile, base, seed) for name in postures}
    finally:
        p.restoreState(state)
        p.removeState(state)


def _drive_clear(env, start, goal, samples=6) -> bool:
    """Can the base move straight from ``start`` to ``goal`` with the arm as it is, touching nothing?"""
    checker = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS)
    return not any(checker(start + a * (goal - start)) for a in np.linspace(0, 1, samples))


def pregrasp_hand(env, joints):
    """Finger joints slightly opened from the grasp, shared by planning and execution."""
    cfg = env.hand_config
    flex = np.array([j in cfg["flexion_joints"] for j in cfg["joint_order"]])
    lo, hi = env.hand_joint_limits()
    return np.clip(np.asarray(joints, float) - .15 * flex, lo, hi)


def _plan_pick(env, candidate, seed, rrt_iters):
    result = FeasibilityPlan()
    rng = np.random.default_rng(seed)
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    q_now = env.get_joints(ARM_AND_TORSO)
    T_obj = env.get_object_transform()
    T = grasp_in_world(np.asarray(candidate.grasp), T_obj)
    _, _, joints = unpack_grasp(np.asarray(candidate.grasp), 16)
    lo, hi = env.hand_joint_limits()
    if np.any(joints < lo) or np.any(joints > hi):
        result.failure = "IK_UNREACHABLE"
        result.diagnostics["reason"] = "hand joint limits"
        return result, None
    # Plan with the opened pre-shape the hand actually wears while it moves in.
    env.set_joints(dict(zip(env.hand_joints, pregrasp_hand(env, joints), strict=True)))
    normal = np.asarray(env.hand_config.get("palm_axes", {}).get("normal", [0, 0, -1]))
    pre = T.copy()
    pre[:3, 3] -= .08 * (T[:3, :3] @ normal)
    rpre = ik.solve_multi(pre[:3, 3], pre[:3, :3], [q_now, ik.rest, rng.uniform(ik.lower, ik.upper)])
    grasp = ik.solve(T[:3, 3], T[:3, :3], q0=rpre.q)
    lift = ik.solve(T[:3, 3] + [0, 0, .14], T[:3, :3], q0=grasp.q)
    if not all(r.success for r in (rpre, grasp, lift)):
        result.failure = "IK_UNREACHABLE"
        return result, None
    # The transit to the pre-grasp may not touch the object at all; only the final straight
    # approach lets the hand meet it. Otherwise the arm can push the object and grasp it elsewhere.
    transit = PyBulletCollisionChecker(env, ARM_AND_TORSO)
    path = rrt_connect(q_now, rpre.q, transit, ik.lower, ik.upper, max_iters=rrt_iters, rng=rng)
    checker = PyBulletCollisionChecker(env, ARM_AND_TORSO)
    checker.allowed_contacts[env.object_id] = env.hand_link_indices
    approach = [rpre.q + a * (grasp.q - rpre.q) for a in np.linspace(0, 1, 40)]
    if path is None or any(checker(q) for q in approach):
        result.failure = "APPROACH_COLLISION"
        return result, None
    path = shortcut(path, transit, n_iters=60, rng=rng)
    result.path, result.q_grasp, result.q_lift = path + approach, grasp.q, lift.q
    result.T_palm_object, result.T_world_object = invert_transform(T) @ T_obj, T_obj
    checker._set(grasp.q)
    attached = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=env.object_id)
    attached.attachment_transform = result.T_palm_object
    lift_path = [grasp.q + a * (lift.q - grasp.q) for a in np.linspace(.1, 1, 40)]
    if any(attached(q) for q in lift_path):
        result.failure = "APPROACH_COLLISION"
        return result, None
    return result, attached


def _plan_carry(env, result, attached, profile, base, seed=0):
    env.set_joints(dict(zip(FETCH_BASE_JOINTS, base, strict=True)))
    attached.allowed_support = set()
    lift = result.q_lift
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    palms = []
    for name, arm in posture_candidates(result.posture):
        q = np.r_[lift[0], arm]
        if np.any(q < ik.lower) or np.any(q > ik.upper):
            continue
        carry, palms, free = [lift + a * (q - lift) for a in np.linspace(0, 1, 80)], [], True
        for v in carry:
            if attached(v):
                free = False
                break
            palms.append(env.get_link_transform(env.hand_root_link))
        if free:
            result.q_carry = q
            result.diagnostics["carry_pose"] = name
            break
    if result.q_carry is None:
        result.failure = "TUCK_FAIL"
        return result
    result.execution = True
    attached._set(result.q_carry)
    times, trajectory = base_trajectory(base, profile, env.dt, env.scenario.travel_yaw)
    stride = max(1, len(trajectory) // 100)
    route = []
    for pose in trajectory[::stride]:
        env.set_joints(dict(zip(FETCH_BASE_JOINTS, pose, strict=True)))
        if attached(result.q_carry):
            result.execution = False
            result.failure = "TUCK_FAIL"
            result.diagnostics["reason"] = "carry route collision"
            return result
        route.append(env.get_link_transform(env.hand_root_link))
    result.predicted_load = _predicted_load(result, palms, route, times[::stride], env.dt)
    result.diagnostics["predicted_load"] = {**load_summary(result.predicted_load),
                                            "trajectory_g": result.predicted_load.tolist()}
    env.set_joints(dict(zip(FETCH_BASE_JOINTS, trajectory[-1], strict=True)))
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    T_goal = make_transform(result.T_world_object[:3, :3], env.goal_pos) @ invert_transform(result.T_palm_object)
    # The target table mirrors the pick table, so the lift configuration is the natural starting guess:
    # it makes the release retrace the checked lift-to-carry motion in reverse.
    pre_goal = ik.solve_multi(T_goal[:3, 3] + [0, 0, .10], T_goal[:3, :3], [result.q_lift, result.q_carry, ik.rest])
    goal = ik.solve(T_goal[:3, 3], T_goal[:3, :3], q0=pre_goal.q)
    if not pre_goal.success or not goal.success:
        result.diagnostics["placement_failure"] = "PLACEMENT_UNREACHABLE"
        return result
    # Execution plans this move again, from the object as it is actually held.
    if plan_placement_approach(attached, result.q_carry, pre_goal.q, goal.q, ik.lower, ik.upper,
                               np.random.default_rng(seed), support={env.target_table_id}) is None:
        result.diagnostics["placement_failure"] = "PLACEMENT_COLLISION"
        return result
    result.placement = True
    result.q_place_pre, result.q_place = pre_goal.q, goal.q
    return result


def plan_placement_approach(checker, q_start, q_above, q_goal, lower, upper, rng, support=()):
    """Arm path from the carry posture to above the goal, then straight down onto it; None if blocked.

    The straight joint-space move is tried first, then RRT-Connect around the robot and the
    tables. Only the final descent may bring the held object into contact with ``support``.
    """
    q_above, q_goal = np.asarray(q_above, float), np.asarray(q_goal, float)
    with PlanningSession(checker):
        path = rrt_connect(q_start, q_above, checker, lower, upper, max_iters=PLACEMENT_RRT_ITERS, rng=rng)
        if path is None:
            return None
        path = shortcut(path, checker, n_iters=60, rng=rng)
        allowed = set(checker.allowed_support)
        checker.allowed_support = allowed | set(support)
        blocked = any(checker(q_above + a * (q_goal - q_above)) for a in np.linspace(0, 1, 40))
        checker.allowed_support = allowed
    return None if blocked else path


def _predicted_load(result, palms, route, route_times, dt):
    """Quasi-static gravity through the arm reorientation, then gravity plus base inertia."""
    offset = result.T_palm_object[:3, 3]
    palms, route = np.asarray(palms), np.asarray(route)
    tuck = specific_load(np.linspace(0, TUCK_STEPS * dt, len(palms)), palms[:, :3, 3], palms[:, :3, :3],
                         quasi_static=True)
    points = route[:, :3, 3] + route[:, :3, :3] @ offset
    drive = specific_load(route_times, points, route[:, :3, :3], quasi_static=len(route) < 3)
    return np.vstack([subsample(tuck, 16), subsample(drive, 32)])
