"""Arm execution of a selected grasp: approach, close and lift, then the carry and placement."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..data.lineage import TrialRecord
from ..envs.fetch import FETCH_BASE_JOINTS
from ..grasp.grasp_vector import invert_transform, make_transform
from ..kin.collision import PyBulletCollisionChecker
from ..kin.ik import DLSIK
from ..sim.closing import RAMP_SECONDS, SQUEEZE_SECONDS, ContactStopClose
from ..sim.feedback import SlipGuard
from ..sim.loads import load_summary, rotation_between_deg, subsample
from ..sim.transport import CarryMonitor, execute_transport
from .common import ARM_AND_TORSO, ArmExecutor
from .feasibility import TUCK_STEPS, FeasibilityPlan, plan_placement_approach, pregrasp_hand

APPROACH_DISPLACEMENT_LIMIT = 0.01
STANDING_SPEED = 0.1
# Release "v2" of the single-object trip, as the two-object set-down: opening time, then a hold.
RELEASE_V2_SECONDS = .5
RELEASE_V2_SETTLE = .3


def execute_plan(env, candidate, plan: FeasibilityPlan, profile, *, scenario_id, seed, method,
                 provenance=None, feedback: str | None = None, release: str = "v1", hold_seconds: float | None = None):
    """``release`` is the set-down: "v1" as every result before 24 Oct (the hand snaps to its open pose); "v2" the two-object set-down's: the base first stands where the palm is as far from it as at the grasp, then the
    hand opens to the preshape over ``RELEASE_V2_SECONDS`` and holds before backing off. ``hold_seconds`` is the hold after
    the lift; None is its 100 steps."""
    if release not in ("v1", "v2"):
        raise ValueError("release is v1 or v2")
    stage, diagnostics = {}, {"pre_execution": plan.diagnostics, "posture": plan.posture,
                              "predicted_placement": bool(plan.placement), "timeline": {}}

    def mark(phase):
        # Control step at which this phase ended; recordings keep one frame every four steps.
        diagnostics["timeline"][phase] = int(env.step_count)

    def finish(failure=None):
        return TrialRecord(candidate.candidate_id, scenario_id, seed, method, "pybullet", **stage,
                           first_failure=failure, protocol="end_to_end",
                           scenario={"environment": asdict(env.scenario), "motion": asdict(profile),
                                     "posture": plan.posture, "feedback": feedback,
                                     "standing_offset": plan.diagnostics.get("standing_offset", 0.)},
                           diagnostics=diagnostics, provenance=provenance or {})

    if not plan.execution:
        # No motion plan exists: recorded separately from physical failures in reports.
        stage["execution"] = False
        diagnostics["planner_rejected"] = True
        return finish(plan.failure or "IK_UNREACHABLE")
    executor = ArmExecutor(env)
    joints = np.asarray(candidate.grasp)[9:]
    cfg = env.hand_config
    lo, hi = env.hand_joint_limits()
    flex = np.array([j in cfg["flexion_joints"] for j in cfg["joint_order"]])
    pre = pregrasp_hand(env, joints)
    target = np.clip(joints + cfg["close_delta"] * flex, lo, hi)
    # One commanded target per joint, updated as the pick moves on. Reading the measured joints
    # back as targets at every step would let the arm sink under gravity while the fingers work.
    command = env.hold_action()
    arm = env.dofs(ARM_AND_TORSO)

    def posed(hand_q):
        return env.action_with_hand(hand_q, command)

    for _ in range(40):
        env.step(posed(pre))
    rest = env.object_position()
    if plan.base_standing is not None:
        # Step up to the standing pose the planner chose, at walking pace, before reaching.
        base, here = env.dofs(FETCH_BASE_JOINTS), env.get_base_pose()
        steps = int(np.linalg.norm(plan.base_standing[:2] - here[:2]) / (STANDING_SPEED * env.dt))
        for k in range(1, steps + 1):
            command[base] = here + (plan.base_standing - here) * k / steps
            env.step(posed(pre))
        for _ in range(30 if steps else 0):
            env.step(posed(pre))
    executor.follow_path(plan.path, base=posed(pre))
    command[arm] = plan.path[-1]
    mark("reached")
    # A grasp only counts as this candidate's grasp if the object is still where it was observed.
    pushed = float(np.linalg.norm(env.object_position() - rest))
    diagnostics["approach_displacement"] = pushed
    if pushed > APPROACH_DISPLACEMENT_LIMIT:
        stage["execution"] = False
        return finish("APPROACH_DISTURBED")
    start_z = env.object_position()[2]
    # The same contact-stop close as the floating lift test, for every selector method.
    close = ContactStopClose(cfg, env.hand_joint_values(), target, lo, hi)
    proposed = invert_transform(env.get_link_transform(env.hand_root_link)) @ env.get_object_transform()
    for a in np.linspace(0, 1, round(RAMP_SECONDS / env.dt)):
        env.step(posed(close.command(a)))
        close.observe(env.finger_contacts(env.object_id), env.hand_joint_values())
    for a in np.linspace(0, 1, round(SQUEEZE_SECONDS / env.dt)):
        env.step(posed(close.squeeze(a)))
    target = close.squeeze(1.)
    mark("closed")
    # Where the palm and the base are at the grasp, for release "v2"'s base correction.
    grasp_palm, grasp_base = env.get_link_transform(env.hand_root_link)[:3, 3].copy(), command[env.dofs(FETCH_BASE_JOINTS)].copy()
    closed = invert_transform(env.get_link_transform(env.hand_root_link)) @ env.get_object_transform()
    diagnostics["closing"] = {"rotation_deg": rotation_between_deg(proposed, closed),
                              "displacement": float(np.linalg.norm(closed[:3, 3] - proposed[:3, 3])),
                              "fingers": sorted(close.halted)}
    executor.move_to(plan.q_lift, steps=100, base=posed(target))
    command[arm] = plan.q_lift
    executor.hold(100 if hold_seconds is None else round(hold_seconds / env.dt), base=posed(target))
    mark("lifted")
    height = float(env.object_position()[2] - start_z)
    diagnostics["mounted_lift_height"] = height
    if height < .08:
        stage["execution"] = False
        return finish("MOUNTED_LIFT_FAIL")
    stage["execution"] = True

    monitor = CarryMonitor(env)
    guard = SlipGuard.for_env(env, feedback) if feedback else None
    q_start = env.get_joints(ARM_AND_TORSO)
    hold = posed(target)
    for k in range(1, TUCK_STEPS + 1):
        a = hold if guard is None else guard(monitor, hold)[0]
        a = np.array(a, copy=True)
        a[arm] = q_start + (plan.q_carry - q_start) * k / TUCK_STEPS
        if not monitor.step(a):
            break
    hold[arm] = plan.q_carry
    mark("carry_posture")
    reorientation = monitor.result()
    if reorientation["retained"]:
        transport = execute_transport(env, profile, hold, monitor=monitor, feedback=guard)
        transport["phase"] = "drive"
        mark("arrived")
    else:
        transport = {**reorientation, "requested": asdict(profile), "measured_features": None, "phase": "reorientation"}
    loads = monitor.loads()
    diagnostics["transport"] = transport
    diagnostics["carry"] = {"reorientation": reorientation,
                            "measured_load": load_summary(loads) if loads is not None else None,
                            "measured_load_trajectory_g": subsample(loads, 48).tolist() if loads is not None else None,
                            "feedback": guard.summary() if guard else None}
    stage["transport"] = transport["retained"]
    if not stage["transport"]:
        if transport["phase"] == "reorientation":
            return finish("REORIENT_SLIP")
        return finish("TRANSPORT_DROP" if transport["dropped"] else "TRANSPORT_SLIP")

    # Release by reversing the pick: the palm returns to its grasp orientation, shifted so the
    # object as actually held (it may have slid or pivoted while closing) comes down on the goal.
    # How upright it ends is measured afterwards, not assumed here.
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    actual_relative = invert_transform(env.get_link_transform(env.hand_root_link)) @ env.get_object_transform()
    palm_rotation = (plan.T_world_object @ invert_transform(plan.T_palm_object))[:3, :3]
    held_rotation = palm_rotation @ actual_relative[:3, :3]
    lowest = float((env.scenario.object_spec().local_corners() @ held_rotation.T)[:, 2].min())
    goal = np.array([env.goal_pos[0], env.goal_pos[1], env.table_top_z() - lowest + .003])
    T_goal = make_transform(palm_rotation, goal - palm_rotation @ actual_relative[:3, 3])
    diagnostics["in_hand_rotation_deg"] = rotation_between_deg(plan.T_palm_object, actual_relative)
    if release == "v2":
        diagnostics["base_correction"] = _stand_for_release(env, T_goal[:3, 3], grasp_palm, grasp_base, target)
    current = env.get_joints(ARM_AND_TORSO)
    seeds = [plan.q_place_pre if plan.q_place_pre is not None else plan.q_lift, plan.q_lift, current, ik.rest]
    pre_goal = ik.solve_multi(T_goal[:3, 3] + [0, 0, .10], T_goal[:3, :3], seeds)
    goal = ik.solve(T_goal[:3, 3], T_goal[:3, :3], q0=pre_goal.q)
    if not pre_goal.success or not goal.success:
        stage["placement"] = False
        return finish("PLACEMENT_UNREACHABLE")
    # Plan the move from the carry posture to above the goal around the robot and the tables, with
    # the object where it actually sits in the hand. The straight joint-space move is tried first.
    checker = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=env.object_id)
    approach = plan_placement_approach(checker, current, pre_goal.q, goal.q, ik.lower, ik.upper,
                                       np.random.default_rng(seed), support={env.target_table_id})
    if approach is None:
        stage["placement"] = False
        return finish("PLACEMENT_COLLISION")
    diagnostics["placement_approach_waypoints"] = len(approach)
    executor.follow_path(approach, base=env.action_with_hand(target))
    executor.move_to(goal.q, base=env.action_with_hand(target))
    if release == "v2":
        # Reverse the pick: open to the preshape gradually (an instant opening flicks thin objects over), hold, back off.
        for a in np.linspace(0, 1, round(RELEASE_V2_SECONDS / env.dt)):
            env.step(executor.action_for(goal.q, env.action_with_hand(target + a * (pre - target))))
        executor.hold(round(RELEASE_V2_SETTLE / env.dt), base=env.action_with_hand(pre))
        executor.move_to(pre_goal.q, base=env.action_with_hand(pre))
    else:
        opened = executor.action_for(goal.q, env.action_with_hand(cfg["open_pose"]))
        for _ in range(100):
            env.step(opened)
        executor.move_to(pre_goal.q, base=env.action_with_hand(cfg["open_pose"]))
    executor.hold(120)
    mark("released")
    placement = env.evaluate()
    diagnostics["placement"] = placement
    stage["placement"] = placement["success"]
    if placement["success"]:
        return finish(None)
    if not placement["released"]:
        return finish("PLACEMENT_RELEASE_FAIL")
    on_target = placement["dist_to_goal"] <= env.scenario.target_tolerance and placement["supported"]
    return finish("PLACEMENT_TIPPED" if on_target and not placement["upright"] else "PLACEMENT_TARGET_MISS")


def _stand_for_release(env, palm_goal, grasp_palm, grasp_base, hand_q) -> dict:
    """Release "v2": step the base so that the palm's release position is where the palm was, seen from the base, at
    the grasp (the two-object set-down's correction, ``trip_stages.place``). Skipped if the base has turned since the
    grasp or the move would collide."""
    base, p = env.dofs(FETCH_BASE_JOINTS), env.p
    command = env.hold_action()
    here = command[base].copy()
    if abs(here[2] - grasp_base[2]) > 1e-3:
        return {"moved": 0., "skipped": "turned"}
    stand = here.copy()
    stand[:2] = palm_goal[:2] - (grasp_palm[:2] - grasp_base[:2])
    checker = PyBulletCollisionChecker(env, FETCH_BASE_JOINTS, attached_body=env.object_id)
    saved = p.saveState()
    clear = not any(checker(here + a * (stand - here)) for a in np.linspace(0, 1, 8)[1:])
    p.restoreState(saved)
    p.removeState(saved)
    if not clear:
        return {"moved": 0., "skipped": "collision"}
    steps = int(np.linalg.norm(stand[:2] - here[:2]) / (STANDING_SPEED * env.dt))
    for k in range(1, steps + 1):
        command[base] = here + (stand - here) * k / steps
        env.step(env.action_with_hand(hand_q, command))
    for _ in range(30 if steps else 0):
        env.step(env.action_with_hand(hand_q, command))
    return {"moved": float(np.linalg.norm(stand[:2] - here[:2]))}
