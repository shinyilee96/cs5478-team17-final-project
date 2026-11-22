"""One errand on the robot: the planner picks a trip, the robot runs it, and after a failed two-object trip errand
recovery (``recover_errand``) takes over.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from ..control.grasp_controller import ScriptedClose, drive
from ..data.lineage import CandidateRecord
from ..envs.errand import LIBRARY_FIRST_Y
from ..grasp.grasp_vector import matrix_to_quat_xyzw
from ..perception.errand_look import look
from ..perception.fit import Fit
from ..sim.transport import MotionProfile
from ..sim.two_object import TwoObjectPlan
from .common import ARM_AND_TORSO, ArmExecutor
from .execution import execute_plan
from .feasibility import FeasibilityPlan, check_candidate
from .observed import PARK, ObservedPlanning, plan_frame, proxy_body
from .trip_stages import (
    CARRY_POSTURES,
    NAMES,
    TripState,
    back_to_lift,
    carry,
    deliver,
    drive_back,
    finish,
    hand_on,
    pick_pair,
    place,
    recover,
    settle,
    stow,
    withdraw,
)
from .two_object import HOLD, RELEASE_SECONDS, UPRIGHT_DEG, Failed, ReachOptions, dry_run, end_state, posed, second_approach, second_phase

PLANS_TRIED = 10 # at run time, an option's accepted plans tried by the dry run
PAIR_RELEASE, SINGLE_RELEASE = "v1", "v2"
SINGLE_POSTURE = "front_down_r0"
# Where a single object stands relative to the base, and which target spot it goes to: to the right (spot 0), else to the left.
SINGLE_STANCES = ((LIBRARY_FIRST_Y, 0), (-LIBRARY_FIRST_Y, 1))
STILL_THERE = .08 # an object stays available if it stands within this of its slot (the look's reach)
SETTLE_SECONDS = .5
SECOND_LOOK_SECONDS = 1. # the wait after the arm has stowed, on a trip with something still to count
PAIR_HOLD_SECONDS = 1. # the hold after the second lift


@dataclass
class ErrandRun:
    """What differs between configurations. ``vision``: "model" (estimated poses against the known model), "exact" (the
    same look, its estimates unused) or "fit" (fitted primitives only, ``kinds`` naming each role's primitive-shape library cell kind)."""
    decide: Callable # (standing, closed keys) -> Option or None
    plans: Callable # (option) -> [(plan index, TwoObjectPlan)], best first
    grasps: Callable # (slot, fit or None) -> ([CandidateRecord], cell kind or None)
    options: ReachOptions
    recovery: str | None = "postures"
    vision: str = "model"
    kinds: Callable | None = None # (option) -> {"first": kind, "second": kind}
    release: str = PAIR_RELEASE # the wrapped object's release on pair trips (``trip_stages.RELEASES``)
    values: Callable | None = None # (standing, closed keys) -> each open option's worth, for the trip's record
    hold: float | None = None # seconds held after the lift, on every trip; None is each trip's own (the sensitivity)
    from_here: bool = False # the any-pair counter: a pair's carry is as long as the drive from where the base is when
                            # it starts (after a step between the grasps); False measures it from the stance, as on the long table
    wide_look: float | None = None # the any-pair counter: a pair whose slots are more than this apart (m) gets a look at each object before
                                   # the pick, both timed; None is one look, as on the long table
    study: Callable | None = None # the any-pair counter: (option) -> True for an option the long table could already do; that option then
    study_options: ReachOptions | None = None # runs as on the long table: these reach options, the carry from the stance, one look
    yaw_starts: str = "box" # every look's yaw fit, ``estimate``'s ``starts``; "quarter" is the yaw fix
    found: frozenset | None = None # the round table: the objects the whole-table look found; the rest are never tried. None: every one
    knocks: InjectedKnocks | None = None # knocks injected after a trip that failed at the table (an assumption); None: none


def shell(env, y: float, start: np.ndarray, rest: np.ndarray, seed: int) -> TripState:
    """A ``TripState`` with no trip in it, which is all ``stow`` and ``drive_back`` need; ``home`` is stance ``y``."""
    state = TripState(env, None, MotionProfile(), seed=seed, recovery="postures")
    state.pickup = SimpleNamespace(executor=ArmExecutor(env), command=env.hold_action())
    state.result = {"diagnostics": {"timeline": {}}}
    state.rest, state.hand = rest, np.clip(np.asarray(env.hand_config["open_pose"], float), *env.hand_joint_limits())
    state.home = np.array([start[0], y, start[2]])
    return state


def go_to(env, y: float, start: np.ndarray, rest: np.ndarray, seed: int) -> None:
    """Stow the arm and drive straight to stance ``y`` on the trip's speed limits (``trip_stages.drive_back``, which goes to
    ``TripState.home``). An arm at rest and a base at its stance cost a hold only."""
    drive_back(shell(env, y, start, rest, seed))


def open_hand(env) -> None:
    """Whatever the trip left: the hand opens to its start pose where it is, and holds there."""
    lo, hi = env.hand_joint_limits()
    target, hand, command = np.clip(np.asarray(env.hand_config["open_pose"], float), lo, hi), env.hand_joint_values(), env.hold_action()
    for alpha in np.linspace(0, 1, round(RELEASE_SECONDS / env.dt)):
        env.step(env.action_with_hand(hand + alpha * (target - hand), command))
    for _ in range(round(SETTLE_SECONDS / env.dt)):
        env.step(env.action_with_hand(target, command))


def second_look(env, spots: dict, start: np.ndarray, rest: np.ndarray, seed: int) -> dict:
    """stow the arm as the next drive would, wait ``SECOND_LOOK_SECONDS`` and test each slot in ``spots`` (slot ->
    target spot) again. Reads the same test as every other check: ``env.set_down``."""
    state = shell(env, start[1], start, rest, seed)
    stow(state)
    state.pickup.executor.hold(state.steps(SECOND_LOOK_SECONDS), base=posed(env, state.command, state.hand))
    return {k: {**env.set_down(env.bodies[k], env.specs[k], spot), "where": end_state(env, env.bodies[k])["where"]} for k, spot in spots.items()}


def score(env, delivered: dict, upright: dict, spots: dict, start: np.ndarray, rest: np.ndarray, seed: int) -> dict:
    """The score of record, every configuration and practice alike. ``delivered`` and ``upright`` (slot -> bool) hold
    what the trip's own tests said, and change only for an object that passes the second look. The hand opens. A trip with
    nothing left to count ends there, as it did on the long table. Returns the second look's measures."""
    open_hand(env)
    waiting = {k: spots[k] for k in spots if not delivered[k]}
    looked = second_look(env, waiting, start, rest, seed) if waiting else {}
    for k, check in looked.items():
        if check["success"]:
            delivered[k], upright[k] = True, check["tilt_deg"] <= UPRIGHT_DEG
    return looked


def take_stock(env, standing: set) -> set:
    """What can still be fetched, from the simulator (labelled privileged). What stands on the counter near its slot stays;
    what has fallen on the counter stays there as an obstacle; anything else is taken out of the errand."""
    left = set()
    for k in sorted(standing):
        state = end_state(env, env.bodies[k])
        near = np.linalg.norm(np.asarray(state["position"][:2]) - env.slots[k]) <= STILL_THERE
        if state["where"] == "standing_pick" and near:
            left.add(k)
        elif state["where"] != "fallen_pick" and state["where"] != "standing_pick":
            env.park(k)
    return left


def deliver_held(state: TripState) -> list:
    """Recovery that retakes nothing, for objects the system has no model of (``recover``'s first step: its retake of a second
    object plans on the true pose). What is still in the hand at the pick table is carried over and set down."""
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
        log.append({"action": "deliver", "objects": in_hand, "failure": code})
    state.observe()
    return log


def second_stands(state: TripState) -> bool:
    """``place`` tests the second object 0.5 s after the hand rises, and a miss then need not be one. After a held
    second, does it pass, or stand on the target table? Then the wrapped object still in the hand goes on to its set-down."""
    settle(state)
    check = state.env.set_down(*state.bodies["second"])
    state.notes["second_miss_rechecked"] = {**check, "where": state.where["second"]}
    return check["success"] or state.where["second"] == "standing_target"


def pair_trip(env, plan: TwoObjectPlan, profile: MotionProfile, seed: int, options: ReachOptions, recovery: str | None,
              retake=True, release=PAIR_RELEASE, hold=PAIR_HOLD_SECONDS, context: SimpleNamespace | None = None,
              from_here: bool = False) -> dict:
    """The chain of ``run_trip_with_recovery``, on a shifted stance. ``recovery`` None stops at the failure. errand recovery needs
    ``context``: the errand's ``run``, ``start``, ``rest`` and the trip's ``slots`` (role -> slot). ``from_here`` (the any-pair counter): the
    carry ends where ``profile`` from the stance would, its length measured from where the base is (``from_here``)."""
    state = TripState(env, plan, profile, seed=seed, posture=HOLD, options=options, recovery=recovery or "postures",
                      release=release)
    if from_here:
        state.drive_whole = profile
    failure = pick_pair(state, hold)
    drive_from_here(state)
    if from_here:
        state.notes["carry_distance"] = state.profile.distance # for the record (the any-pair counter); only when asked, so the long table's records are as ever
    failure = failure or _rest_of_pair(state)
    result = finish(state, failure)
    if failure is None or recovery is None:
        return result
    if recovery == "errand":
        state.notes["recovery"] = recover_errand(state, context, hold)
    else:
        state.notes["recovery"] = recover(state) if retake else deliver_held(state)
    return finish(state, failure)


def drive_from_here(state: TripState) -> None:
    """the any-pair counter: if the trip was asked for it (``state.drive_whole``, the profile from the stance) and no carry has
    begun, the carry's profile is cut to the drive left from where the base is now, along the travel direction, so that a step
    between the grasps does not carry the robot past the target table. Reads the base's pose only; steps nothing. Without
    ``drive_whole`` (every earlier run) nothing changes."""
    whole = getattr(state, "drive_whole", None)
    if whole is None or state.home is None or state.carry is not None:
        return
    env = state.env
    yaw = float(state.home[2]) + env.scenario.travel_yaw
    along = float((env.get_base_pose()[:2] - np.asarray(state.home[:2], float)) @ np.array([np.cos(yaw), np.sin(yaw)]))
    if whole.distance - along > 1e-3:
        state.profile = replace(whole, distance=whole.distance - along)


def _rest_of_pair(state: TripState) -> str | None:
    """A pair trip after both are lifted: carry, the two set-downs and the withdrawal. Returns the first failure or None."""
    # The lift's posture holds the hand out over the counter, where a two-object scene has nothing along the drive and an
    # errand has the other objects, so the carry can be refused: the posture library's shapes come next.
    failure = carry(state, CARRY_POSTURES) or place(state, "second")
    went_on = failure == "SECOND_PLACE_MISSED" and second_stands(state)
    failure = None if went_on else failure
    failure = failure or place(state, "first") or withdraw(state)
    # ``withdraw`` says SECOND_KNOCKED of any second object off its spot; one that went on off its spot was missed, not knocked.
    return "SECOND_PLACE_MISSED" if went_on and failure == "SECOND_KNOCKED" else failure


# Errand recovery settings, fixed before any evaluation run.
ERRAND_RECOVERY = {
    "retry": True, # step 3: the second grasp once more, the wrapped object still held, from a new look
    "retry_posture": False, # step 2 before it: the look moves the arm, wrapped object and all, to the viewing posture;
                            # False looks with the arm where it is, the hand at the first lift
    "retry_lift": .10, # before going back to the first lift's arm pose, the hand rises this much (m) off the second
    "retake_other_stance": True, # step 4: if the stance that suits the object's spot sees or plans nothing, the other one
}


def recover_errand(state: TripState, context: SimpleNamespace, hold: float) -> list:
    """errand recovery after a failed pair trip, in this order:
    3 (first, when it applies, so that step 1 does not carry the wrapped object off alone): the wrapped object still held at
      the pick table and the second still standing there: look again at the second with the wrapped object in hand (step 2)
      and try the second grasp once more, with the same plan, from the new estimate; if both are then held, the trip goes on
      as a pair trip;
    1 whatever is still in the hand is set down on its spot, wherever the hand is (``trip_stages.deliver``);
    4 every trip object still standing on the counter is taken alone, with the grasp single trips use (the frozen order),
      from the stance that suits its spot, after a look of its own (step 2); an object still in the hand at the target
      (its set-down could not be planned) is first let go where it is;
    5 fallen objects stay.
    Every step runs in simulated time. Returns one record per step: action, objects, failure, ``began``, ``seconds`` (its
    end), ``metres`` driven, ``before`` and ``where`` (each object, in ``end_state``'s categories)."""
    env, log = state.env, []

    def record(action, objects, code, began, metres, before):
        state.observe()
        log.append({"action": action, "objects": list(objects), "failure": code, "began": began, "seconds": env.step_count * env.dt,
                    "metres": getattr(env, "odometer", 0.) - metres, "before": before, "where": dict(state.where)})

    def mark():
        state.observe()
        return env.step_count * env.dt, getattr(env, "odometer", 0.), dict(state.where)

    state.observe()
    if ERRAND_RECOVERY["retry"] and _can_retry(state, context):
        began, metres, before = mark()
        code, look = _retry_second(state, context, hold)
        record("retry", NAMES, code, began, metres, before)
        log[-1]["look"] = look
        drive_from_here(state) # the retry may have stepped the base (the any-pair counter); nothing otherwise
        if code is None:
            began, metres, before = mark()
            record("pair", NAMES, _rest_of_pair(state), began, metres, before)
    state.observe()
    if any(state.where[name] == "in_hand" for name in NAMES):
        began, metres, before = mark()
        held = [name for name in NAMES if state.where[name] == "in_hand"]
        drive_from_here(state)
        record("deliver", held, deliver(state), began, metres, before)
    for name in NAMES:
        state.observe()
        if state.where[name] != "standing_pick":
            continue
        if any(state.where[other] == "in_hand" for other in NAMES):
            began, metres, before = mark()
            open_hand(env)
            record("let_go", [other for other in NAMES if before[other] == "in_hand"], None, began, metres, before)
        began, metres, before = mark()
        code, details = _retake(state, context, name)
        record("single", [name], code, began, metres, before)
        log[-1]["retake"] = details
    state.observe()
    return log


def _can_retry(state: TripState, context) -> bool:
    """Step 3's condition: a pickup failure at the pick table, the wrapped object in the hand, the second standing near its
    slot, the first phase complete (its context exists) and a model of the second to plan on."""
    env, pk = state.env, state.pickup
    if context is None or context.run.vision == "fit" or state.at != "pick" or state.carry is not None or state.pick is not None:
        return False
    if pk is None or getattr(pk, "ctx", None) is None or pk.command is None:
        return False
    near = np.linalg.norm(env.body_transform(env.second_id)[:2, 3] - np.asarray(env.second_xy)) <= STILL_THERE
    # Held as the pickup judges it: within the slip limits of where the wrapped object settled after its lift.
    return (state.where["first"] == "in_hand" and state.where["second"] == "standing_pick" and bool(near)
            and pk.kept(pk.ctx["reference"], env.object_id))


def _retry_second(state: TripState, context, hold: float) -> tuple[str | None, dict]:
    """Steps 2 and 3 of errand recovery: the second grasp's fingers open (the wrapping ones keep their grip), the hand rises off the
    second object and the arm goes back to where the first lift ended, the wrapped object in hand; one look at the second;
    then ``second_approach`` and ``second_phase`` again, planned on the new estimate with a fresh random stream. The trip's
    own result is kept; the retry's goes in its record. Returns its failure (None: both held, ``state.pick`` set) and the
    look's record."""

    env, pk = state.env, state.pickup
    first = env.object_id
    q_pre = pk.ctx["q_pre"]
    hand = np.asarray(getattr(pk, "last_hand", q_pre), float)
    for alpha in np.linspace(0, 1, state.steps(.5)):
        env.step(pk.posed(hand + alpha * (q_pre - hand)))
    if not back_to_lift(state, q_pre, ERRAND_RECOVERY["retry_lift"]):
        return "RETRY_NO_WAY_BACK", {}
    if not pk.kept(pk.ctx["reference"], first):
        return "RETRY_FIRST_LOST_BACK", {}
    run = context.run
    spec = None if run.vision == "fit" else env.second_spec
    # The look holds the grip as commanded: ``env.hold_action`` is where the joints are, and a hand told to stay where its
    # fingers stand lets the wrapped object slide out.
    seen = look(env, {"second": (env.second_id, spec, env.second_xy)}, np.random.default_rng([state.seed, 5, 2]), held=first,
                posture=ERRAND_RECOVERY["retry_posture"], action=posed(env, pk.command, q_pre), yaw_starts=getattr(run, "yaw_starts", "box"))
    looked = {"seconds": seen.seconds, "posture": seen.posture, "seen": bool(seen.poses), "errors": seen.errors.get("second")}
    if run.vision == "model" and env.second_id in seen.poses:
        env.perceived = {**(env.perceived or {}), env.second_id: seen.poses[env.second_id]}
    if not pk.kept(pk.ctx["reference"], first):
        return "RETRY_FIRST_LOST_LOOK", looked
    trip_result, trip_seed = pk.result, pk.seed
    pk.result = {"both_held": False, "clean": False, "first_failure": None, "diagnostics": {"timeline": {}}}
    pk.seed = [state.seed, 5, 3]
    try:
        ctx = second_approach(pk, pk.ctx)
        if isinstance(ctx, Failed):
            return pk.result["first_failure"], looked
        controller = ScriptedClose()
        controller.reset()
        result, pick = drive(second_phase(pk, ctx, hold, controller.params), controller)
        looked["retry_diagnostics"] = {k: v for k, v in result["diagnostics"].items() if k != "timeline"}
        if pick is None:
            return result["first_failure"], looked
    finally:
        pk.result, pk.seed = trip_result, trip_seed
    pk.ctx = ctx
    state.pick, state.held, state.hand = pick, set(NAMES), pick["grip"]
    if state.rng is None:
        state.rng = np.random.default_rng([state.seed, 2])
    return None, looked


def _retake(state: TripState, context, name: str) -> tuple[str | None, dict]:
    """Step 4 of errand recovery: ``name`` alone, as a single trip takes it (``_single``: drive to the stance, a look of its own, the
    grasps, the single trip's set-down), onto the spot of its role in the pair so that the trip is scored as it stands; from
    the stance that suits that spot, then (``retake_other_stance``) the other. The hand opens first. The trip's own objects are
    presented again afterwards. Returns the failure (None if it was delivered) and what the single trip recorded."""
    env, run = state.env, context.run
    i, j = context.slots["first"], context.slots["second"]
    k, other = (i, j) if name == "first" else (j, i)
    spot = NAMES.index(name)
    stances = [s for s in SINGLE_STANCES if s[1] == spot]
    stances += [(offset, spot) for offset, s in SINGLE_STANCES if s != spot] if ERRAND_RECOVERY["retake_other_stance"] else []
    open_hand(env)
    seed = state.seed * 10 + 1 + spot
    perceived = env.perceived
    try:
        env.perceived = None
        result = _single(env, run, k, stances, context.start, context.rest, seed, other=other)
    finally:
        env.present(i, j)
        env.perceived = perceived
    details = {key: result.get(key) for key in ("stance", "tried", "candidate", "refusals", "look_errors", "look_seconds", "failure", "refused")}
    if "refused" in result:
        return result["refused"], details
    return (None if result["delivered"][k] else result["failure"] or "SINGLE_NOT_DELIVERED"), details


def _planned(env, candidate, profile, seed, pose, proxy) -> FeasibilityPlan:
    """``check_candidate`` on the estimated pose: the object stands there for planning, or its proxy (id, frame) does in
    its place. Both go back with the restore: a proxy left standing would be a solid body in the physics."""
    p, body = env.p, env.object_id
    saved = p.saveState()
    try:
        if pose is not None:
            p.resetBasePositionAndOrientation(body, pose[:3, 3], matrix_to_quat_xyzw(pose[:3, :3]))
        if proxy is not None:
            shape, frame = proxy
            p.resetBasePositionAndOrientation(body, PARK.tolist(), [0., 0., 0., 1.])
            p.resetBasePositionAndOrientation(shape, frame[:3, 3], matrix_to_quat_xyzw(frame[:3, :3]))
            env.object_id = shape
        return check_candidate(env, candidate, profile, seed=seed, posture=SINGLE_POSTURE)
    finally:
        env.object_id = body
        p.restoreState(saved)
        p.removeState(saved)


def single_trip(env, candidates, profile, spot, seed, pose=None, proxy=None, hold=None) -> dict | None:
    """``run_single_trip`` on a shifted stance: the first grasp the planner accepts through to the release is executed.
    None if the planner accepts none. Its timeline is in seconds, as a pair trip's."""
    env.goal_pos = np.array([*env.goal_spots[spot][:2], env.scenario.table_height + env.first_spec.support_offset_z])
    refusals = []
    for rank, candidate in enumerate(candidates):
        plan = _planned(env, candidate, profile, seed, pose, proxy)
        if not (plan.execution and plan.placement):
            refusals.append(plan.failure or plan.diagnostics.get("placement_failure"))
            continue
        trial = execute_plan(env, candidate, plan, profile, scenario_id="errand", seed=seed, method="errand", release=SINGLE_RELEASE, hold_seconds=hold)
        placement = trial.diagnostics.get("placement", {})
        return {"delivered": bool(trial.placement), "first_failure": trial.first_failure, "candidate": rank, "refusals": refusals,
                "upright": bool(trial.placement and placement.get("tilt_deg", 180.) <= UPRIGHT_DEG), "placement": placement,
                "timeline": {phase: step * env.dt for phase, step in trial.diagnostics.get("timeline", {}).items()}}
    return None


def _look_at(env, run: ErrandRun, roles: dict, seed: int):
    """One look at the trip's objects; ``roles``: role -> (body, spec, slot)."""
    targets = {role: (body, None if run.vision == "fit" else spec, slot) for role, (body, spec, slot) in roles.items()}
    return look(env, targets, np.random.default_rng([seed, 7]), yaw_starts=getattr(run, "yaw_starts", "box"))


def _look_each(env, run: ErrandRun, roles: dict, seed: int):
    """the any-pair counter: a pair too wide for one image gets a look at each object, in role order, from the trip's stance;
    the first with ``_look_at``'s stream, the second with its own. One ``Look`` of both: poses, fits, errors and points joined,
    seconds summed."""
    looks = [look(env, {role: (body, None if run.vision == "fit" else spec, slot)}, np.random.default_rng([seed, 7] if k == 0 else [seed, 7, k]),
                  yaw_starts=getattr(run, "yaw_starts", "box"))
             for k, (role, (body, spec, slot)) in enumerate(roles.items())]
    joined = looks[0]
    for other in looks[1:]:
        joined.poses.update(other.poses)
        joined.fits.update(other.fits)
        joined.errors.update(other.errors)
        joined.points.update(other.points)
        joined.seconds += other.seconds
        joined.posture = joined.posture and other.posture
    return joined


def look_products(seen, kinds: dict) -> dict:
    """What a look without a model made of each target, for the trip's record and the unseen-object clip: its
    points (thinned, to the millimetre), the fitted shape and the primitive kind it mapped to. The plan is the trip's ``plan``."""
    return {role: {"points": np.round(seen.points[role], 3).tolist(), "kind": fit.kind, "sizes": fit.sizes.tolist(), "centre": fit.centre.tolist(), "yaw": float(fit.yaw),
                   "residual": float(fit.residual), "cell_kind": kinds.get(role)} for role, fit in seen.fits.items()}


def attempt_pair(env, run: ErrandRun, option, start, rest, seed) -> dict:
    """Stance, look, dry runs, trip. ``{"refused": reason}`` if no plan could start: the option costs the drive and look."""
    if getattr(run, "study", None) is not None and run.study(option):
        # the any-pair counter: an option the long table could already do runs exactly as on the long table's code runs it, recovery included.
        run = replace(run, options=run.study_options, from_here=False, wide_look=None, study=None, study_options=None)
    stance = env.present(option.i, option.j)
    go_to(env, stance, start, rest, seed)
    roles = {"first": (env.object_id, env.first_spec, env.first_xy), "second": (env.second_id, env.second_spec, env.second_xy)}
    wide = run.wide_look is not None and abs(env.first_xy[1] - env.second_xy[1]) > run.wide_look + 1e-9
    seen = _look_each(env, run, roles, seed) if wide else _look_at(env, run, roles, seed)
    record = {"stance": stance, "look_seconds": seen.seconds, "look_errors": seen.errors}
    if wide:
        record["looks"] = 2
    if (run.vision == "fit" and set(seen.fits) != set(NAMES)) or (run.vision != "fit" and len(seen.poses) != 2):
        return {**record, "refused": "NOT_PERCEIVED"}
    view = None
    if run.vision == "model":
        env.perceived = seen.poses
    kinds = run.kinds(option) if run.vision == "fit" else None
    if kinds is not None:
        record["look_products"] = look_products(seen, kinds)
    refusals, chosen = [], None
    try:
        for k, plan in run.plans(option)[:PLANS_TRIED]:
            if run.vision == "fit":
                view = ObservedPlanning.build(env, seen.fits, kinds, held_identity=True)
                env.observed = view
            try:
                refusal = dry_run(env, plan, seed, run.options)
            finally:
                if view is not None:
                    env.observed = None
                    view.remove()
                    view = None
            if refusal is None:
                chosen = (k, plan)
                break
            refusals.append(refusal)
        if chosen is None:
            return {**record, "refused": "NO_PLAN", "refusals": refusals}
        if run.vision == "fit":
            view = ObservedPlanning.build(env, seen.fits, kinds)
            env.observed = view
        profile = MotionProfile(distance=env.scenario.target_xy[1] - stance)
        context = SimpleNamespace(run=run, start=start, rest=rest, slots={"first": option.i, "second": option.j}) if run.recovery == "errand" else None
        result = pair_trip(env, chosen[1], profile, seed, run.options, run.recovery, retake=run.vision != "fit", release=run.release,
                           hold=PAIR_HOLD_SECONDS if run.hold is None else run.hold, context=context, from_here=run.from_here)
    finally:
        env.perceived, env.observed = None, None
        if view is not None:
            view.remove()
    notes = result["diagnostics"]
    # the any-pair counter only (these keys exist only when its options are on): the drive between the grasps and the carry's length.
    step_notes = {key: notes[key] for key in ("second_step", "second_base_step", "carry_distance") if key in notes and "carry_distance" in notes}
    return {**record, "plan": chosen[0], "refusals": refusals, "failure": result["first_failure"], "delivered": dict(zip((option.i, option.j), result["delivered"], strict=True)),
            "upright": dict(zip((option.i, option.j), result["upright"], strict=True)), "recovery": notes.get("recovery"),
            "end_state": {k: v["where"] for k, v in notes["end_state"].items()}, "timeline": notes["timeline"],
            "spots": {option.i: env.goal_spots[0].tolist(), option.j: env.goal_spots[1].tolist()}, "carry": notes.get("carry"),
            "second_miss_rechecked": notes.get("second_miss_rechecked"), **step_notes}


def attempt_single(env, run: ErrandRun, option, start, rest, seed) -> dict:
    """One object: from the stance that puts it to the right, else the one that puts it to the left."""
    return _single(env, run, option.i, SINGLE_STANCES, start, rest, seed)


def _single(env, run: ErrandRun, i: int, stances, start, rest, seed, other=None) -> dict:
    """``attempt_single`` from the given ``stances`` ((offset, target spot), tried in order); ``other`` is the slot presented
    beside ``i`` (default the next one), which only names the scene's second object."""
    record = {"tried": []}
    env.present(i, (i + 1) % len(env.bodies) if other is None else other)
    for offset, spot in stances:
        stance = env.slots[i][1] - offset
        go_to(env, stance, start, rest, seed)
        seen = _look_at(env, run, {"first": (env.object_id, env.first_spec, env.first_xy)}, seed)
        record.update(stance=stance, look_seconds=record.get("look_seconds", 0.) + seen.seconds, look_errors=seen.errors)
        fit: Fit | None = seen.fits.get("first")
        if (run.vision == "fit" and fit is None) or (run.vision != "fit" and not seen.poses):
            record["tried"].append("NOT_PERCEIVED")
            continue
        candidates, kind = run.grasps(i, fit)
        if fit is not None:
            record["look_products"] = look_products(seen, {"first": kind})
        pose = seen.poses.get(env.object_id) if run.vision == "model" else None
        proxy = None
        if run.vision == "fit":
            frame = plan_frame(fit, kind, env.scenario.table_height)
            proxy = (proxy_body(env.p, fit, frame), frame)
        profile = MotionProfile(distance=env.scenario.target_xy[1] - stance)
        try:
            result = single_trip(env, candidates, profile, spot, seed, pose, proxy, run.hold)
        finally:
            if proxy is not None:
                env.p.removeBody(proxy[0])
        if result is not None:
            return {**record, "failure": result["first_failure"], "candidate": result["candidate"], "refusals": result["refusals"],
                    "delivered": {i: result["delivered"]}, "upright": {i: result["upright"]}, "spots": {i: env.goal_spots[spot].tolist()},
                    "placement": result["placement"], "timeline": result["timeline"]}
        record["tried"].append("NO_GRASP")
    return {**record, "refused": "NO_GRASP"}


def run_errand(env, run: ErrandRun, seed: int, record_trip: Callable | None = None) -> dict:
    """One errand from ``env.reset``'s scene. Returns what happened to each object and when."""
    start, rest = env.get_base_pose(), env.get_joints(ARM_AND_TORSO)
    standing, closed, refused = set(range(len(env.bodies)) if run.found is None else run.found), set(), set()
    delivered, upright, trips, last_delivery = set(), set(), [], 0.

    def count(k, is_upright):
        nonlocal last_delivery
        delivered.add(k)
        upright.update({k} if is_upright else set())
        last_delivery = env.step_count * env.dt
        env.park(k)
        standing.discard(k)

    injected = getattr(run, "knocks", None)
    starts = {k: env.p.getBasePositionAndOrientation(b) for k, b in enumerate(env.bodies)} if injected is not None else None
    while True:
        option = run.decide(standing, closed | refused)
        if option is None:
            break
        worth = run.values(standing, closed | refused) if run.values is not None else None
        trip_seed = seed * 100 + len(trips)
        began = env.step_count * env.dt
        env.note("trip_start", trip=option.kind, i=option.i, j=option.j)
        attempt = attempt_single if option.kind == "single" else attempt_pair
        result = attempt(env, run, option, start, rest, trip_seed)
        entry = {"kind": option.kind, "i": option.i, "j": option.j, "began": began, "ended": env.step_count * env.dt, **result}
        if worth is not None:
            entry["decision"] = {"standing": sorted(standing), "closed": sorted(map(str, closed | refused)), "drive": option.drive, "odds": option.codes, "options": worth}
        trips.append(entry)
        if "refused" in result:
            refused.add(option.key)
            env.note("trip_end", refused=result["refused"])
            continue
        closed.add(option.key)
        refused.clear()
        before = set(standing)
        entry["first_look"] = {"delivered": dict(result["delivered"]), "upright": dict(result["upright"])}
        if env.check_log is not None:
            entry["ended_at"] = {k: end_state(env, env.bodies[k]) for k in option.objects}
        for k, ok in result["delivered"].items():
            if ok:
                count(k, result["upright"][k])
        looked = score(env, result["delivered"], result["upright"], result["spots"], start, rest, seed * 100 + len(trips))
        for k, check in looked.items():
            if check["success"]:
                count(k, result["upright"][k])
        entry["second_look"] = looked
        standing = take_stock(env, standing)
        if injected is not None:
            entry["injected"] = inject_knocks(env, injected, option, result, standing, seed, len(trips) - 1, starts)
            standing = take_stock(env, standing)
        entry["ended"] = env.step_count * env.dt
        entry["knocked"] = sorted(before - option.objects - standing)
        if env.check_log is not None:
            entry["checks"], env.check_log = env.check_log, []
        env.note("trip_end", delivered=[k for k, ok in result["delivered"].items() if ok], failure=result.get("failure"))
        if record_trip is not None:
            record_trip(entry)
    go_to(env, start[1], start, rest, seed * 100 + len(trips))
    return {"objects": len(env.bodies), "delivered": sorted(delivered), "upright": sorted(upright), "seconds": env.step_count * env.dt,
            "last_delivery": last_delivery, "trips": trips, "left": sorted(standing), "metres": getattr(env, "odometer", 0.)}


KNOCK_STREAM = 47
KNOCK_SPIN = 6. # rad/s about the horizontal axis that tips an object away from the failed trip's objects
KNOCK_PUSH = .3 # m/s along the same direction
KNOCK_SETTLE = 1. # s of physics after the push (the robot holds still)
STAND_UP_SETTLE = .5 # s after an object is stood up again
SINGLE_TABLE_FAILURES = ("APPROACH_DISTURBED", "MOUNTED_LIFT_FAIL", "TUCK_FAIL") # a single trip's failures at the execution stage


@dataclass(frozen=True)
class InjectedKnocks:
    """After a trip that failed at the table (a pair's pickup failure with a code in ``pair_codes``, a single's in
    ``SINGLE_TABLE_FAILURES``), each object still standing within ``radius`` of where the trip's objects stood is toppled with chance
    ``p``, a draw of its own. Toppled for real: a push and a spin, then the physics. Severity ``lost``: it stays fallen, out of the
    errand; otherwise it is stood up again at its start pose after ``seconds`` (the robot waits), an injected stand-up."""
    p: float
    radius: float
    pair_codes: frozenset
    lost: bool = True
    seconds: float = 0.


def _idle(env, seconds: float) -> None:
    command, hand = env.hold_action(), env.hand_joint_values()
    for _ in range(round(seconds / env.dt)):
        env.step(env.action_with_hand(hand, command))


def topple(env, k: int, away_from) -> bool:
    """Tip slot ``k``'s object over, away from the point ``away_from`` (xy): a push and a spin about the horizontal axis, then
    ``KNOCK_SETTLE`` s of physics. If it still stands, it is laid on its side there and settles. Returns whether the push alone did it."""
    body = env.bodies[k]
    position, turn = env.p.getBasePositionAndOrientation(body)
    u = np.asarray(position[:2], float) - np.asarray(away_from, float)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1., 0.])
    axis = np.array([-u[1], u[0], 0.])
    env.p.resetBaseVelocity(body, [*(KNOCK_PUSH * u), 0.], (KNOCK_SPIN * axis).tolist())
    _idle(env, KNOCK_SETTLE)
    if end_state(env, body)["where"] != "standing_pick":
        return True
    lying = env.p.multiplyTransforms([0, 0, 0], env.p.getQuaternionFromAxisAngle(axis.tolist(), np.pi / 2), [0, 0, 0],
                                     env.p.getBasePositionAndOrientation(body)[1])[1]
    position = env.p.getBasePositionAndOrientation(body)[0]
    env.p.resetBasePositionAndOrientation(body, position, lying)
    low = env.p.getAABB(body)[0][2]
    env.p.resetBasePositionAndOrientation(body, [position[0], position[1], position[2] + env.scenario.table_height + .002 - low], lying)
    env.p.resetBaseVelocity(body, [0., 0., 0.], [0., 0., 0.])
    _idle(env, KNOCK_SETTLE)
    return False


def inject_knocks(env, knocks: InjectedKnocks, option, result: dict, standing: set, seed: int, trip: int, starts: dict) -> dict:
    """the injected knocks of one trip (``InjectedKnocks``). Returns the trip's record of them."""
    failure = result.get("failure")
    table = failure in (knocks.pair_codes if option.kind == "pair" else SINGLE_TABLE_FAILURES)
    record = {"failed_at_table": bool(table), "near": [], "draws": {}, "toppled": [], "pushed_over": {}, "stood_up": []}
    if not table:
        return record
    origins = [np.asarray(env.positions[x], float) for x in sorted(option.objects)]
    for k in sorted(standing - option.objects):
        xy = np.asarray(env.p.getBasePositionAndOrientation(env.bodies[k])[0][:2], float)
        if min(np.linalg.norm(xy - o) for o in origins) > knocks.radius + 1e-9:
            continue
        record["near"].append(k)
        u = float(np.random.default_rng([seed, KNOCK_STREAM, trip, k]).uniform())
        record["draws"][k] = u
        if u >= knocks.p:
            continue
        nearest = min(origins, key=lambda o: np.linalg.norm(xy - o))
        env.note("knock_injected", slot=k)
        record["pushed_over"][k] = topple(env, k, nearest)
        record["toppled"].append(k)
    if not knocks.lost and record["toppled"]:
        _idle(env, knocks.seconds * len(record["toppled"]))
        for k in record["toppled"]:
            env.p.resetBasePositionAndOrientation(env.bodies[k], *starts[k])
            env.p.resetBaseVelocity(env.bodies[k], [0., 0., 0.], [0., 0., 0.])
            env.note("stood_up_injected", slot=k)
            record["stood_up"].append(k)
        _idle(env, STAND_UP_SETTLE)
    return record


BLOCK_REACH = .36 # m: the likely blockers of an option are the other found objects this close to either of its objects
BLOCK_SEED = 91


def blocking(env, run: ErrandRun, options, seed: int, poses: dict, turns: dict | None = None) -> dict:
    """the round table: which options can start now, and which must wait for a blocker. For each option (key -> record), with the planners on
    the look's estimated ``poses`` (body -> 4x4) for the option's own objects and every other object at its true pose, no time
    passing: a pair's first ``PLANS_TRIED`` plans are dry-run from its stance, a single's grasps from its stances. Refused, its likely
    blockers (found objects within ``BLOCK_REACH`` of either of its objects, by the estimates in ``env.slots``) are taken off the table and
    the dry run repeats; then the nearest blocker whose removal alone is enough, else all of them, are what the option needs gone.
    Record: ``open`` (None: refused even with its blockers gone), ``needs_gone`` (object indices), ``plan`` (the rank of the first plan
    or grasp accepted). The scene, base and presentation are put back as they were. ``turns`` (body -> [4x4,...]): other turns the look
    cannot rule out; a refused option is dry-run again with each turn of its own objects, one at a time, before its blockers count."""
    p = env.p
    saved, hold = p.saveState(), env._hold.copy()
    found = sorted(run.found if run.found is not None else range(len(env.bodies)))
    out = {}
    try:
        for option in options:
            def test(gone: tuple[int, ...]) -> int | None:
                p.restoreState(saved)
                for k in gone:
                    p.resetBasePositionAndOrientation(env.bodies[k], [4.5, -2. + .4 * k, .15], [0., 0., 0., 1.])
                for variant in _variants(env, option, poses, turns or {}):
                    rank = _accepted(env, run, option, seed * 100 + BLOCK_SEED, variant)
                    if rank is not None:
                        return rank
                    p.restoreState(saved)
                    for k in gone:
                        p.resetBasePositionAndOrientation(env.bodies[k], [4.5, -2. + .4 * k, .15], [0., 0., 0., 1.])
                return None
            rank, gone = test(()), ()
            if rank is None:
                near = sorted((min(float(np.linalg.norm(np.subtract(env.slots[k], env.slots[o]))) for o in option.objects), k)
                              for k in found if k not in option.objects)
                blockers = tuple(k for d, k in near if d <= BLOCK_REACH)
                if blockers and (rank := test(blockers)) is not None:
                    gone = blockers
                    for k in blockers if len(blockers) > 1 else ():
                        alone = test((k,))
                        if alone is not None:
                            rank, gone = alone, (k,)
                            break
            out[option.key] = {"open": None if rank is None else not gone, "needs_gone": list(gone), "plan": rank}
    finally:
        p.restoreState(saved)
        p.removeState(saved)
        env._hold = hold
        env.perceived = None
        env.present(0, 1)
    return out


def _variants(env, option, poses: dict, turns: dict) -> list[dict]:
    """``poses`` as estimated, then with each other turn of the option's own objects, one at a time (the second case)."""
    out = [poses]
    for k in sorted(option.objects):
        body = env.bodies[k]
        out += [{**poses, body: turned} for turned in turns.get(body, [])]
    return out


def _accepted(env, run: ErrandRun, option, seed: int, poses: dict) -> int | None:
    """The rank of the first plan (pair) or grasp (single) the planner accepts for ``option`` in the scene as it stands, or None."""
    if option.kind == "pair":
        env.stand_at(env.present(option.i, option.j))
        env.perceived = {b: poses[b] for b in (env.object_id, env.second_id) if b in poses}
        try:
            for rank, (_, plan) in enumerate(run.plans(option)[:PLANS_TRIED]):
                if dry_run(env, plan, seed, run.options) is None:
                    return rank
        finally:
            env.perceived = None
        return None
    i = option.i
    env.present(i, (i + 1) % len(env.bodies))
    candidates, _ = run.grasps(i, None)
    for offset, spot in SINGLE_STANCES:
        stance = env.slots[i][1] - offset
        env.stand_at(stance)
        env.goal_pos = np.array([*env.goal_spots[spot][:2], env.scenario.table_height + env.first_spec.support_offset_z])
        profile = MotionProfile(distance=env.scenario.target_xy[1] - stance)
        for rank, candidate in enumerate(candidates):
            plan = _planned(env, candidate, profile, seed, poses.get(env.object_id), None)
            if plan.execution and plan.placement:
                return rank
    return None


def known_grasps(spec, entries) -> list[CandidateRecord]:
    """A single-grasp library's entries, as ``run_single_trip`` takes them."""
    return [CandidateRecord(spec.object_id, "privileged", tuple(e["grasp"]), "sim-closure", "sim-closure") for e in entries]
