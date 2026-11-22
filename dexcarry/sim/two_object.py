"""Two objects in one hand on the floating LEAP hand: the side wrap (ring finger against the palm), then a pinch or
tripod grasp. The plan search here uses true geometry (privileged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..data.catalog import ObjectSpec
from ..grasp.grasp_vector import (
    invert_transform,
    make_transform,
    matrix_to_quat_xyzw,
    transform_from_pos_quat_xyzw,
)
from .bullet_floating import BulletFloatingHarness
from .closing import RAMP_SECONDS, SQUEEZE_SECONDS, ContactStopClose
from .loads import (
    SLIP_ROTATION_DEG,
    SLIP_TRANSLATION,
    anchor_friction,
    load_directions,
    rotation_between_deg,
    set_contact_friction,
)

# How the first object is taken: where it must start and which fingers close on it.
FIRST_GRASPS = {"wrap": ("table", ("ring",)), "hook": ("rack", ("middle", "ring"))}
SECOND_GRASPS = {"pinch": ("thumb", "index"), "tripod": ("thumb", "index", "middle")}
SETTLE_SECONDS = 1.0
LIFT = 0.14
# Lowest point of the LEAP hand below the palm origin when the ring side points down (metres).
HAND_LOW_EDGE = 0.114


@dataclass(frozen=True)
class TwoObjectPlan:
    """One two-object grasp. Poses are palm poses in each object's own frame."""

    first_id: str
    second_id: str
    first_pose: list
    first_pre: list
    first_closed: list
    second_pose: list
    second_pre: list
    second_closed: list
    first_fingers: tuple = ("ring",)
    second_fingers: tuple = ("thumb", "index", "middle")
    support: str = "table"
    # Where the first object settled in the palm frame when the search lifted it; None in older records.
    first_held: list | None = None

    @classmethod
    def from_dict(cls, data: dict) -> TwoObjectPlan:
        data = dict(data)
        if "hook_pose" in data:
            # Records written before the side wrap existed: a hook from a rack, then a pinch.
            data = {"support": "rack", **{key.replace("hook_", "first_").replace("pinch_", "second_"): value
                                          for key, value in data.items()}}
        return cls(**{**data, "first_fingers": tuple(data["first_fingers"]), "second_fingers": tuple(data["second_fingers"])})


def lying_rotation(half_extents) -> np.ndarray:
    """Rotation that lays a box down with its longest axis along world y and its shortest vertical."""
    order = np.argsort(half_extents) # shortest, middle, longest
    R = np.zeros((3, 3))
    R[2, order[0]], R[0, order[1]], R[1, order[2]] = 1., 1., 1.
    if np.linalg.det(R) < 0:
        R[0] *= -1
    return R


def perturb(T, rng, position_std=0., angle_std_deg=0.) -> np.ndarray:
    """Pose error of the hand: a shift in the world and a small turn about the palm origin."""
    T = np.array(T, float)
    if position_std > 0:
        T[:3, 3] += rng.normal(0, position_std, 3)
    if angle_std_deg > 0:
        axis = rng.normal(size=3)
        T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(rng.normal(0, angle_std_deg))).as_matrix()
    return T


def _wrap_joints(rng, cfg, fingers, closed_joints):
    order, groups = cfg["joint_order"], cfg["finger_groups"]
    pre, closed = np.zeros(16), np.zeros(16)
    for finger in fingers:
        mcp, rot, pip, dip = (order.index(j) for j in groups[finger])
        # Start slightly bent back so the finger clears the object, then curl fully.
        pre[[mcp, pip, dip]] = rng.uniform(-.3, .1), .1, .1
        pre[rot] = closed[rot] = rng.uniform(-.15, .15)
        closed[[mcp, pip, dip]] = closed_joints
    return pre, closed


def sample_hook(rng, cfg, thickness, fingers=("middle", "ring")):
    """Centre of a lying object in a level, palm-down palm frame, and open and closed hook joints.

    The object lies across the fingers near the knuckles, below the open fingers by its half
    thickness plus a gap.
    """
    centre = np.array([rng.uniform(.005, .06), rng.uniform(-.05, .03), -(thickness + rng.uniform(.02, .04))])
    return (centre, *_wrap_joints(rng, cfg, fingers, (2.0, 1.4, 1.0)))


def sample_heading(rng, windows_deg=None) -> float:
    """Heading of the open fingers in the horizontal plane (radians): anywhere, or inside one of
    ``windows_deg``, (from, to) pairs in degrees, drawn in proportion to their width."""
    if not windows_deg:
        return float(rng.uniform(0, 2 * np.pi))
    windows = np.asarray(windows_deg, float)
    if windows.ndim != 2 or windows.shape[1] != 2 or np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("heading windows are (from, to) pairs in degrees with from < to")
    low, high = windows[rng.choice(len(windows), p=np.diff(windows, axis=1)[:, 0] / np.diff(windows, axis=1).sum())]
    return float(np.radians(rng.uniform(low, high)) % (2 * np.pi))


def sample_side_wrap(rng, cfg, half_width, centre_height, fingers=("ring",), headings_deg=None):
    """Centre of a standing object in the palm frame of a hand turned on its side, the hand's
    heading, and open and closed joints of the wrapping fingers.

    The palm origin sits just high enough for the ring side of the hand to clear the table;
    the object stands in front of the palm, beyond the open fingers, near their base.
    ``headings_deg`` limits the heading to windows an arm can reach (see ``sample_heading``).
    """
    centre = np.array([rng.uniform(-.03, .06), centre_height - (HAND_LOW_EDGE + rng.uniform(.003, .02)),
                       -(half_width + rng.uniform(.045, .075))])
    return (centre, sample_heading(rng, headings_deg), *_wrap_joints(rng, cfg, fingers, (2.0, 1.6, 1.4)))


class TwoObjectHarness(BulletFloatingHarness):
    """Floating hand, the first object on a table or a rack, and a second object on the table."""

    def __init__(self, first: ObjectSpec, second: ObjectSpec, *, support="table", rack_height=.25,
                 second_position=(.45, 0.), paths=None, dt=1 / 240):
        if support not in ("table", "rack"):
            raise ValueError("the first object starts on the table or on a rack")
        if support == "rack" and first.visual_mesh:
            raise ValueError("the rack is only defined for box-shaped first objects")
        super().__init__(first, paths=paths, dt=dt)
        self.second_spec, self.support, self.rack_height = second, support, float(rack_height)
        self.second_position = tuple(second_position)
        self._first, self._both, self.posts = None, None, []
        if support == "rack":
            self.lying = lying_rotation(first.half_extents)
            extents = np.abs(self.lying @ np.asarray(first.half_extents))
            self.first_thickness = float(extents[2])
            p = self.p
            try:
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.012, .008, self.rack_height / 2])
                self.posts = [p.createMultiBody(0, shape, basePosition=[0, y, self.rack_height / 2])
                              for y in (-(extents[1] - .008), extents[1] - .008)]
            except BaseException:
                self.close()
                raise


    def _reset_scene(self):
        """First object standing on the table or lying on the rack, second on the table, open hand parked above."""
        p = self.p
        for name in ("object", "second"):
            if hasattr(self, name):
                p.removeBody(getattr(self, name))
        p.setGravity(0, 0, -9.81)
        self._first = self._both = None
        park = make_transform(np.eye(3), [0, 0, .8])
        self._place_palm(park)
        if self.support == "rack":
            self.object = self.spec.create_body(p, [0, 0, self.rack_height + self.first_thickness + .001],
                                                matrix_to_quat_xyzw(self.lying))
        else:
            self.object = self.spec.create_body(p, [0, 0, self.spec.support_offset_z + .002])
        self.second = self.second_spec.create_body(p, [*self.second_position, self.second_spec.support_offset_z + .002])
        anchor_friction(p, self.object)
        anchor_friction(p, self.second)
        q_open = np.clip(self.cfg["open_pose"], self.lower, self.upper)
        for j, q in zip(self.joints, q_open, strict=True):
            p.resetJointState(self.hand, j, q)
        for _ in range(240):
            self._control(q_open, park)
        return self._pose(self.object)

    def set_condition(self, contact_friction=None, dt=None):
        """Simulator setting for every later trial: friction between hand and objects, and the time step."""
        if contact_friction is not None:
            if self.spec.friction != self.second_spec.friction:
                raise ValueError("one hand coefficient cannot give the same contact friction on objects that differ")
            set_contact_friction(self.p, self.hand, contact_friction, self.spec.friction)
        if dt is not None:
            self.dt = float(dt)
            self.p.setTimeStep(self.dt)

    def _pose(self, body):
        return transform_from_pos_quat_xyzw(*self.p.getBasePositionAndOrientation(body))

    def _in_palm(self, body):
        return invert_transform(self._palm()) @ self._pose(body)

    def _kept(self, reference, body) -> bool:
        now = self._in_palm(body)
        return bool(np.linalg.norm(now[:3, 3] - reference[:3, 3]) <= SLIP_TRANSLATION
                    and rotation_between_deg(reference, now) <= SLIP_ROTATION_DEG)

    def _touching(self, body) -> set[str]:
        links = {c[3] for c in self.p.getContactPoints(self.hand, body)}
        return {f for f, names in self.cfg["finger_groups"].items() if links & set(self._finger_links(names))}

    def _forces(self, body) -> dict[str, float]:
        contacts = self.p.getContactPoints(self.hand, body)
        return {finger: sum(c[9] for c in contacts if c[3] in self._finger_links(names))
                for finger, names in self.cfg["finger_groups"].items()}

    def _penetrates(self, body, depth) -> bool:
        return any(c[8] < -depth for c in self.p.getContactPoints(self.hand, body))

    def _dofs(self, finger):
        return [self.cfg["joint_order"].index(j) for j in self.cfg["finger_groups"][finger]]

    def _move(self, q, start, goal, seconds):
        """Carry the hand from one pose to another at constant speed, with fixed finger targets."""
        turn = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], goal[:3, :3]]))
        for alpha in np.linspace(0, 1, max(2, round(seconds / self.dt))):
            self._control(q, make_transform(turn(alpha).as_matrix(), start[:3, 3] + alpha * (goal[:3, 3] - start[:3, 3])))

    def _hold(self, q, T, seconds):
        for _ in range(round(seconds / self.dt)):
            self._control(q, T)

    def level_hook_pose(self, centre_in_palm) -> np.ndarray:
        """Palm pose in the first object's frame for a level, palm-down hand with the object's
        centre at ``centre_in_palm`` (palm frame). Call after the scene has been reset."""
        first = self._pose(self.object)
        return invert_transform(first) @ make_transform(np.eye(3), first[:3, 3] - np.asarray(centre_in_palm, float))

    def side_wrap_pose(self, centre_in_palm, yaw) -> np.ndarray:
        """Palm pose in the first object's frame for a hand on its side: thumb side up, ring finger
        lowest, fingers pointing along ``yaw`` in the horizontal plane, palm facing the object."""
        first = self._pose(self.object)
        x, y = np.array([np.cos(yaw), np.sin(yaw), 0.]), np.array([0., 0., 1.])
        R = np.stack([x, y, np.cross(x, y)], axis=1)
        return invert_transform(first) @ make_transform(R, first[:3, 3] - R @ np.asarray(centre_in_palm, float))


    def take_first(self, palm_in_first, pre, closed, fingers=("ring",), *, noise=None) -> str | None:
        """Wrap ``fingers`` around the first object without the thumb and lift it. Returns a failure code or None."""
        p = self.p
        T = self._pose(self.object) @ np.asarray(palm_in_first, float)
        if noise is not None:
            T = noise(T)
        pre, closed = (np.clip(np.asarray(v, float), self.lower, self.upper) for v in (pre, closed))
        self._place_palm(T)
        for j, q in zip(self.joints, pre, strict=True):
            p.resetJointState(self.hand, j, q)
        p.performCollisionDetection()
        if self._penetrates(self.object, .003) or any(self._penetrates(b, .001) for b in (*self.posts, self.table, self.second)):
            return "FIRST_START_COLLISION"
        close = ContactStopClose(self.cfg, pre, closed, self.lower, self.upper, envelop=fingers)
        for alpha in np.linspace(0, 1, round(RAMP_SECONDS / self.dt)):
            self._control(close.command(alpha), T)
        self._hold(closed, T, SQUEEZE_SECONDS)
        if not self._touching(self.object) & set(fingers):
            return "FIRST_NO_CONTACT"
        start_z = self._pose(self.object)[2, 3]
        up = T.copy()
        up[2, 3] += LIFT
        self._move(closed, T, up, 1.)
        self._hold(closed, up, SETTLE_SECONDS)
        reference = self._in_palm(self.object)
        self._hold(closed, up, 2.)
        if self._pose(self.object)[2, 3] - start_z < LIFT - .04 or not self._kept(reference, self.object):
            return "FIRST_NOT_LIFTED"
        if not self._touching(self.object) <= set(fingers):
            return "FIRST_OTHER_FINGER"
        self._first = {"target": closed, "pose": up, "reference": reference, "fingers": tuple(fingers)}
        return None

    def take_second(self, palm_in_second, pre, closed, fingers=("thumb", "index", "middle"), *, hold_seconds=5.,
                    noise=None, clear_of_table=False) -> dict:
        """With the first object in hand, travel to the second, grasp it with ``fingers`` and lift both.

        The floating hand can press an open finger onto the table and still grasp; an arm planner
        refuses a pose in which the hand is in the table. ``clear_of_table`` rejects such grasps."""
        if self._first is None:
            raise RuntimeError("take the first object before the second")
        p, held = self.p, self._first
        if set(fingers) & set(held["fingers"]):
            raise ValueError("a finger cannot hold both objects")
        self._both = None
        q_pre, q_closed = held["target"].copy(), held["target"].copy()
        for finger in fingers:
            idx = self._dofs(finger)
            q_pre[idx], q_closed[idx] = np.asarray(pre, float)[idx], np.asarray(closed, float)[idx]
        q_pre, q_closed = np.clip(q_pre, self.lower, self.upper), np.clip(q_closed, self.lower, self.upper)
        home = self._pose(self.second)[:3, 3].copy()
        goal = self._pose(self.second) @ np.asarray(palm_in_second, float)
        if noise is not None:
            goal = noise(goal)
        above = goal.copy()
        above[2, 3] += .12
        self._move(q_pre, held["pose"], above, 2.)
        self._move(q_pre, above, goal, 1.)
        result = {"both_held": False, "clean": False, "first_failure": None}
        if held["reference"] is not None and not self._kept(held["reference"], self.object):
            result["first_failure"] = "FIRST_LOST_MOVING"
            return result
        if np.linalg.norm(self._pose(self.second)[:3, 3] - home) > .01:
            result["first_failure"] = "SECOND_PUSHED"
            return result
        if clear_of_table and p.getContactPoints(self.hand, self.table):
            result["first_failure"] = "SECOND_TABLE_CONTACT"
            return result
        # The fingers on the first object keep wrapping; the others stop where they touch the second.
        close = ContactStopClose(self.cfg, q_pre, q_closed, self.lower, self.upper, envelop=held["fingers"])
        for alpha in np.linspace(0, 1, round(RAMP_SECONDS / self.dt)):
            self._control(close.command(alpha), goal)
            close.observe(self._forces(self.second), [p.getJointState(self.hand, j)[0] for j in self.joints])
        for alpha in np.linspace(0, 1, round(SQUEEZE_SECONDS / self.dt)):
            self._control(close.squeeze(alpha), goal)
        grip = close.squeeze(1.)
        reference, start_z = self._in_palm(self.second), self._pose(self.second)[2, 3]
        lifted = goal.copy()
        lifted[2, 3] += LIFT
        self._move(grip, goal, lifted, 1.)
        self._hold(grip, lifted, hold_seconds)
        first = held["reference"] is None or self._kept(held["reference"], self.object)
        second = self._pose(self.second)[2, 3] - start_z >= LIFT - .04 and self._kept(reference, self.second)
        result.update(first_kept=bool(first), second_kept=bool(second), both_held=bool(first and second),
                      clean=bool(self._touching(self.object) <= set(held["fingers"]) and self._touching(self.second) <= set(fingers)))
        if not result["both_held"]:
            result["first_failure"] = "FIRST_LOST_ON_SECOND_GRASP" if second else "SECOND_NOT_LIFTED" if first else "BOTH_LOST"
        else:
            self._both = {"target": grip, "pose": lifted, "first": held["reference"], "second": self._in_palm(self.second)}
        return result

    def run_second_alone(self, plan: TwoObjectPlan, seed=0, *, position_std=0., angle_std_deg=0., hold_seconds=5.,
                         clear_of_table=False) -> dict:
        """The plan's second grasp with nothing in the hand: the control for what holding the first
        object costs. The hand starts where the first lift would have ended, with the wrapping
        fingers curled as if they held the first object, which is set aside. The pose error of
        the second grasp is the one ``run`` draws for the same seed."""
        rng = np.random.default_rng(seed)
        self._reset_scene()
        up = self._pose(self.object) @ np.asarray(plan.first_pose, float)
        up[2, 3] += LIFT
        perturb(up, rng, position_std, angle_std_deg) # the draw ``run`` spends on the first grasp
        closed = np.clip(np.asarray(plan.first_closed, float), self.lower, self.upper)
        self.p.resetBasePositionAndOrientation(self.object, [0, -5, self.spec.support_offset_z + 1.], [0, 0, 0, 1])
        self._place_palm(up)
        for j, q in zip(self.joints, closed, strict=True):
            self.p.resetJointState(self.hand, j, q)
        self._hold(closed, up, .5)
        self._first = {"target": closed, "pose": up, "reference": None, "fingers": tuple(plan.first_fingers)}
        return self.take_second(plan.second_pose, plan.second_pre, plan.second_closed, plan.second_fingers,
                                hold_seconds=hold_seconds, noise=lambda T: perturb(T, rng, position_std, angle_std_deg),
                                clear_of_table=clear_of_table)

    def run(self, plan: TwoObjectPlan, seed=0, *, position_std=0., angle_std_deg=0., hold_seconds=5.,
            clear_of_table=False) -> dict:
        """Execute a stored plan from a fresh scene, with optional hand-pose error at each grasp.

        ``clear_of_table`` fails a replay whose second grasp starts with the hand on the table,
        as the arm planner would refuse it."""
        if (plan.first_id, plan.second_id) != (self.spec.object_id, self.second_spec.object_id):
            raise ValueError("plan was made for other objects")
        if plan.support != self.support:
            raise ValueError(f"plan needs the first object on the {plan.support}")
        rng = np.random.default_rng(seed)

        def noise(T):
            return perturb(T, rng, position_std, angle_std_deg)
        self._reset_scene()
        failure = self.take_first(plan.first_pose, plan.first_pre, plan.first_closed, plan.first_fingers, noise=noise)
        if failure:
            return {"both_held": False, "clean": False, "first_failure": failure}
        return self.take_second(plan.second_pose, plan.second_pre, plan.second_closed, plan.second_fingers,
                                hold_seconds=hold_seconds, noise=noise, clear_of_table=clear_of_table)

    def tilt(self, directions=None, *, level=1., hold_seconds=1.) -> dict:
        """Which palm-frame load directions keep each object, from the state after a successful ``take_second``.

        Gravity is turned from "down" to each direction at ``level`` g with the hand fixed, as in the
        single-object load map. The first entry is the control: the direction both were just held in.
        """
        if self._both is None:
            raise RuntimeError("both objects must be held first")
        p, both = self.p, self._both
        directions = load_directions() if directions is None else np.asarray(directions, float)
        rotation = self._palm()[:3, :3]
        down = np.array([0., 0., -1.])
        state = p.saveState()
        first, second = [], []
        try:
            for direction in [rotation.T @ down, *directions]:
                p.restoreState(state)
                goal = rotation @ direction
                angle = float(np.arccos(np.clip(down @ goal, -1, 1)))
                axis = np.cross(down, goal)
                axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-8 else np.array([1., 0., 0.])
                for a in np.linspace(0, angle, max(2, int(240 * angle / np.pi))):
                    g = down * np.cos(a) + np.cross(axis, down) * np.sin(a) + axis * (axis @ down) * (1 - np.cos(a))
                    p.setGravity(*(9.81 * level * g))
                    self._control(both["target"], both["pose"])
                self._hold(both["target"], both["pose"], hold_seconds)
                first.append(self._kept(both["first"], self.object))
                second.append(self._kept(both["second"], self.second))
        finally:
            p.setGravity(0, 0, -9.81)
            p.restoreState(state)
            p.removeState(state)
        return {"directions": np.asarray(directions).tolist(), "control": [first[0], second[0]],
                "first_kept": first[1:], "second_kept": second[1:]}


    def sample_first(self, rng, kind="wrap", headings_deg=None):
        """One proposal for the thumbless grasp of the first object: palm pose in its frame, joints, fingers.

        The object stands unturned here, so a heading in its frame is the heading in the world."""
        support, fingers = FIRST_GRASPS[kind]
        if support != self.support:
            raise ValueError(f"a {kind} needs the first object on the {support}")
        if kind == "hook":
            centre, pre, closed = sample_hook(rng, self.cfg, self.first_thickness, fingers)
            return self.level_hook_pose(centre), pre, closed, fingers
        half_width = float(max(self.spec.half_extents[:2]))
        centre, yaw, pre, closed = sample_side_wrap(rng, self.cfg, half_width, self._pose(self.object)[2, 3], fingers,
                                                    headings_deg)
        if self.spec.visual_mesh:
            # A mesh has no symmetry to hide behind: what lies between palm and finger is its
            # extent toward the palm at this heading, measured from its origin, not its widest side.
            toward_palm = np.array([np.sin(yaw), -np.cos(yaw), 0.])
            centre[2] += half_width - float((self.spec.local_corners() @ toward_palm).max())
        return self.side_wrap_pose(centre, yaw), pre, closed, fingers

