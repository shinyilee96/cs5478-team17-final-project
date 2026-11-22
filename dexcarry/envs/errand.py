"""The errand scene: 4 to 6 objects on the pick table (a row of slots on the long table, scattered on the round
table) and the target table ``distance`` metres away.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..sim.loads import anchor_friction
from .two_object import TwoObjectEnv

SLOT_SPACING = .24
SLOT_X = .45
LIBRARY_FIRST_Y = -.12 # where the plan library's wrapped object stands, relative to the base
TABLE_GAP = .3 # the smallest gap between the tables: the two-object scene's at 1 m
TARGET_HALF_WIDTH = .3 # the target table's half length along y (``DexCarryEnv.reset``)
PARK_XY = (4.5, -4.) # on the floor, out of every route and view
PARK_STEP = .4
FLOOR_HALF, FLOOR_MARGIN = 5., 1. # the scene's floor, and how far it must reach past the target table
POSE_EVERY = 4 # as the video's frames
DISC = {"centre": (.55, 0.), "radius": .5, "thickness": .05} # the round table
TABLES = ("box", "disc")
# On the round table the base may also stand 0.30 m forward: the band's far edge is past the long table's 0.22 m for some grasps.
# Tried last, for every grasp; the counter keeps ``Scenario.standing_offsets``.
DISC_STANDING_OFFSETS = (0., .08, .15, .22, .30)


def slot_positions(n: int, spacing: float = SLOT_SPACING, x: float = SLOT_X) -> list[tuple[float, float]]:
    """``n`` slots in a row along the counter, centred on y = 0."""
    return [(x, float((k - (n - 1) / 2) * spacing)) for k in range(n)]


def with_object(scenario, spec):
    """``scenario`` with ``spec`` as its object: the single-object pipeline reads the object's shape from there."""
    return replace(scenario, object_id=spec.object_id, object_half_extents=spec.half_extents, object_mass=spec.mass,
                   object_friction=spec.friction, object_visual_mesh=spec.visual_mesh,
                   object_collision_meshes=spec.collision_meshes, object_scale=spec.scale)


class ErrandEnv(TwoObjectEnv):
    odometer = 0. # metres the base has driven since the reset
    pose_log = None # a list to fill every ``POSE_EVERY`` control steps
    events = None # a list to fill with the scene's events, for the pose log
    check_log = None # a list to fill with every ``set_down`` test
    follow = None # a recording's camera follows the base: (eye, target), each an offset from it

    def __init__(self, specs, *, seed=0, spacing=SLOT_SPACING, distance=1., positions=None, table="box", **kwargs):
        if not 2 <= len(specs) <= 8:
            raise ValueError("an errand has 2 to 8 objects")
        if table not in TABLES:
            raise ValueError(f"table is one of {TABLES}")
        self.specs, self.table = list(specs), table
        if positions is not None:
            if len(positions) != len(self.specs):
                raise ValueError("one position per object")
            kwargs.setdefault("jitter", 0.)
        self.positions = [tuple(map(float, xy)) for xy in positions] if positions is not None else slot_positions(len(self.specs), spacing)
        self.slots = list(self.positions)
        self.half_width = DISC["radius"] if table == "disc" else max(abs(y) for _, y in self.slots) + .1
        self.bodies: list[int] = []
        super().__init__(self.specs[0], self.specs[1], seed=seed, first_xy=self.slots[0], second_xy=self.slots[1],
                         distance=max(distance, self.half_width + TABLE_GAP + TARGET_HALF_WIDTH), **kwargs)

    def _pick_table(self):
        if self.table == "box":
            return super()._pick_table()
        p, (x, y), r, t = self.p, DISC["centre"], DISC["radius"], DISC["thickness"]
        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=t)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=t, rgbaColor=(.6, .6, .6, 1))
        return p.createMultiBody(0., shape, visual, basePosition=[x, y, self.scenario.table_height - t / 2])

    def reset(self, seed=None):
        p = self.p
        # The two-object reset removes ``object_id`` and ``second_id``: point them at slots 0 and 1 again first.
        if self.bodies:
            self.object_id, self.second_id = self.bodies[0], self.bodies[1]
            for body in self.bodies[2:]:
                p.removeBody(body)
        self.first_spec, self.second_spec = self.specs[0], self.specs[1]
        self.slots = list(self.positions)
        self.first_xy, self.second_xy = self.slots[0], self.slots[1]
        self.scenario = with_object(replace(self.scenario, pick_table_half_width=self.half_width), self.specs[0])
        if self.table == "disc":
            self.scenario = replace(self.scenario, standing_offsets=DISC_STANDING_OFFSETS)
        out = super().reset(seed)
        # The floor is 5 m to either side (``DexCarryEnv.reset``); the base must not run off it and catch its edge coming back.
        if self.scenario.target_xy[1] + FLOOR_MARGIN > FLOOR_HALF:
            p.removeBody(self.plane_id)
            self.plane_id = self._box((FLOOR_HALF, self.scenario.target_xy[1] + FLOOR_MARGIN, .05), (0, 0, -.05))
        height = self.scenario.table_height
        self.bodies = [self.object_id, self.second_id]
        for spec, xy in zip(self.specs[2:], self.positions[2:]):
            at = np.asarray(xy) + self.rng.uniform(-self.jitter, self.jitter, 2)
            body = spec.create_body(p, [*at, height + spec.support_offset_z + .001],
                                    p.getQuaternionFromEuler((0, 0, self.rng.uniform(-.15, .15))))
            anchor_friction(p, body)
            self.bodies.append(body)
        if len(self.bodies) > 2:
            for _ in range(60):
                self.step(self._hold)
            self.step_count = 0
        self.present(0, 1)
        self.odometer = 0.
        self._was = self._base_xy()
        return out

    def _base_xy(self) -> np.ndarray:
        return np.array([self.p.getJointState(self.robot_id, self.joint_index(j))[0] for j in ("root_x_axis_joint", "root_y_axis_joint")])

    def _capture_frame(self):
        if self.follow is not None:
            base = np.array([*self._base_xy(), 0.])
            self._camera = tuple((base + offset).tolist() for offset in self.follow)
        super()._capture_frame()

    def step(self, action):
        """As ever, and it keeps the base's path length and, if asked, a log of every pose. Neither touches the physics."""
        super().step(action)
        was = getattr(self, "_was", None)
        if was is not None:
            now = self._base_xy()
            self.odometer += float(np.linalg.norm(now - was))
            self._was = now
        if self.pose_log is not None and self.step_count % POSE_EVERY == 0:
            bodies = [self.p.getBasePositionAndOrientation(b) for b in self.bodies]
            self.pose_log.append((self.step_count, self.agent_qpos, np.array([[*position, *turn] for position, turn in bodies])))

    def note(self, kind: str, **data) -> None:
        """Log a scene event at this control step, if a log is on (trip start and end, an object cleared)."""
        if self.events is not None:
            self.events.append({"step": self.step_count, "kind": kind, **data})

    def set_down(self, body, spec, spot) -> dict:
        """As ever, and every test is kept if a log is on: the trip's record, with the moment and the slot."""
        out = super().set_down(body, spec, spot)
        if self.check_log is not None:
            slot = self.bodies.index(body) if body in self.bodies else None
            self.check_log.append({"seconds": self.step_count * self.dt, "slot": slot, **out})
        return out

    def present(self, i: int, j: int) -> float:
        """Make slot ``i``'s object the trip's wrapped (first) object and slot ``j``'s the three-finger (second) one.
        Returns the base's y for the trip's stance (slot i 0.12 m to its right, as in the plan library)."""
        if i == j or not (0 <= i < len(self.bodies) and 0 <= j < len(self.bodies)):
            raise ValueError("two different slots")
        self.object_id, self.second_id = self.bodies[i], self.bodies[j]
        self.first_spec, self.second_spec = self.specs[i], self.specs[j]
        self.first_xy, self.second_xy = self.slots[i], self.slots[j]
        self.scenario = with_object(self.scenario, self.first_spec)
        self.initial_object_rotation = self.get_object_transform()[:3, :3]
        return self.slots[i][1] - LIBRARY_FIRST_Y

    def stand_at(self, y: float) -> None:
        """Put the base at stance ``y`` along the counter, with no time passing: for dry runs and practice pickups. In an
        errand the base drives there (``pipeline.errand``)."""
        self.set_joints({"root_y_axis_joint": float(y)})
        self._hold = np.clip(self.agent_qpos, self.joint_lower, self.joint_upper)

    def park(self, i: int) -> None:
        """Take slot ``i``'s object out of the errand: it lies on the floor, out of every route and view."""
        self.note("cleared", slot=i)
        x, y = PARK_XY
        self.p.resetBasePositionAndOrientation(self.bodies[i], [x, y + i * PARK_STEP, .15], [0., 0., 0., 1.])
        self.p.resetBaseVelocity(self.bodies[i], [0., 0., 0.], [0., 0., 0.])
