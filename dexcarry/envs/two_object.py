"""The Fetch robot with two objects standing on the pick table and two marked spots on the target table."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..data.catalog import ObjectSpec
from ..grasp.grasp_vector import transform_from_pos_quat_xyzw
from ..sim.loads import anchor_friction
from .benchmark import DexCarryEnv, Scenario


class TwoObjectEnv(DexCarryEnv):
    """``object_id`` is the first object (taken without the thumb), ``second_id`` the second.

    Each object stands at its nominal table position, moved by up to ``jitter`` metres and turned
    by up to 0.15 rad about the vertical, drawn from the scenario seed. ``distance`` is how far
    the target table is from the pick table along the base's travel direction.
    """

    def __init__(self, first: ObjectSpec, second: ObjectSpec, *, seed=0, first_xy=(.45, -.12), second_xy=(.45, .12),
                 jitter=.01, spot_spacing=.2, distance=1., **kwargs):
        if distance < .75:
            raise ValueError("the target table would overlap the pick table below 0.75 m")
        self.first_spec, self.second_spec = first, second
        self.first_xy, self.second_xy = tuple(first_xy), tuple(second_xy)
        self.jitter, self.spot_spacing = float(jitter), float(spot_spacing)
        super().__init__(Scenario.for_object(first, seed=seed, target_xy=(.45, float(distance))), **kwargs)

    def reset(self, seed=None):
        p = self.p
        if seed is not None:
            self.scenario = replace(self.scenario, seed=int(seed))
        for name in ("second_id", *(f"_spot_{k}" for k in range(2))):
            if hasattr(self, name):
                p.removeBody(getattr(self, name))
        super().reset()
        height = self.scenario.table_height

        def stand(spec, xy):
            xy = np.asarray(xy) + self.rng.uniform(-self.jitter, self.jitter, 2)
            return [*xy, height + spec.support_offset_z + .001], p.getQuaternionFromEuler((0, 0, self.rng.uniform(-.15, .15)))
        p.resetBasePositionAndOrientation(self.object_id, *stand(self.first_spec, self.first_xy))
        self.second_id = self.second_spec.create_body(p, *stand(self.second_spec, self.second_xy))
        anchor_friction(p, self.second_id)
        # Two marked spots on the target table, side by side along the table front: the first
        # object's spot and the second's. Markers are visual only and never collide.
        tx, ty = self.scenario.target_xy
        self.goal_spots = np.array([[tx, ty - self.spot_spacing / 2, height], [tx, ty + self.spot_spacing / 2, height]])
        for k, spot in enumerate(self.goal_spots):
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.05, length=.002, rgbaColor=[.2, .6, .9, 1])
            setattr(self, f"_spot_{k}", p.createMultiBody(0, -1, visual, basePosition=[*spot[:2], height + .001]))
        for _ in range(60):
            self.step(self._hold)
        self.step_count = 0
        self.initial_object_rotation = self.get_object_transform()[:3, :3]
        return self.get_object_transform()

    def body_transform(self, body):
        return transform_from_pos_quat_xyzw(*self.p.getBasePositionAndOrientation(body))

    def set_down(self, body, spec: ObjectSpec, spot) -> dict:
        """Is a body delivered to a marked spot? The same test as ``evaluate``: at rest on the target
        table, free of the hand, its centre within the scenario tolerance of where it would stand
        on the spot. How far it leans from the vertical is reported, not required."""
        p = self.p
        T = self.body_transform(body)
        goal = np.array([spot[0], spot[1], self.scenario.table_height + spec.support_offset_z])
        distance = float(np.linalg.norm(T[:3, 3] - goal))
        speed = float(np.linalg.norm(p.getBaseVelocity(body)[0]))
        supported, released = bool(p.getContactPoints(body, self.target_table_id)), not p.getContactPoints(body, self.robot_id)
        return {"success": bool(distance <= self.scenario.target_tolerance and speed < .02 and supported and released),
                "dist_to_goal": distance, "supported": supported, "released": bool(released), "speed": speed,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))))}

    def fingers_touching(self, body) -> set[str]:
        return {finger for finger, force in self.finger_contacts(body, count=True).items() if force}

    def finger_contacts(self, body_id, count=False):
        """Normal force (N) between each finger and a body; with ``count``, the number of contacts instead."""
        contacts = self.p.getContactPoints(self.robot_id, body_id)
        return {finger: sum(1 if count else c[9] for c in contacts if c[3] in {self.joint_index("leap_" + j) for j in names})
                for finger, names in self.hand_config["finger_groups"].items()}
