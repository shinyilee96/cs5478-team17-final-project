"""Collision predicates for planning in PyBullet."""

from __future__ import annotations

import numpy as np
import pybullet as p

from ..grasp.grasp_vector import invert_transform, matrix_to_quat_xyzw, transform_from_pos_quat_xyzw


class PyBulletCollisionChecker:
    def __init__(self, env, joint_names: list[str], *, ignore_bodies: set[int] | None = None,
                 attached_body: int | None = None, margin: float = 0.0,
                 ignore_links_of_robot: set[int] | None = None):
        """
        Parameters
        ----------
        joint_names
            The joints a configuration ``q`` refers to (same order).
        ignore_bodies
            Bodies whose contacts with the robot do not count (e.g. the goal marker).
        attached_body
            An object held in the hand: its contacts with the robot are ignored, but its contacts
            with the rest of the world count.
        margin
            A contact counts when its distance is below this value (0 = penetration only).
        """

        self.p = p
        self.env = env
        self.joint_idx = [env.joint_index(n) for n in joint_names]
        self.ignore = set(ignore_bodies or set())
        if getattr(env, "goal_marker_id", None) is not None:
            self.ignore.add(env.goal_marker_id)
        self.attached = attached_body
        self.margin = margin
        self.ignore_links = set(ignore_links_of_robot or set())
        self.allowed_self_pairs = getattr(env, "allowed_self_pairs", None)
        # Fingers closed on an object may rest on each other or on the palm. Unless the hand's own
        # joints are being planned, that contact is the same in every configuration, so it is not
        # a collision of the motion.
        hand = set(getattr(env, "hand_link_indices", set()))
        self.fixed_hand_links = hand if not hand & set(self.joint_idx) else set()
        self.allowed_contacts: dict[int, set[int]] = {}
        self.allowed_support: set[int] = set()
        self.n_checks = 0
        # What the last colliding configuration touched: ("robot", other body, robot link, depth) or
        # ("held", held body, other body, depth), depth in metres. For diagnostics only.
        self.last_collision = None
        self.attachment_transform = None
        self.attachment_link = getattr(env, "hand_root_link", "wrist_roll_link")
        # Further bodies held in the same hand (body -> pose in the attachment link). They move with
        # the hand like ``attached_body`` but may rest on nothing: ``allowed_support`` is for the
        # one body that is being set down.
        self.also_attached: dict[int, np.ndarray] = {}
        if attached_body is not None:
            self.attachment_transform = self._pose_in_hand(attached_body)

    def _pose_in_hand(self, body) -> np.ndarray:

        return (invert_transform(self.env.get_link_transform(self.attachment_link)) @
                transform_from_pos_quat_xyzw(*self.p.getBasePositionAndOrientation(body)))

    def attach_also(self, body: int) -> None:
        """Hold a second body where it is now. Needs ``attached_body``; restore the simulator state yourself."""
        if self.attached is None or body == self.attached:
            raise ValueError("a second held body needs a different first one")
        self.also_attached[body] = self._pose_in_hand(body)

    def _set(self, q: np.ndarray) -> None:
        for j, v in zip(self.joint_idx, q, strict=True):
            self.p.resetJointState(self.env.robot_id, j, float(v))
        if self.attachment_transform is not None:
            hand = self.env.get_link_transform(self.attachment_link)
            for body, relative in ((self.attached, self.attachment_transform), *self.also_attached.items()):
                T = hand @ relative
                self.p.resetBasePositionAndOrientation(body, T[:3, 3], matrix_to_quat_xyzw(T[:3, :3]))

    def snapshot(self) -> np.ndarray:
        return np.array([self.p.getJointState(self.env.robot_id, j)[0] for j in self.joint_idx])

    def restore(self, q: np.ndarray) -> None:
        self._set(q)

    def in_collision(self, q: np.ndarray) -> bool:
        p = self.p
        self.n_checks += 1
        self._set(q)
        p.performCollisionDetection()
        held = {self.attached, *self.also_attached} - {None}
        for c in p.getContactPoints(bodyA=self.env.robot_id):
            other = c[2] if c[1] == self.env.robot_id else c[1]
            link_robot = c[3] if c[1] == self.env.robot_id else c[4]
            if other == self.env.robot_id:
                if c[3] in self.fixed_hand_links and c[4] in self.fixed_hand_links:
                    continue
                pair = tuple(sorted((c[3], c[4])))
                if self.allowed_self_pairs is not None and pair not in self.allowed_self_pairs and c[8] < self.margin:
                    self.last_collision = ("robot", other, pair[0], -c[8])
                    return True
                continue
            if other == getattr(self.env, "plane_id", None) and link_robot in getattr(self.env, "support_links", set()):
                continue
            if other in self.ignore or link_robot in self.ignore_links:
                continue
            if link_robot in self.allowed_contacts.get(other, set()):
                continue
            if other in held:
                if link_robot in getattr(self.env, "hand_link_indices", {link_robot}):
                    continue
            if c[8] < self.margin:
                self.last_collision = ("robot", other, link_robot, -c[8])
                return True
        for body in held:
            # Held bodies may touch the hand and each other; only the first may touch its support.
            support = self.allowed_support if body == self.attached else set()
            for c in p.getContactPoints(bodyA=body):
                other = c[2] if c[1] == body else c[1]
                if other == self.env.robot_id or other in held or other in self.ignore or other in support:
                    continue
                if c[8] < self.margin:
                    self.last_collision = ("held", body, other, -c[8])
                    return True
        return False

    def __call__(self, q: np.ndarray) -> bool:
        return self.in_collision(q)


class PlanningSession:
    """Context manager: snapshot joint states on entry, restore them on exit."""

    def __init__(self, checker: PyBulletCollisionChecker):
        self.checker = checker
        self.saved = None

    def __enter__(self):
        self.saved = self.checker.snapshot()
        self.velocities = [self.checker.p.getJointState(self.checker.env.robot_id, j)[1]
                           for j in self.checker.joint_idx]
        self.object_state = None
        if self.checker.attached is not None:
            p = self.checker.p
            self.object_state = (p.getBasePositionAndOrientation(self.checker.attached),
                                 p.getBaseVelocity(self.checker.attached))
        return self.checker

    def __exit__(self, *exc):
        self.checker.restore(self.saved)
        p = self.checker.p
        for j, q, velocity in zip(self.checker.joint_idx, self.saved, self.velocities, strict=True):
            p.resetJointState(self.checker.env.robot_id, j, q, velocity)
        if self.object_state is not None:
            (pos, quat), (linear, angular) = self.object_state
            p.resetBasePositionAndOrientation(self.checker.attached, pos, quat)
            p.resetBaseVelocity(self.checker.attached, linear, angular)
        return False
