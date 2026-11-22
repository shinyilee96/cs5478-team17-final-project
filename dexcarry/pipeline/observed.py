"""Planning without the true shape: the arm planners see an unseen object only as the primitive shape fitted to it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grasp.grasp_vector import invert_transform, matrix_to_quat_xyzw

PARK = np.array([40., 40., -5.]) # where an unseen body waits while the planners look at its proxy


def plan_frame(fit, cell_kind: str, table_top: float) -> np.ndarray:
    """The frame the cell's plans are placed in: the fitted centre at half the fitted height, yaw reduced by the cell's
    symmetry to the turn nearest 0 (the catalogue's orientation, as ``perceive.canonical_yaw``): a quarter turn for a
    square box cell, none for a cylinder cell."""
    yaw = 0. if cell_kind == "cyl" or fit.kind == "cyl" else float((fit.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    T[:3, 3] = [fit.centre[0], fit.centre[1], table_top + fit.sizes[2] / 2]
    return T


def proxy_body(p, fit, frame: np.ndarray) -> int:
    """A static, collision-only body of the fitted primitive, its shape turned within the plan frame to the fitted yaw."""
    if fit.kind == "cyl":
        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(fit.sizes[0] / 2), height=float(fit.sizes[2]))
    else:
        frame_yaw = float(np.arctan2(frame[1, 0], frame[0, 0]))
        turn = fit.yaw - frame_yaw
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[float(fit.sizes[0] / 2), float(fit.sizes[1] / 2),
                                                                float(fit.sizes[2] / 2)],
                                       collisionFrameOrientation=p.getQuaternionFromEuler((0., 0., turn)))
    return p.createMultiBody(0, shape, -1, basePosition=PARK.tolist(), baseOrientation=matrix_to_quat_xyzw(np.eye(3)))


@dataclass
class ObservedPlanning:
    """The planners' view of a two-object scene whose objects are known only by fitted primitives.

    ``frames``: role ("first", "second") -> plan frame at the look; ``fits``: role -> ``perception.fit.Fit``;
    ``held_offset``: role -> pose of the proxy in the true body's frame for an object no longer where the look saw it
    (default: as at the look; a dry run that puts the body itself where the plan's primitive settled passes identity)."""
    env: object
    fits: dict
    frames: dict
    held_offset: dict = field(default_factory=dict)
    true: dict = field(default_factory=dict)
    proxy: dict = field(default_factory=dict)
    active: bool = False

    @classmethod
    def build(cls, env, fits: dict, cells: dict, *, held_identity=False):
        """Proxies for both objects of ``env`` from this look's fits; ``cells``: role -> the mapped cell's kind."""
        top = env.scenario.table_height
        frames = {role: plan_frame(fits[role], cells[role], top) for role in ("first", "second")}
        view = cls(env, fits, frames)
        view.true = {"first": env.object_id, "second": env.second_id}
        for role, body in view.true.items():
            view.proxy[role] = proxy_body(env.p, fits[role], frames[role])
            view.held_offset[role] = (np.eye(4) if held_identity
                                      else invert_transform(env.body_transform(body)) @ frames[role])
        return view

    def enter(self, env, bodies) -> None:
        """Called after each restore inside a planner: ``bodies`` (true or proxy ids) are standing where the look saw
        them; any other object is wherever the simulator has it now, its proxy carried along."""
        p = env.p
        back = {self.proxy[r]: self.true[r] for r in self.true}
        standing = {back.get(b, b) for b in bodies}
        self.leave(env)
        for role, body in self.true.items():
            T = self.frames[role] if body in standing else env.body_transform(body) @ self.held_offset[role]
            p.resetBasePositionAndOrientation(self.proxy[role], T[:3, 3], matrix_to_quat_xyzw(T[:3, :3]))
            p.resetBasePositionAndOrientation(body, (PARK + [0., 1. + (role == "second"), 0.]).tolist(), [0., 0., 0., 1.])
        env.object_id, env.second_id = self.proxy["first"], self.proxy["second"]
        self.active = True

    def leave(self, env) -> None:
        """Give the planners' ids back to the true bodies (positions come back with the planner's own restore)."""
        if self.active:
            env.object_id, env.second_id = self.true["first"], self.true["second"]
            self.active = False

    def remove(self) -> None:
        """Delete the proxies (before the scene is reset)."""
        self.leave(self.env)
        for body in self.proxy.values():
            self.env.p.removeBody(body)
        self.proxy = {}
