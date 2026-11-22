"""Arm execution shared by the pickup and trip code."""

from __future__ import annotations

import numpy as np

from ..envs.fetch import FETCH_ARM_JOINTS, FETCH_TORSO_JOINT
from ..kin.rrt import densify_path

ARM_AND_TORSO = [FETCH_TORSO_JOINT] + FETCH_ARM_JOINTS


class ArmExecutor:
    """Moves the arm and torso through joint-space waypoints with position control."""

    def __init__(self, env, joint_names: list[str] = ARM_AND_TORSO):
        self.env = env
        self.joint_names = list(joint_names)
        self.dofs = env.dofs(joint_names)

    def action_for(self, q: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
        a = self.env.hold_action() if base is None else np.asarray(base, dtype=np.float32).copy()
        a[self.dofs] = np.asarray(q, dtype=np.float32)
        return a

    def move_to(self, q_target: np.ndarray, *, steps: int = 60, base: np.ndarray | None = None) -> int:
        """Interpolate from the current configuration to ``q_target`` over ``steps`` env steps."""
        q0 = self.env.get_joints(self.joint_names)
        q_target = np.asarray(q_target, dtype=float)
        for k in range(1, steps + 1):
            q = q0 + (q_target - q0) * (k / steps)
            self.env.step(self.action_for(q, base))
        return steps

    def follow_path(self, path: list[np.ndarray], *, step: float = 0.03, steps_per_point: int = 2,
                    base: np.ndarray | None = None, settle: int = 20, settle_max: int = 240,
                    tolerance: float = 0.01) -> int:
        dense = densify_path(path, step)
        n = 0
        for q in dense:
            for _ in range(steps_per_point):
                self.env.step(self.action_for(q, base))
                n += 1
        # Position control lags the path: wait until the arm has actually arrived, so what
        # happens next does not depend on how long the planned path happened to be.
        for k in range(max(settle, settle_max)):
            self.env.step(self.action_for(dense[-1], base))
            n += 1
            if k + 1 >= settle and np.abs(self.env.get_joints(self.joint_names) - dense[-1]).max() < tolerance:
                break
        return n

    def hold(self, steps: int, base: np.ndarray | None = None) -> None:
        a = self.env.hold_action() if base is None else base
        for _ in range(steps):
            self.env.step(a)
