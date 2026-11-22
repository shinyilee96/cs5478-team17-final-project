"""Damped-least-squares inverse kinematics in PyBullet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from ..grasp.grasp_vector import quat_xyzw_to_matrix, rotation_error_vector


@dataclass
class IKResult:
    success: bool
    q: np.ndarray # values for ``joint_names`` (same order)
    pos_err: float
    rot_err: float
    iters: int


class DLSIK:
    """IK for a link of the Fetch, moving a chosen subset of joints."""

    def __init__(self, env, ee_link: str, joint_names: list[str], *, damping: float = 0.05,
                 step_scale: float = 0.6, max_iters: int = 150, pos_tol: float = 0.005,
                 rot_tol: float = 0.03, rest_weight: float = 0.02, rest: np.ndarray | None = None):

        self.p = p
        self.env = env
        self.ee_link_name = ee_link
        self.ee_link = env.link_index(ee_link)
        self.joint_names = list(joint_names)
        self.joint_idx = [env.joint_index(n) for n in joint_names]
        self.dof_pos = [env.dof(n) for n in joint_names] # positions in the movable-joint list
        self.all_joint_idx = list(env.joint_indices)
        self.lower, self.upper = env.joint_limits(joint_names)
        self.damping = damping
        self.step_scale = step_scale
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.rest_weight = rest_weight
        self.rest = np.asarray(rest, dtype=float) if rest is not None else 0.5 * (self.lower + self.upper)
        st = p.getLinkState(env.robot_id, self.ee_link, computeForwardKinematics=True)
        # point of the URDF link frame origin, expressed in the link's inertial (COM) frame
        R_i = quat_xyzw_to_matrix(st[3])
        self.local_point = (-R_i.T @ np.asarray(st[2], dtype=float)).tolist()

    def _fk(self, q_all: np.ndarray):
        p = self.p
        for j, v in zip(self.all_joint_idx, q_all, strict=True):
            p.resetJointState(self.env.robot_id, j, float(v))
        st = p.getLinkState(self.env.robot_id, self.ee_link, computeForwardKinematics=True)
        return np.asarray(st[4], dtype=float), quat_xyzw_to_matrix(st[5])

    def _restore(self, positions, velocities):
        for j, q, velocity in zip(self.all_joint_idx, positions, velocities, strict=True):
            self.p.resetJointState(self.env.robot_id, j, float(q), float(velocity))

    def solve(self, target_pos, target_R, q0: np.ndarray | None = None) -> IKResult:
        p = self.p
        target_pos = np.asarray(target_pos, dtype=float)
        target_R = np.asarray(target_R, dtype=float)
        q_all = self.env.agent_qpos.astype(float)
        saved = q_all.copy()
        velocities = [p.getJointState(self.env.robot_id, j)[1] for j in self.all_joint_idx]
        try:
            if q0 is not None:
                q_all[self.dof_pos] = np.asarray(q0, dtype=float)
            n_all = len(self.all_joint_idx)
            zeros = [0.0] * n_all
            pos_err = rot_err = float("inf")
            it = 0
            for it in range(1, self.max_iters + 1): # the iteration count is returned below
                pos, R = self._fk(q_all)
                e_pos = target_pos - pos
                e_rot = rotation_error_vector(R, target_R)
                pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
                if pos_err < self.pos_tol and rot_err < self.rot_tol:
                    break
                jl, ja = p.calculateJacobian(self.env.robot_id, self.ee_link, self.local_point,
                                             q_all.tolist(), zeros, zeros)
                J = np.vstack([np.asarray(jl), np.asarray(ja)])[:, self.dof_pos] # 6 x k
                e = np.concatenate([e_pos, e_rot])
                JJt = J @ J.T + (self.damping**2) * np.eye(6)
                J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
                dq = J_pinv @ e
                q_sub = q_all[self.dof_pos]
                dq_rest = self.rest_weight * (self.rest - q_sub)
                dq += (np.eye(len(q_sub)) - J_pinv @ J) @ dq_rest
                q_sub = np.clip(q_sub + self.step_scale * dq, self.lower, self.upper)
                q_all[self.dof_pos] = q_sub
            q_out = q_all[self.dof_pos].copy()
            return IKResult(success=(pos_err < self.pos_tol and rot_err < self.rot_tol), q=q_out,
                            pos_err=pos_err, rot_err=rot_err, iters=it)
        finally:
            self._restore(saved, velocities)

    def solve_multi(self, target_pos, target_R, seeds: list[np.ndarray]) -> IKResult:
        """Try several seeds (e.g. current config, rest, random) and keep the best."""
        best = None
        for s in seeds:
            r = self.solve(target_pos, target_R, q0=s)
            if best is None or (r.success and not best.success) or (r.success == best.success and r.pos_err < best.pos_err):
                best = r
            if r.success:
                break
        return best

