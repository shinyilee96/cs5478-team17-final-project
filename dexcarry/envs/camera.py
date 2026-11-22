"""Depth camera on the Fetch head and back-projection to a point cloud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from ..grasp.grasp_vector import quat_xyzw_to_matrix


@dataclass
class CameraIntrinsics:
    width: int = 320
    height: int = 240
    fov_deg: float = 60.0
    near: float = 0.05
    far: float = 5.0


def camera_pose_from_link(env, link_name: str = "head_camera_rgb_optical_frame"):
    """Eye position, forward and up vectors of an optical-frame link."""

    st = p.getLinkState(env.robot_id, env.link_index(link_name), computeForwardKinematics=True)
    eye = np.asarray(st[4], dtype=float)
    R = quat_xyzw_to_matrix(st[5])
    forward = R[:, 2]
    up = -R[:, 1]
    return eye, forward, up


def render_depth_and_mask(env, intr: CameraIntrinsics, eye, forward, up):

    view = p.computeViewMatrix(eye.tolist(), (eye + forward).tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(intr.fov_deg, intr.width / intr.height, intr.near, intr.far)
    _, _, rgba, depth, seg = p.getCameraImage(intr.width, intr.height, view, proj, renderer=p.ER_TINY_RENDERER)
    depth = np.asarray(depth, dtype=np.float64).reshape(intr.height, intr.width)
    seg = np.asarray(seg, dtype=np.int64).reshape(intr.height, intr.width)
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(intr.height, intr.width, 4)
    return rgba, depth, seg


def depth_buffer_to_metric(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    return far * near / (far - (far - near) * depth)


def back_project(depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics, eye, forward, up) -> np.ndarray:
    """World points for the masked pixels."""
    h, w = depth_m.shape
    vs, us = np.nonzero(mask)
    z = depth_m[vs, us]
    tan_half = np.tan(np.deg2rad(intr.fov_deg) / 2.0)
    aspect = w / h
    x_ndc = (2.0 * (us + 0.5) / w - 1.0)
    y_ndc = (1.0 - 2.0 * (vs + 0.5) / h)
    x_cam = x_ndc * z * tan_half * aspect
    y_cam = y_ndc * z * tan_half
    f = forward / np.linalg.norm(forward)
    u = up - np.dot(up, f) * f
    u /= max(np.linalg.norm(u), 1e-9)
    r = np.cross(f, u)
    return eye + z[:, None] * f + x_cam[:, None] * r + y_cam[:, None] * u

