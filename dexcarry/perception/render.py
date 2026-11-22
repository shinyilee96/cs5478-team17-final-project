"""The robot's head camera: aim the head, render depth, add noise, and return the image with the camera pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from ..envs.camera import (
    CameraIntrinsics,
    back_project,
    camera_pose_from_link,
    depth_buffer_to_metric,
    render_depth_and_mask,
)
from .noise import DepthNoise

FETCH_HEAD = CameraIntrinsics(width=640, height=480, fov_deg=45., near=.35, far=5.)


@dataclass
class View:
    points: np.ndarray # (n, 3) world points of the valid pixels
    pixels: np.ndarray # (n, 2) their (row, column)
    depth: np.ndarray # (h, w) metric depth after noise (NaN: no return)
    seg: np.ndarray # (h, w) simulator body ids (-1 none): for scoring only, never an input
    eye: np.ndarray


def look_at(env, point, steps: int = 30, action=None) -> None:
    """Pan and tilt the head toward a world point and let it settle (``envs/camera.look_at_object`` for any point). Every other
    joint holds ``action`` (default ``env.hold_action``, where the joints are: a hand gripping something then lets go)."""
    st = p.getLinkState(env.robot_id, env.link_index("head_pan_link"), computeForwardKinematics=True)
    d = np.asarray(point, float) - np.asarray(st[4], float)
    pan = (np.arctan2(d[1], d[0]) - env.get_base_pose()[2] + np.pi) % (2 * np.pi) - np.pi
    tilt = -np.arctan2(d[2], np.linalg.norm(d[:2]))
    lo, hi = env.joint_limits(["head_pan_joint", "head_tilt_joint"])
    a = env.action_with(action, head_pan_joint=float(np.clip(pan, lo[0], hi[0])), head_tilt_joint=float(np.clip(tilt, lo[1], hi[1])))
    for _ in range(steps):
        env.step(a)


def view(env, rng: np.random.Generator, intr: CameraIntrinsics = FETCH_HEAD, noise: DepthNoise | None = None) -> View:
    """One depth image from the head camera as it points now, with ``noise`` (none if None)."""
    eye, forward, up = camera_pose_from_link(env)
    _, buffer, seg = render_depth_and_mask(env, intr, eye, forward, up)
    depth = depth_buffer_to_metric(buffer, intr.near, intr.far)
    depth[buffer >= 1] = np.nan
    if noise is not None:
        depth = noise.apply(depth, rng)
    valid = np.isfinite(depth) & (depth > intr.near) & (depth < intr.far)
    points = back_project(np.nan_to_num(depth), valid, intr, eye, forward, up)
    seg = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
    return View(points, np.argwhere(valid), depth, seg, eye)
