"""A viewing posture for the head camera, and the arm motion to reach it."""

from __future__ import annotations

import numpy as np

from ..envs.fetch import FETCH_ARM_JOINTS
from ..kin.collision import PyBulletCollisionChecker
from ..kin.ik import DLSIK
from ..kin.rrt import rrt_connect, shortcut
from ..pipeline.common import ARM_AND_TORSO

VIEW_PAN = -1.0 # rad: the shoulder pan of the viewing posture


def view_posture(env, rest=None) -> np.ndarray:
    """The arm-and-torso joints of the viewing posture: ``rest`` (default: where the arm is) with the shoulder pan at
    ``VIEW_PAN``."""
    q = np.asarray(env.get_joints(ARM_AND_TORSO) if rest is None else rest, float).copy()
    q[ARM_AND_TORSO.index(FETCH_ARM_JOINTS[0])] = VIEW_PAN
    return q


def plan_arm(env, goal, rng: np.random.Generator, iters: int = 3000, held=None):
    """A collision-free arm-and-torso path from where the arm is to ``goal``, or None. Nothing is executed. ``held``: a
    body in the hand, carried along in the check (errand recovery's look with the wrapped object held)."""
    p = env.p
    ik = DLSIK(env, env.hand_root_link, ARM_AND_TORSO)
    saved = p.saveState()
    try:
        checker = PyBulletCollisionChecker(env, ARM_AND_TORSO, attached_body=held)
        path = rrt_connect(env.get_joints(ARM_AND_TORSO), np.asarray(goal, float), checker, ik.lower, ik.upper,
                           max_iters=iters, rng=rng)
        if path is not None:
            path = shortcut(path, checker, n_iters=60, rng=rng)
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    return path
