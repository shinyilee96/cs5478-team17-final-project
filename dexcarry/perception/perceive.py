"""The robot looks before it picks: one depth image from the viewing posture gives the estimated pose of each object."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from ..pipeline.common import ARM_AND_TORSO, ArmExecutor
from .estimate import ROUND_TOLERANCE, estimate
from .identify import extents_of
from .noise import DepthNoise
from .posture import plan_arm, view_posture
from .render import look_at, view
from .segment import PLANE_MARGIN, cluster_indices, describe

PICK_BOX = ((.12, .78), (-.38, .38))
DROP = .02 # perception's 2% of pixels dropped at random


@dataclass
class Perceived:
    poses: dict = field(default_factory=dict) # body id -> 4x4 estimated world pose (missing: no cluster found)
    seconds: float = 0. # simulated time of the look, there and back
    posture: bool = False # was the viewing posture reached
    errors: dict = field(default_factory=dict) # body id -> (position error m, yaw error deg): reported only


def standing_pose(spec, xy, yaw: float, table_top: float) -> np.ndarray:
    """A standing object's world pose from its planar pose."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    T[:3, 3] = [xy[0], xy[1], table_top + spec.support_offset_z]
    return T


def canonical_yaw(spec, yaw: float) -> float:
    """The estimate's yaw reduced by the model's symmetry to the equivalent nearest the catalogue's own orientation.
    A symmetric model looks the same in each, so any is as correct as the others; but a plan is defined in the model's
    frame, and taken a quarter turn off it approaches from another side. A round model's yaw is unobservable and is taken as 0."""
    h = np.asarray(spec.half_extents, float)
    if abs(h[0] - h[1]) < ROUND_TOLERANCE / 2:
        if spec.visual_mesh:
            return 0.
        period = np.pi / 2
    else:
        period = np.pi
    return float((yaw + period / 2) % period - period / 2)


def _assign(clusters, specs: dict) -> dict:
    """Each object's cluster index: the distinct clusters with the smallest summed extent distance (None if too few)."""
    names = list(specs)
    best, best_cost = {n: None for n in names}, np.inf
    for chosen in permutations(range(len(clusters)), min(len(names), len(clusters))):
        cost = sum(float(np.max(np.abs(clusters[k].extents - extents_of(specs[n])))) for k, n in zip(chosen, names))
        if cost < best_cost:
            best, best_cost = {n: (chosen[i] if i < len(chosen) else None) for i, n in enumerate(names)}, cost
    return best


def perceive(env, rng: np.random.Generator, noise: DepthNoise | None = None) -> Perceived:
    """Look at the pick table of a two-object scene and estimate both objects' poses."""
    noise = noise if noise is not None else DepthNoise(random_drop=DROP)
    bodies = {"first": (env.object_id, env.first_spec), "second": (env.second_id, env.second_spec)}
    rest, executor, start = env.get_joints(ARM_AND_TORSO), ArmExecutor(env), env.step_count
    there = plan_arm(env, view_posture(env), rng)
    if there is not None:
        executor.follow_path(there, base=env.hold_action())
    top = env.scenario.table_height
    look_at(env, [*np.mean([env.first_xy, env.second_xy], axis=0), top + .08])
    v = view(env, rng, noise=noise)
    p = v.points
    (x0, x1), (y0, y1) = PICK_BOX
    keep = np.flatnonzero((p[:, 2] > top + PLANE_MARGIN) & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))
    groups = [keep[c] for c in cluster_indices(p[keep])]
    clusters = [describe(p[g], top) for g in groups]
    chosen = _assign(clusters, {k: spec for k, (_, spec) in bodies.items()})
    out = Perceived(posture=there is not None)
    for k, (body, spec) in bodies.items():
        i = chosen.get(k)
        if i is None:
            continue
        c = clusters[i]
        e = estimate(c.points, c.centre, c.axes, spec, top, rng=np.random.default_rng(0))
        yaw = canonical_yaw(spec, e.yaw)
        out.poses[body] = standing_pose(spec, e.xy, yaw, top)
        truth = env.body_transform(body)
        yaw_true = float(np.arctan2(truth[1, 0], truth[0, 0]))
        out.errors[body] = (float(np.linalg.norm(e.xy - truth[:2, 3])),
                            float(np.degrees(abs((yaw - yaw_true + np.pi) % (2 * np.pi) - np.pi))))
    back = plan_arm(env, rest, rng)
    if back is not None:
        executor.follow_path(back, base=env.hold_action())
    out.seconds = (env.step_count - start) * env.dt
    return out


def perturbed(env, bodies, position: float, yaw_deg: float, rng: np.random.Generator) -> dict:
    """The true standing poses of ``bodies`` moved by ``position`` metres in a uniformly random direction and turned by
    ``yaw_deg`` either way."""
    out = {}
    for body in bodies:
        T = env.body_transform(body)
        a = rng.uniform(0, 2 * np.pi)
        turn = np.radians(yaw_deg) * rng.choice([-1., 1.])
        c, s = np.cos(turn), np.sin(turn)
        R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        moved = T.copy()
        moved[:3, :3] = R @ T[:3, :3]
        moved[:2, 3] += position * np.array([np.cos(a), np.sin(a)])
        out[body] = moved
    return out
