"""The look before each errand trip: pose estimates (known objects) or fitted shapes (unseen objects)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pipeline.common import ARM_AND_TORSO, ArmExecutor
from .estimate import estimate
from .fit import fit_primitive
from .noise import DepthNoise
from .observe import SLOT_REACH
from .perceive import DROP, PICK_BOX, canonical_yaw, standing_pose
from .posture import plan_arm, view_posture
from .render import look_at, view
from .segment import PLANE_MARGIN, cluster_indices, describe

HALF_WIDTH = .5 # either side of the slots' midpoint: two slots apart (0.72 m) with a slot's reach to spare
KEPT_POINTS = 300 # a cluster's points kept for the record, every n-th, so no random draw is spent


@dataclass
class Look:
    poses: dict = field(default_factory=dict) # body id -> 4x4 estimated pose (a target with a model)
    fits: dict = field(default_factory=dict) # role -> fit.Fit (a target without one)
    seconds: float = 0.
    posture: bool = False
    errors: dict = field(default_factory=dict) # role -> position error m (and yaw error deg with a model): reported only
    points: dict = field(default_factory=dict) # role -> the cluster's points, thinned to ``KEPT_POINTS``, of a target without a model
    clusters: dict = field(default_factory=dict) # role -> the ``segment.Cluster`` taken for it (the yaw check; never written to a record)


def look(env, targets: dict, rng: np.random.Generator, noise: DepthNoise | None = None, held=None, posture=True, action=None,
         yaw_starts: str = "box") -> Look:
    """``targets``: role -> (body, spec or None, slot). Look at the slots and estimate each target. For errand recovery's look before
    its retry, with the wrapped object in the hand: ``held`` is carried along when the arm moves, ``posture`` False keeps the arm
    where it is, and ``action`` is what the robot holds meanwhile (default ``env.hold_action``, which would let the grip go).
    ``yaw_starts`` is ``estimate``'s ``starts``: "box" (default) or "quarter" (the yaw fix: four quarter turns)."""
    noise = noise if noise is not None else DepthNoise(random_drop=DROP)
    rest, executor, start = env.get_joints(ARM_AND_TORSO), ArmExecutor(env), env.step_count
    there = plan_arm(env, view_posture(env), rng, held=held) if posture else None
    if there is not None:
        executor.follow_path(there, base=env.hold_action() if action is None else action)
    top = env.scenario.table_height
    centre = np.mean([slot for _, _, slot in targets.values()], axis=0)
    look_at(env, [*centre, top + .08], action=action)
    p = view(env, rng, noise=noise).points
    (x0, x1) = PICK_BOX[0]
    keep = np.flatnonzero((p[:, 2] > top + PLANE_MARGIN) & (p[:, 0] >= x0) & (p[:, 0] <= x1)
                          & (np.abs(p[:, 1] - centre[1]) <= HALF_WIDTH))
    clusters = [describe(p[keep[c]], top) for c in cluster_indices(p[keep])]
    out = Look(posture=there is not None)
    taken = set()
    for role, (body, spec, slot) in targets.items():
        near = [(len(c.points), -i) for i, c in enumerate(clusters)
                if i not in taken and np.linalg.norm(c.centre[:2] - np.asarray(slot)) <= SLOT_REACH]
        if not near:
            continue
        i = -max(near)[1]
        taken.add(i)
        c, truth = clusters[i], env.body_transform(body)
        out.clusters[role] = c
        if spec is None:
            out.fits[role] = fit_primitive(c, top)
            out.errors[role] = (float(np.linalg.norm(out.fits[role].centre - truth[:2, 3])),)
            out.points[role] = c.points[::max(1, len(c.points) // KEPT_POINTS)]
            continue
        e = estimate(c.points, c.centre, c.axes, spec, top, rng=np.random.default_rng(0), starts=yaw_starts)
        yaw = canonical_yaw(spec, e.yaw)
        out.poses[body] = standing_pose(spec, e.xy, yaw, top)
        yaw_true = float(np.arctan2(truth[1, 0], truth[0, 0]))
        out.errors[role] = (float(np.linalg.norm(e.xy - truth[:2, 3])),
                            float(np.degrees(abs((yaw - yaw_true + np.pi) % (2 * np.pi) - np.pi))))
    back = plan_arm(env, rest, rng, held=held) if posture else None
    if back is not None:
        executor.follow_path(back, base=env.hold_action() if action is None else action)
    out.seconds = (env.step_count - start) * env.dt
    return out
