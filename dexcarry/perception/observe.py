"""The look when the objects are unseen: the largest cluster near each slot and the primitive shape fitted to it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pipeline.common import ARM_AND_TORSO, ArmExecutor
from .fit import fit_primitive
from .noise import DepthNoise
from .perceive import DROP, PICK_BOX
from .posture import plan_arm, view_posture
from .render import look_at, view
from .segment import PLANE_MARGIN, cluster_indices, describe

SLOT_REACH = .08 # a cluster further than this from an object's slot is not taken for it


@dataclass
class Observed:
    fits: dict = field(default_factory=dict) # role ("first", "second") -> fit.Fit (missing: no cluster)
    seconds: float = 0.
    posture: bool = False
    errors: dict = field(default_factory=dict) # role -> fitted centre's distance from the true centre, m (reported)


def observe(env, rng: np.random.Generator, noise: DepthNoise | None = None) -> Observed:
    """Look at the pick table and fit a primitive to each object's cluster."""
    noise = noise if noise is not None else DepthNoise(random_drop=DROP)
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
    clusters = [describe(p[keep[c]], top) for c in cluster_indices(p[keep])]
    out = Observed(posture=there is not None)
    taken = set()
    for role, slot, body in (("first", env.first_xy, env.object_id), ("second", env.second_xy, env.second_id)):
        # The largest cluster near the slot: a half-seen object's centre lies a few centimetres from its slot, and the
        # depth noise leaves small fragments at its edges that can lie nearer.
        near = [(len(c.points), -i) for i, c in enumerate(clusters)
                if i not in taken and np.linalg.norm(c.centre[:2] - np.asarray(slot)) <= SLOT_REACH]
        if not near:
            continue
        i = -max(near)[1]
        taken.add(i)
        out.fits[role] = fit_primitive(clusters[i], top)
        out.errors[role] = float(np.linalg.norm(out.fits[role].centre - env.body_transform(body)[:2, 3]))
    back = plan_arm(env, rest, rng)
    if back is not None:
        executor.follow_path(back, base=env.hold_action())
    out.seconds = (env.step_count - start) * env.dt
    return out
