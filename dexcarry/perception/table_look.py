"""The whole-table look at the start of an errand: finds the objects and estimates their poses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pipeline.common import ARM_AND_TORSO, ArmExecutor
from .estimate import estimate
from .identify import identify_by_fit, visual_classes
from .noise import DepthNoise
from .perceive import DROP, canonical_yaw, standing_pose
from .posture import plan_arm, view_posture
from .render import look_at, view
from .segment import PLANE_MARGIN, cluster_indices, describe

IDENTITY_REACH = .08 # m between a cluster's centre and the body it takes, as the per-trip look's slot reach
LOOK_HEIGHT = .08 # the head aims this far above the table's top, as the per-trip look
TOP_LIMIT = .40 # points more than this above the table are not objects (the tallest stands about 0.25 m)
TURN_MARGIN = .25 # a turn the look cannot rule out: its fit ends at most this much (relative) above the best residual
TURN_APART = 10. # degrees from the kept turn, after the model's symmetry, for a turn to count as another


@dataclass
class TableLook:
    found: dict = field(default_factory=dict) # object index -> estimated (x, y)
    poses: dict = field(default_factory=dict) # body id -> 4x4 estimated pose
    errors: dict = field(default_factory=dict) # object index -> (position error m, yaw error deg): reported only
    missing: list = field(default_factory=list) # object indices not found
    clusters: int = 0 # clusters seen on the table
    identity: dict = field(default_factory=dict) # object index -> (identify_by_fit's class, the true object's class): reported only
    seconds: float = 0.
    posture: bool = False
    bodies: dict = field(default_factory=dict) # object index -> body id, of the objects found
    clusters_kept: dict = field(default_factory=dict) # object index -> its ``segment.Cluster`` (development checks; never in a record)
    turns: dict = field(default_factory=dict) # body id -> [4x4,...]: other turns the look cannot rule out (the second case)

    def record(self) -> dict:
        return {"found": {str(k): list(v) for k, v in self.found.items()}, "errors": {str(k): list(v) for k, v in self.errors.items()},
                "missing": list(self.missing), "clusters": self.clusters, "seconds": self.seconds, "posture": self.posture,
                "identity": {str(k): list(v) for k, v in self.identity.items()},
                "other_turns": {str(k): len(self.turns.get(b, [])) for k, b in self.bodies.items()}}


def other_turns(spec, e, yaw: float, top: float) -> list:
    """The poses of the other turns ``e``'s fits cannot rule out: within ``TURN_MARGIN`` of the best residual and ``TURN_APART`` from
    ``yaw`` (both after the model's symmetry), each once."""
    out, kept = [], [yaw]
    for xy, other, res in e.fits:
        turned = canonical_yaw(spec, other)
        apart = min(np.degrees(abs((turned - k + np.pi) % (2 * np.pi) - np.pi)) for k in kept)
        if res <= e.residual * (1 + TURN_MARGIN) and apart > TURN_APART:
            kept.append(turned)
            out.append(standing_pose(spec, xy, turned, top))
    return out


def table_look(env, rng: np.random.Generator, centre, radius: float, noise: DepthNoise | None = None, yaw_starts: str = "quarter",
               catalog: dict | None = None, classes: dict | None = None, aims=None, box=None) -> TableLook:
    """Look once at the disc (``centre`` (x, y), ``radius``) of ``env`` (an ``ErrandEnv``: ``bodies``, ``specs``) and estimate every
    object found. ``catalog`` and ``classes`` (the robot's catalogue and visual classes) only add ``identify_by_fit``'s record.

    The long table: ``aims`` is a list of (x, y) table points; the head turns to each in order and takes one
    image there, and every image's points are pooled before clustering (they are all in the world frame). ``box`` ((x0, x1), (y0, y1))
    replaces the disc as the region whose points count. With both None the look is the round table's, unchanged."""
    noise = noise if noise is not None else DepthNoise(random_drop=DROP)
    rest, executor, start = env.get_joints(ARM_AND_TORSO), ArmExecutor(env), env.step_count
    there = plan_arm(env, view_posture(env), rng)
    if there is not None:
        executor.follow_path(there, base=env.hold_action())
    top = env.scenario.table_height
    images = []
    for aim in (aims if aims is not None else [centre]):
        look_at(env, [*aim, top + LOOK_HEIGHT])
        images.append(view(env, rng, noise=noise).points)
    p = np.concatenate(images) if len(images) > 1 else images[0]
    if box is not None:
        (x0, x1), (y0, y1) = box
        over = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
    else:
        over = np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) <= radius
    keep = np.flatnonzero((p[:, 2] > top + PLANE_MARGIN) & (p[:, 2] < top + TOP_LIMIT) & over)
    clusters = [describe(p[keep[c]], top) for c in cluster_indices(p[keep])]
    out = TableLook(posture=there is not None, clusters=len(clusters))
    truth = {k: env.body_transform(body) for k, body in enumerate(env.bodies)}
    taken = set()
    for c in clusters: # largest first
        near = sorted((float(np.linalg.norm(T[:2, 3] - c.centre[:2])), k) for k, T in truth.items() if k not in taken)
        if not near or near[0][0] > IDENTITY_REACH:
            continue
        k = near[0][1]
        taken.add(k)
        spec = env.specs[k]
        e = estimate(c.points, c.centre, c.axes, spec, top, rng=np.random.default_rng(0), starts=yaw_starts, keep_fits=True)
        yaw = canonical_yaw(spec, e.yaw)
        out.turns[env.bodies[k]] = other_turns(spec, e, yaw, top)
        out.bodies[k] = env.bodies[k]
        out.found[k] = (float(e.xy[0]), float(e.xy[1]))
        out.clusters_kept[k] = c
        out.poses[env.bodies[k]] = standing_pose(spec, e.xy, yaw, top)
        yaw_true = float(np.arctan2(truth[k][1, 0], truth[k][0, 0]))
        out.errors[k] = (float(np.linalg.norm(e.xy - truth[k][:2, 3])), float(np.degrees(abs((yaw - yaw_true + np.pi) % (2 * np.pi) - np.pi))))
        if catalog is not None:
            classes = classes or visual_classes(catalog)
            fitted, _, _ = identify_by_fit(c, catalog, top, classes)
            out.identity[k] = (fitted, classes.get(spec.object_id, spec.object_id))
    out.missing = sorted(set(truth) - taken)
    back = plan_arm(env, rest, rng)
    if back is not None:
        executor.follow_path(back, base=env.hold_action())
    out.seconds = (env.step_count - start) * env.dt
    return out
