"""Pose of a standing known object from its depth points, by point-to-plane ICP from each quarter turn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROUND_TOLERANCE = .002 # m: a model whose horizontal sides differ by less is treated as round about its axis


@dataclass
class Estimate:
    xy: np.ndarray
    yaw: float
    residual: float # mean point-to-plane distance after ICP, metres
    yaw_observable: bool # False for a round model
    fits: tuple = () # with ``estimate(keep_fits=True)``: every start's (xy, yaw, residual), in start order


def model_surface(spec, n: int = 4000, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points and outward normals on the model's surface, in its own frame (standing: z up, base at -half height).
    A box spec is sampled face by face; a mesh spec from its visual mesh."""
    rng = rng or np.random.default_rng(0)
    if spec.visual_mesh:
        mesh = trimesh.load(spec.visual_mesh, force="mesh", process=False)
        points, faces = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
        return np.asarray(points, float) * spec.scale, np.asarray(mesh.face_normals[faces], float)
    h = np.asarray(spec.half_extents, float)
    area = np.array([h[1] * h[2], h[1] * h[2], h[0] * h[2], h[0] * h[2], h[0] * h[1], h[0] * h[1]])
    faces = rng.choice(6, size=n, p=area / area.sum())
    points, normals = rng.uniform(-1, 1, (n, 3)) * h, np.zeros((n, 3))
    for f in range(6):
        axis, sign = f // 2, 1. if f % 2 == 0 else -1.
        points[faces == f, axis] = sign * h[axis]
        normals[faces == f, axis] = sign
    return points, normals


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def icp_xy_yaw(seen: np.ndarray, model_pts: np.ndarray, model_nrm: np.ndarray, xy0, yaw0, iterations: int = 30,
               reject: float = .02) -> tuple[np.ndarray, float, float]:
    """Point-to-plane ICP over (x, y, yaw). ``seen`` are world points with z measured from the model's centre height.
    Correspondences farther than ``reject`` metres are ignored. Returns (xy, yaw, mean residual)."""
    tree = cKDTree(model_pts)
    xy, yaw = np.asarray(xy0, float).copy(), float(yaw0)
    residual = np.inf
    for _ in range(iterations):
        R = _rot(yaw)
        local = np.c_[(seen[:, :2] - xy) @ R, seen[:, 2]] # world -> model frame
        dist, idx = tree.query(local)
        keep = dist < reject
        if keep.sum() < 10:
            break
        p, q, n = local[keep], model_pts[idx[keep]], model_nrm[idx[keep]]
        r = np.einsum("ij,ij->i", p - q, n) # signed point-to-plane distance
        residual = float(np.mean(np.abs(r)))
        # Linearised: a small change (dx, dy, dyaw) of the pose moves a model-frame point p by -R^T d - dyaw * J p.
        Jp = np.c_[-p[:, 1], p[:, 0]] # d(p_xy)/d(-yaw)
        A = np.c_[-(n[:, :2] @ R.T), -np.einsum("ij,ij->i", n[:, :2], Jp)]
        step, *_ = np.linalg.lstsq(A, -r, rcond=None)
        xy += step[:2]
        yaw += step[2]
        if np.linalg.norm(step[:2]) < 1e-5 and abs(step[2]) < 1e-5:
            break
    return xy, yaw, residual


YAW_STARTS = ("box", "quarter")


def estimate(cluster_points: np.ndarray, box_centre, box_axes, spec, table_top: float,
             rng: np.random.Generator | None = None, starts: str = "box", keep_fits: bool = False) -> Estimate:
    """The standing pose of ``spec`` that best explains the cluster, over the model's symmetric yaw candidates.
    ``starts``: "box" starts ICP at the box's long side, two turns for a model that
    is not square; "quarter" starts it at all four quarter turns of the box's axis for any model that is not round, whatever its
    sides, and keeps the lowest residual, so a footprint near square whose box takes the model's short side for its long one is
    not fitted a quarter turn off."""
    if starts not in YAW_STARTS:
        raise ValueError(f"starts must be one of {YAW_STARTS}")
    pts, nrm = model_surface(spec, rng=rng)
    h = np.asarray(spec.half_extents, float)
    round_ = abs(h[0] - h[1]) < ROUND_TOLERANCE / 2 and bool(spec.visual_mesh)
    seen = np.c_[cluster_points[:, :2], cluster_points[:, 2] - (table_top + spec.support_offset_z)]
    base = float(np.arctan2(box_axes[0][1], box_axes[0][0])) # the box's long side
    # The model's long horizontal side is its x axis when hx >= hy.
    base -= 0. if h[0] >= h[1] else np.pi / 2
    square = abs(h[0] - h[1]) < ROUND_TOLERANCE / 2
    turns = (0.,) if round_ else (0., np.pi / 2, np.pi, 3 * np.pi / 2) if square or starts == "quarter" else (0., np.pi)
    best, fits = None, []
    for t in turns:
        xy, yaw, res = icp_xy_yaw(seen, pts, nrm, np.asarray(box_centre[:2], float), base + t)
        if best is None or res < best.residual:
            best = Estimate(xy, float((yaw + np.pi) % (2 * np.pi) - np.pi), res, not round_)
        fits.append((xy, float(yaw), res))
    if keep_fits:
        best.fits = tuple(fits)
    return best
