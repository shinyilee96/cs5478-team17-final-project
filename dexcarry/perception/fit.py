"""A box or a cylinder fitted to a depth cluster, and the nearest shape of the primitive-shape plan library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Fit:
    kind: str # "box" or "cyl"
    sizes: np.ndarray # (3,) long side (or diameter), short side (or diameter), height; metres
    centre: np.ndarray # (2,) horizontal centre
    yaw: float # the box's long-side direction (0 for a cylinder)
    residual: float # mean point-to-surface distance of the kept shape, metres
    other_residual: float # the same for the shape not kept


def _box_distance(points, centre_xy, axes, half, table_top, height) -> np.ndarray:
    """Distance of each point to the surface of a standing box (inside points: to the nearest face)."""
    local = np.c_[(points[:, :2] - centre_xy) @ axes.T, points[:, 2] - (table_top + height / 2)]
    h = np.array([half[0], half[1], height / 2])
    q = np.abs(local) - h
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return np.abs(outside + inside)


def _cyl_distance(points, centre_xy, radius, table_top, height) -> np.ndarray:
    """Distance of each point to the surface of a standing cylinder."""
    r = np.linalg.norm(points[:, :2] - centre_xy, axis=1)
    z = points[:, 2] - (table_top + height / 2)
    q = np.c_[r - radius, np.abs(z) - height / 2]
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return np.abs(outside + inside)


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the circle best fitting 2-D points: an algebraic (Kasa) start, then geometric least squares."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r0 = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-8)))
    sol = least_squares(lambda v: np.linalg.norm(xy - v[:2], axis=1) - v[2], [cx, cy, r0])
    return sol.x[:2], float(abs(sol.x[2]))


def fit_box(points, centre_xy, axes, extents, table_top: float, height: float):
    """Centre, yaw and half sizes of the standing box best fitting the points (geometric least squares from the
    cluster's trimmed rectangle, which alone sits a millimetre or two inside the faces). Returns (centre, axes, half)."""
    yaw0 = float(np.arctan2(axes[0][1], axes[0][0]))

    def unpack(v):
        c, s = np.cos(v[2]), np.sin(v[2])
        return v[:2], np.array([[c, s], [-s, c]]), np.abs(v[3:5])
    sol = least_squares(lambda v: _box_distance(points, *unpack(v)[:2], unpack(v)[2], table_top, height),
                        [*centre_xy, yaw0, extents[0] / 2, extents[1] / 2], loss="soft_l1", f_scale=.003)
    return unpack(sol.x)


def fit_primitive(cluster, table_top: float) -> Fit:
    """The box and the cylinder that best explain a cluster; the one with the lower mean surface distance is kept."""
    points = cluster.points
    height = float(cluster.extents[2])
    box_centre, box_axes, box_half = fit_box(points, cluster.centre[:2], cluster.axes, cluster.extents, table_top, height)
    box_res = float(np.mean(_box_distance(points, box_centre, box_axes, box_half, table_top, height)))
    # The side points only (the top cap would pull the circle inwards).
    side = points[points[:, 2] < table_top + height - .005]
    centre, radius = fit_circle((side if len(side) >= 10 else points)[:, :2])
    cyl_res = float(np.mean(_cyl_distance(points, centre, radius, table_top, height)))
    if cyl_res < box_res:
        return Fit("cyl", np.array([2 * radius, 2 * radius, height]), centre, 0., cyl_res, box_res)
    sides = 2 * np.sort(box_half)[::-1]
    long_axis = box_axes[int(np.argmax(box_half))]
    return Fit("box", np.array([*sides, height]), box_centre, float(np.arctan2(long_axis[1], long_axis[0])), box_res, cyl_res)


def cell_sizes(spec) -> np.ndarray:
    """An primitive-shape library primitive's (long, short, height), from its half extents."""
    h = 2 * np.asarray(spec.half_extents, float)
    return np.array([max(h[0], h[1]), min(h[0], h[1]), h[2]])


def nearest_cell(fit: Fit, role: str, primitives: dict) -> tuple[str, float]:
    """The primitive-shape library primitive of ``role`` ("w": wrapped, "t": three-finger) and of the fitted kind whose sizes are nearest
    by the largest relative difference; returns its name and that difference."""
    cells = {n: s for n, s in primitives.items() if n.startswith(f"atlas-{role}-{fit.kind}-")}
    if not cells:
        raise ValueError(f"no primitive-shape library entry for role {role!r} and kind {fit.kind!r}")
    gap = {n: float(np.max(np.abs(fit.sizes - cell_sizes(s)) / cell_sizes(s))) for n, s in cells.items()}
    best = min(gap, key=lambda n: (gap[n], n))
    return best, gap[best]
