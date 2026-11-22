"""Segmentation of a depth image into objects without the simulator's mask: table plane removed, Euclidean clusters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

RADIUS = .01
MIN_POINTS = 30
PLANE_MARGIN = .005


@dataclass
class Cluster:
    points: np.ndarray # (n, 3) world points
    centre: np.ndarray # (3,) centre of the oriented box: x-y box centre, z mid-height above the table
    axes: np.ndarray # (2, 2) the box's horizontal axes (rows), the longer first
    extents: np.ndarray # (3,) full sizes: long side, short side, height


def cluster_indices(points: np.ndarray, radius: float = RADIUS, min_points: int = MIN_POINTS) -> list[np.ndarray]:
    """Indices of each connected component of the graph joining points closer than ``radius``; largest first."""
    n = len(points)
    if n == 0:
        return []
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    count, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == k) for k in range(count)]
    return sorted((g for g in groups if len(g) >= min_points), key=len, reverse=True)


def describe(points: np.ndarray, table_top: float, low: float = 2., high: float = 98.) -> Cluster:
    """The cluster's oriented horizontal box and height. The box is the minimum-area rectangle over the directions of
    the x-y convex hull's edges (principal axes are arbitrary for a square cross-section); its sides are measured
    between the ``low`` and ``high`` percentiles, which trim stray points."""
    xy = points[:, :2]
    mid = xy.mean(axis=0)
    try:
        hull = xy[ConvexHull(xy).vertices]
        edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
        angles = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2), 4))
    except (QhullError, ValueError):
        angles = np.array([0.])
    best = None
    for a in angles:
        axes = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        local = (xy - mid) @ axes.T
        lo, hi = np.percentile(local, low, axis=0), np.percentile(local, high, axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, axes, lo, hi)
    _, axes, lo, hi = best
    centre_xy = mid + ((lo + hi) / 2) @ axes
    top = float(np.percentile(points[:, 2], high))
    sizes = hi - lo
    if sizes[1] > sizes[0]:
        axes, sizes = axes[::-1], sizes[::-1]
    height = top - table_top
    return Cluster(points, np.array([*centre_xy, table_top + height / 2]), axes, np.array([*sizes, height]))

