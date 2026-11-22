"""Point clouds for the camera code, in the object frame and in metres."""

from __future__ import annotations

import numpy as np

from ..grasp.grasp_vector import apply_transform, invert_transform


def prepare_cloud(points: np.ndarray, n_points: int, *, rng: np.random.Generator,
                  T_world_object: np.ndarray | None = None, noise_std: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or n_points <= 0 or noise_std < 0:
        raise ValueError("expected (N,3) points, positive count and nonnegative noise")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("no finite object points visible")
    if T_world_object is not None:
        points = apply_transform(invert_transform(T_world_object), points)
    if noise_std:
        points = points + rng.normal(0, noise_std, points.shape)
    idx = rng.choice(len(points), n_points, replace=len(points) < n_points)
    return points[idx].astype(np.float32)
