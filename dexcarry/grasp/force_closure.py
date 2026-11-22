"""Force-closure test and Ferrari-Canny grasp quality from contact points."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Below this the origin is on the hull boundary to within numerical noise.
MIN_EPSILON = 1e-4


def contact_wrenches(contacts, scale: float, edges: int = 8) -> np.ndarray:
    """(N*edges, 6) wrenches of friction-cone edges with unit normal force.

    ``contacts`` rows are position (3), inward unit normal (3) and friction
    coefficient, all in the object's centre-of-mass frame.
    """
    contacts = np.asarray(contacts, float).reshape(-1, 7)
    if scale <= 0:
        raise ValueError("positive torque scale required")
    wrenches = []
    for row in contacts:
        position, normal, mu = row[:3], row[3:6] / np.linalg.norm(row[3:6]), row[6]
        helper = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
        t1 = np.cross(normal, helper)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        for angle in np.linspace(0, 2 * np.pi, edges, endpoint=False):
            force = normal + mu * (np.cos(angle) * t1 + np.sin(angle) * t2)
            wrenches.append(np.r_[force, np.cross(position, force) / scale])
    return np.array(wrenches).reshape(-1, 6)


def force_closure(contacts, scale: float, edges: int = 8) -> dict:
    """``{"closed", "epsilon", "n_contacts"}`` for a set of point contacts with friction."""

    contacts = np.asarray(contacts, float).reshape(-1, 7)
    result = {"closed": False, "epsilon": 0., "n_contacts": int(len(contacts))}
    if len(contacts) < 2:
        return result
    wrenches = contact_wrenches(contacts, scale, edges)
    if np.linalg.matrix_rank(wrenches, tol=1e-6) < 6:
        return result
    try:
        hull = ConvexHull(wrenches)
    except QhullError:
        return result
    # Facets satisfy normal. x + offset <= 0 inside the hull; the origin's depth is the smallest -offset.
    epsilon = float(np.min(-hull.equations[:, -1]))
    if epsilon > MIN_EPSILON:
        result.update(closed=True, epsilon=epsilon)
    return result
