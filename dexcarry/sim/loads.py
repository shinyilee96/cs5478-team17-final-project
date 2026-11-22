"""The load on a held object in the palm frame (gravity and base acceleration), in units of g."""

from __future__ import annotations

import numpy as np

G = 9.81
# One definition of "still held" for the lift test, the load map and the mounted carry:
# the object may neither translate nor pivot in the palm by more than these amounts.
SLIP_TRANSLATION = 0.03
SLIP_ROTATION_DEG = 20.0
GRAVITY_WORLD = np.array([0., 0., -1.])


def anchor_friction(p, body) -> None:
    """Stop PyBullet's slow sliding of bodies held by friction.

    Its default friction acts on velocity only, so an object gripped by a still hand
    turns about a degree per second, more with a larger time step. Friction anchors
    correct the accumulated drift at position level. Every body that grips or is
    gripped needs this, in the floating tests and on the robot alike.
    """
    for link in range(-1, p.getNumJoints(body)):
        p.changeDynamics(body, link, frictionAnchor=1)


def set_contact_friction(p, body, contact_friction: float, object_friction: float = 0.8, links=None) -> float:
    """Give the gripping ``body`` the coefficient that makes hand times object equal ``contact_friction``.

    PyBullet multiplies the coefficients of the two bodies in contact, and a hand left at the
    default 0.5 gives 0.4 against our objects at 0.8. Changing the hand and not the object keeps
    the friction between object and table as it was. ``links`` limits the change to those links
    (the hand of a mounted robot); the default is every link. Returns the coefficient set."""
    value = float(contact_friction) / float(object_friction)
    for link in (range(-1, p.getNumJoints(body)) if links is None else links):
        p.changeDynamics(body, link, lateralFriction=value)
    return value


def rotation_between_deg(a, b) -> float:
    """Angle of the rotation taking frame ``a`` to frame ``b`` (3x3 or 4x4 inputs), in degrees."""
    a, b = np.asarray(a, float)[:3, :3], np.asarray(b, float)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def load_directions(n: int = 14) -> np.ndarray:
    """Fixed palm-frame unit directions: six axes, plus eight cube corners when n=14."""
    if n not in (6, 14):
        raise ValueError("supported direction sets have 6 or 14 members")
    axes = np.vstack([np.eye(3), -np.eye(3)])
    if n == 6:
        return axes
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) / np.sqrt(3)
    return np.vstack([axes, corners])


def specific_load(times, positions, rotations, *, quasi_static=False) -> np.ndarray:
    """(T,3) load in g, palm frame, from world palm-point positions and rotations.

    ``quasi_static`` keeps only the reoriented gravity term; use it for kinematic
    arm interpolation whose finite-difference acceleration is not a command.
    """
    times, positions, rotations = (np.asarray(v, float) for v in (times, positions, rotations))
    n = len(times)
    if positions.shape != (n, 3) or rotations.shape != (n, 3, 3) or n < 1:
        raise ValueError("aligned times, positions and rotation matrices required")
    world = np.tile(GRAVITY_WORLD, (n, 1))
    if not quasi_static:
        if n < 3 or np.any(np.diff(times) <= 0):
            raise ValueError("inertial loads need at least three increasing timestamps")
        velocity = np.gradient(positions, times, axis=0)
        world = world - np.gradient(velocity, times, axis=0) / G
    return np.einsum("tji,tj->ti", rotations, world)


def smooth(values, window: int) -> np.ndarray:
    """Centred moving average for measured trajectories before differentiation."""
    values = np.asarray(values, float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])], axis=1)


def required_load(loads, directions) -> np.ndarray:
    """Largest load component demanded along each direction over the trajectory (g)."""
    loads, directions = np.asarray(loads, float), np.asarray(directions, float)
    return np.maximum(loads @ directions.T, 0.).max(axis=0)


def load_summary(loads, directions=None) -> dict:
    loads = np.asarray(loads, float)
    directions = load_directions() if directions is None else np.asarray(directions, float)
    magnitude = np.linalg.norm(loads, axis=1)
    peak = int(magnitude.argmax())
    return {"peak_g": float(magnitude[peak]), "peak_direction": (loads[peak] / max(magnitude[peak], 1e-9)).tolist(),
            "final_direction": (loads[-1] / max(magnitude[-1], 1e-9)).tolist(),
            "required_g": required_load(loads, directions).tolist()}


def subsample(loads, n: int = 32) -> np.ndarray:
    loads = np.asarray(loads, float)
    if len(loads) <= n:
        return loads
    return loads[np.linspace(0, len(loads) - 1, n).round().astype(int)]

