"""Base motion profiles for the carry, and measurements of the carried object during the drive."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..envs.fetch import FETCH_BASE_JOINTS
from ..grasp.grasp_vector import invert_transform
from .loads import (
    SLIP_ROTATION_DEG,
    SLIP_TRANSLATION,
    load_summary,
    rotation_between_deg,
    smooth,
    specific_load,
)


@dataclass(frozen=True)
class MotionProfile:
    distance: float = 1.0
    speed: float = 0.25
    acceleration: float = 0.3
    braking: float = 0.5
    turn_angle: float = 0.0
    turn_rate: float = 0.3
    turn_acceleration: float = 0.4

    def __post_init__(self):
        if any(not np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("motion parameters must be finite")
        if self.distance <= 0 or min(self.speed, self.acceleration, self.braking,
                                     self.turn_rate, self.turn_acceleration) <= 0:
            raise ValueError("distance, speed and motion limits must be positive")


def _segment(distance: float, speed: float, accel: float, brake: float, dt: float):
    # Analytic triangular/trapezoidal speed profile with exact final displacement.
    peak = min(speed, np.sqrt(2 * distance / (1 / accel + 1 / brake)))
    ta, tb = peak / accel, peak / brake
    cruise = max(0.0, (distance - peak * (ta + tb) / 2) / peak)
    duration = ta + cruise + tb
    t = np.linspace(0, duration, max(2, int(np.ceil(duration / dt)) + 1))
    x = np.where(t < ta, 0.5 * accel * t**2,
                 np.where(t < ta + cruise, 0.5 * peak * ta + peak * (t - ta),
                          distance - 0.5 * brake * (duration - t)**2))
    return t, x


def base_trajectory(start: np.ndarray, profile: MotionProfile, dt: float, travel_yaw: float = 0.0):
    if dt <= 0:
        raise ValueError("dt must be positive")
    t, x = _segment(profile.distance, profile.speed, profile.acceleration, profile.braking, dt)
    pose = np.tile(np.asarray(start, dtype=float), (len(t), 1))
    pose[:, :2] += x[:, None] * np.array([np.cos(start[2] + travel_yaw), np.sin(start[2] + travel_yaw)])
    if profile.turn_angle:
        tt, angle = _segment(abs(profile.turn_angle), profile.turn_rate,
                             profile.turn_acceleration, profile.turn_acceleration, dt)
        yaw = np.sign(profile.turn_angle) * np.r_[angle[1:], angle[-1] - angle[1:]]
        turn = np.tile(pose[-1], (len(yaw), 1))
        turn[:, 2] += yaw
        pose = np.vstack([pose, turn])
        t = np.concatenate([t, t[-1] + tt[1:], t[-1] + tt[-1] + tt[1:]])
    return t, pose


def return_route(here: np.ndarray, home: np.ndarray, profile: MotionProfile, dt: float):
    """Base poses from ``here`` straight to ``home`` (x, y and yaw together) on the profile's speed,
    acceleration and braking; the last pose is ``home`` exactly. The return drive of a recovery."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    here, home = np.asarray(here, dtype=float), np.asarray(home, dtype=float)
    distance = float(np.linalg.norm(home[:2] - here[:2]))
    if distance < 1e-9:
        return np.zeros(1), home[None].copy()
    t, x = _segment(distance, profile.speed, profile.acceleration, profile.braking, dt)
    poses = here + (x / distance)[:, None] * (home - here)
    poses[-1] = home
    return t, poses


def trajectory_features(t: np.ndarray, poses: np.ndarray) -> dict:
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("at least three increasing timestamps required")
    velocity = np.gradient(poses, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    tangent_accel = np.sum(acceleration[:, :2] * velocity[:, :2], axis=1) / np.maximum(speed, 1e-8)
    return {"duration": float(t[-1] - t[0]), "peak_speed": float(speed.max()),
            "peak_linear_acceleration": float(np.linalg.norm(acceleration[:, :2], axis=1).max()),
            "peak_turn_rate": float(np.abs(velocity[:, 2]).max()),
            "peak_angular_acceleration": float(np.abs(acceleration[:, 2]).max()),
            "peak_braking": float(max(0, -tangent_accel.min()))}


class CarryMonitor:
    """Object-in-palm drift and measured palm motion, shared by reorientation and driving.

    One reference taken after the lift covers the whole carry, so slip during the
    arm reorientation and slip during base motion are judged against the same pose.
    """

    def __init__(self, env, *, slip_tolerance=SLIP_TRANSLATION, drop_tolerance=0.12,
                 slip_rotation_deg=SLIP_ROTATION_DEG):
        if not 0 < slip_tolerance < drop_tolerance or slip_rotation_deg <= 0:
            raise ValueError("require 0 < slip threshold < drop threshold and a positive rotation limit")
        self.env, self.slip_tolerance, self.drop_tolerance = env, slip_tolerance, drop_tolerance
        self.slip_rotation_deg, self.rotation = slip_rotation_deg, [0.0]
        palm = env.get_link_transform(env.hand_root_link)
        self.reference = invert_transform(palm) @ env.get_object_transform()
        self.times, self.drift, self.base, self.palm = [0.0], [0.0], [env.get_base_pose()], [palm]
        self.failure_time = None

    def step(self, action) -> bool:
        """Advance one control step; False once the object has left the hand."""
        env = self.env
        env.step(action)
        palm = env.get_link_transform(env.hand_root_link)
        relative = invert_transform(palm) @ env.get_object_transform()
        self.drift.append(float(np.linalg.norm(relative[:3, 3] - self.reference[:3, 3])))
        self.rotation.append(rotation_between_deg(self.reference, relative))
        self.times.append(len(self.times) * env.dt)
        self.base.append(env.get_base_pose())
        self.palm.append(palm)
        slipping = self.drift[-1] > self.slip_tolerance or self.rotation[-1] > self.slip_rotation_deg
        if slipping and self.failure_time is None:
            self.failure_time = self.times[-1]
        return self.drift[-1] <= self.drop_tolerance

    def result(self, start=0) -> dict:
        """Held means neither sliding nor pivoting in the palm beyond the slip limits."""
        maximum, turned = max(self.drift[start:]), max(self.rotation[start:])
        slipped = maximum > self.slip_tolerance or turned > self.slip_rotation_deg
        return {"retained": not slipped, "slipped": slipped,
                "dropped": maximum > self.drop_tolerance, "max_relative_displacement": maximum,
                "max_relative_rotation_deg": turned, "failure_time": self.failure_time}

    def loads(self, start=0, window=9) -> np.ndarray | None:
        """Measured palm-frame load (g) at the held point; None when too short to differentiate."""
        palm = np.asarray(self.palm[start:])
        if len(palm) < 3:
            return None
        points = palm[:, :3, 3] + palm[:, :3, :3] @ self.reference[:3, 3]
        return specific_load(self.times[start:], smooth(points, window), palm[:, :3, :3])


def execute_transport(env, profile: MotionProfile, hold_action: np.ndarray, *, slip_tolerance=SLIP_TRANSLATION,
                      drop_tolerance=0.12, monitor: CarryMonitor | None = None, feedback=None) -> dict:
    """Drive the base template. ``feedback(monitor, action)`` may return an adjusted
    action and a progress rate in (0, 1]; without it the template is replayed as is."""
    monitor = monitor or CarryMonitor(env, slip_tolerance=slip_tolerance, drop_tolerance=drop_tolerance)
    start = len(monitor.times) - 1
    t, poses = base_trajectory(env.get_base_pose(), profile, env.dt, env.scenario.travel_yaw)
    base_dofs, last, progress = env.dofs(FETCH_BASE_JOINTS), len(poses) - 1, 0.0
    for _ in range(4 * last):
        a, rate = np.array(hold_action, copy=True), 1.0
        if feedback is not None:
            a, rate = feedback(monitor, a)
        progress = min(last, progress + float(np.clip(rate, 0.05, 1.0)))
        lower = int(np.floor(progress))
        upper = min(last, lower + 1)
        a[base_dofs] = poses[lower] + (progress - lower) * (poses[upper] - poses[lower])
        if not monitor.step(a) or progress >= last:
            break
    times = np.asarray(monitor.times[start:]) - monitor.times[start]
    actual = np.asarray(monitor.base[start:])
    loads = monitor.loads(start)
    return {**monitor.result(start), "requested": asdict(profile),
            "commanded_features": trajectory_features(t, poses),
            "measured_features": trajectory_features(times, actual) if len(times) >= 3 else None,
            "measured_load": load_summary(loads) if loads is not None else None,
            "time": times.tolist(), "base_pose": actual.tolist(),
            "relative_displacement": monitor.drift[start:]}
