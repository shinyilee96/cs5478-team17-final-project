"""Finger controllers for the pickup. At every step of each grasp the pickup asks the controller for the hand target:
``ScriptedClose`` (the default) stops each finger at contact, ``TunedClose`` changes its constants, and ``FeedbackClose``
flexes a finger further when its contact force drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.closing import RAMP_SECONDS, SQUEEZE_SECONDS

STAGES = ("ramp", "squeeze", "lift", "settle", "hold")


@dataclass(frozen=True)
class ClosingParams:
    """Constants of the scripted closing rule. None keeps the hand configuration's value."""

    stop_force: float | None = None # N: a finger halts once its force passes this
    squeeze: float | None = None # rad: extra flexion of halted fingers after the ramp
    ramp_seconds: float = RAMP_SECONDS
    squeeze_seconds: float = SQUEEZE_SECONDS
    wrap_overdrive: float = 0. # rad added to the wrapping finger's closed flexion

    def hand_config(self, cfg: dict) -> dict:
        """The hand configuration the contact-stop rule should use."""
        out = dict(cfg)
        if self.stop_force is not None:
            out["contact_stop_force"] = float(self.stop_force)
        if self.squeeze is not None:
            out["close_squeeze"] = float(self.squeeze)
        return out

class ScriptedClose:
    """The contact-stop rule as it is: command what the pickup proposes."""

    def __init__(self, params: ClosingParams | None = None):
        self.params = params or ClosingParams()

    def reset(self):
        pass

    def __call__(self, point: dict) -> np.ndarray:
        return point["scripted"]


class TunedClose(ScriptedClose):
    """Scripted rule with other constants (a grid point of the tuned baseline)."""

    def __init__(self, stop_force=None, squeeze=None, ramp_seconds=RAMP_SECONDS, wrap_overdrive=0.):
        super().__init__(ClosingParams(stop_force=stop_force, squeeze=squeeze, ramp_seconds=ramp_seconds,
                                       wrap_overdrive=wrap_overdrive))


@dataclass
class FeedbackClose:
    """Flex a finger further while its contact force on its object is below ``drop`` of its peak.

    From the squeeze onward, each finger holding the object of the current phase gains
    ``rate`` rad of flexion per control step while its force is low, up to ``limit``. The extra
    flexion is kept for the rest of the pickup. Force is the simulator's normal force summed over
    the finger's links: a stand-in for touch sensing, named as such in every result.
    """

    drop: float = .5
    rate: float = .01
    limit: float = .3
    params: ClosingParams = field(default_factory=ClosingParams)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.peak: dict[tuple, float] = {}
        self.extra: dict[tuple, float] = {}

    def __call__(self, point: dict) -> np.ndarray:
        q = np.array(point["scripted"], float)
        env = point["env"]
        cfg = env.hand_config
        order, groups = cfg["joint_order"], cfg["finger_groups"]
        flex = np.array([j in cfg["flexion_joints"] for j in order])
        if point["stage"] != "ramp":
            forces = env.finger_contacts(point["object"])
            for finger in point["fingers"]:
                key = (point["phase"], finger)
                self.peak[key] = max(self.peak.get(key, 0.), forces[finger])
                if self.peak[key] > cfg["contact_stop_force"] and forces[finger] < self.drop * self.peak[key]:
                    self.extra[key] = min(self.limit, self.extra.get(key, 0.) + self.rate)
        for (_, finger), extra in self.extra.items():
            idx = [order.index(j) for j in groups[finger]]
            q[idx] += extra * flex[idx]
        lo, hi = point["limits"]
        return np.clip(q, lo, hi)


def drive(steps, controller):
    """Run a pickup generator to its end, answering every decision point with ``controller``."""
    try:
        point = next(steps)
        while True:
            point = steps.send(controller(point))
    except StopIteration as stop:
        return stop.value
