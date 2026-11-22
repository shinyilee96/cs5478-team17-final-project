"""A feedback closing rule: tighten a finger by a fixed step while the object slips in the palm, up to a limit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grasp_controller import FeedbackClose


@dataclass
class SlipFeedbackClose(FeedbackClose):
    """``FeedbackClose`` with a fixed step (``rate`` 0.02 rad per control step) and its cap: a finger whose force is at or
    above the stop force (the rule's ``params.stop_force``, else the hand's ``contact_stop_force``) is not tightened."""

    rate: float = .02

    def __call__(self, point: dict) -> np.ndarray:
        q = np.array(point["scripted"], float)
        env = point["env"]
        cfg = env.hand_config
        order, groups = cfg["joint_order"], cfg["finger_groups"]
        flex = np.array([j in cfg["flexion_joints"] for j in order])
        stop = self.params.stop_force if self.params.stop_force is not None else cfg["contact_stop_force"]
        if point["stage"] != "ramp":
            forces = env.finger_contacts(point["object"])
            for finger in point["fingers"]:
                key = (point["phase"], finger)
                self.peak[key] = max(self.peak.get(key, 0.), forces[finger])
                slipping = self.peak[key] > cfg["contact_stop_force"] and forces[finger] < self.drop * self.peak[key]
                if slipping and forces[finger] < stop:
                    self.extra[key] = min(self.limit, self.extra.get(key, 0.) + self.rate)
        for (_, finger), extra in self.extra.items():
            idx = [order.index(j) for j in groups[finger]]
            q[idx] += extra * flex[idx]
        lo, hi = point["limits"]
        return np.clip(q, lo, hi)
