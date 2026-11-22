"""Contact-stop closing: each finger stops at first contact and the touching fingers then squeeze a little more;
fingers that hook an object keep flexing.
"""

from __future__ import annotations

import numpy as np

# Simulation time of the two phases, in seconds, for both harnesses.
RAMP_SECONDS, SQUEEZE_SECONDS = 1.0, 0.5


class ContactStopClose:
    def __init__(self, cfg, start, target, lower, upper, envelop=()):
        order = cfg["joint_order"]
        self.groups = {finger: [order.index(j) for j in names] for finger, names in cfg["finger_groups"].items()}
        self.flex = np.array([j in cfg["flexion_joints"] for j in order])
        self.stop_force, self.depth = float(cfg["contact_stop_force"]), float(cfg["close_squeeze"])
        self.start, self.target = np.asarray(start, float), np.asarray(target, float)
        self.lower, self.upper = lower, upper
        unknown = set(envelop) - set(self.groups)
        if unknown:
            raise ValueError(f"unknown fingers to envelop: {sorted(unknown)}")
        self.envelop = frozenset(envelop)
        # Finger name -> joint values at which it met the object.
        self.halted: dict[str, np.ndarray] = {}

    def command(self, alpha):
        """Joint targets at fraction ``alpha`` of the ramp, with halted fingers held where they touched."""
        q = self.start + alpha * (self.target - self.start)
        for finger, values in self.halted.items():
            q[self.groups[finger]] = values
        return q

    def observe(self, forces, joints):
        """Halt every finger whose contact force (N, summed over its links) passed the threshold."""
        for finger, force in forces.items():
            if force > self.stop_force and finger not in self.halted and finger not in self.envelop:
                self.halted[finger] = np.array(joints, float)[self.groups[finger]]

    def squeeze(self, alpha):
        """Joint targets at fraction ``alpha`` of the squeeze; ``squeeze(1)`` is the grip to hold."""
        q = self.command(1.)
        for finger in self.halted:
            idx = self.groups[finger]
            q[idx] += alpha * self.depth * self.flex[idx]
        return np.clip(q, self.lower, self.upper)
