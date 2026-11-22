"""A slip guard for the carry, reading object drift from the simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("tighten", "slow", "both")


@dataclass
class SlipGuard:
    hand_dofs: list
    flexion_mask: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mode: str = "both"
    trigger: float = 0.008
    tighten_per_step: float = 0.01
    max_tighten: float = 0.3
    slow_rate: float = 0.5
    extra: float = 0.0
    triggered_steps: int = 0
    slowed: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"feedback mode must be one of {MODES}")
        if not 0 < self.slow_rate <= 1 or self.trigger <= 0:
            raise ValueError("invalid slip guard parameters")

    @classmethod
    def for_env(cls, env, mode="both", **kwargs):
        cfg = env.hand_config
        lower, upper = env.hand_joint_limits()
        mask = np.array([j in cfg["flexion_joints"] for j in cfg["joint_order"]])
        return cls(env.dofs(env.hand_joints), mask, lower, upper, mode, **kwargs)

    def __call__(self, monitor, action):
        if monitor.drift[-1] > self.trigger:
            self.triggered_steps += 1
            if self.mode in ("tighten", "both"):
                self.extra = min(self.max_tighten, self.extra + self.tighten_per_step)
            if self.mode in ("slow", "both"):
                self.slowed = True
        action = np.array(action, copy=True)
        hand = action[self.hand_dofs] + self.extra * self.flexion_mask
        action[self.hand_dofs] = np.clip(hand, self.lower, self.upper)
        return action, self.slow_rate if self.slowed else 1.0

    def summary(self) -> dict:
        return {"mode": self.mode, "triggered_steps": self.triggered_steps,
                "extra_flexion": self.extra, "slowed": self.slowed, "sensing": "privileged-drift"}
