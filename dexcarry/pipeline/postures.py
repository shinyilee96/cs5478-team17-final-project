"""Carry postures: arm shapes in front of the torso, each with wrist rolls, that set how gravity points in the palm
during the drive. ``DEFAULT`` is the fixed-posture choice.
"""

from __future__ import annotations

import numpy as np

DEFAULT = "default"
# shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll
_SHAPES = {"front_high": (0.0, -0.6, 0.0, 2.1, 0.0, 1.5), "front_tilt": (0.0, -0.75, 0.0, 2.1, 0.0, 1.2),
           "front_down": (0.0, -1.2, 0.0, 2.1, 0.0, 0.75)}
_ROLLS = {"r0": 0.0, "r90": np.pi / 2, "r180": np.pi, "r270": -np.pi / 2}
_LIBRARY = ("front_high_r0", "front_high_r90", "front_high_r180", "front_high_r270",
            "front_tilt_r0", "front_tilt_r90", "front_tilt_r270", "front_down_r0")
_DEFAULT_ORDER = ("front_high_r0", "front_down_r0")


def carry_library() -> dict[str, np.ndarray]:
    """Named 7-joint Fetch arm configurations (torso is kept from the lift)."""
    return {name: np.array([*_SHAPES[name.rsplit("_", 1)[0]], _ROLLS[name.rsplit("_", 1)[1]]]) for name in _LIBRARY}


def posture_candidates(name: str | None) -> list[tuple[str, np.ndarray]]:
    """Arm configurations to try, in order, for one requested posture."""
    library = carry_library()
    if name in (None, DEFAULT):
        return [(n, library[n]) for n in _DEFAULT_ORDER]
    if name not in library:
        raise ValueError(f"unknown carry posture {name!r}; choose from {sorted(library)}")
    return [(name, library[name])]
