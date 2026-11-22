"""The plan library's labelled pairs (the perception experiment's pairs are read from it)."""

from __future__ import annotations

from pathlib import Path

from dexcarry.paths import INPUTS

# The plan library: plans, dry runs, replays and robot labels of the earlier pairs.
LIBRARY = INPUTS / "plan_library"


def library_pairs(library: Path = LIBRARY):
    """The old library's pairs that have labels, in name order."""
    return sorted(p.stem.removeprefix("robot-labels-") for p in library.glob("robot-labels-*.json")
                  if "mu0.8" not in p.stem and "finestep" not in p.stem)

