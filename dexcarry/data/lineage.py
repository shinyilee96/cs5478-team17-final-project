"""Records of candidate grasps and of the trials run on them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import numpy as np

from ..grasp.grasp_vector import GRASP_DIM

STAGES = ("grasp", "execution", "transport", "placement")


class Failure(str, Enum):
    GRASP_FAIL = "GRASP_FAIL"
    IK_UNREACHABLE = "IK_UNREACHABLE"
    APPROACH_COLLISION = "APPROACH_COLLISION"
    APPROACH_DISTURBED = "APPROACH_DISTURBED"
    MOUNTED_LIFT_FAIL = "MOUNTED_LIFT_FAIL"
    TUCK_FAIL = "TUCK_FAIL"
    REORIENT_SLIP = "REORIENT_SLIP"
    TRANSPORT_SLIP = "TRANSPORT_SLIP"
    TRANSPORT_DROP = "TRANSPORT_DROP"
    PLACEMENT_UNREACHABLE = "PLACEMENT_UNREACHABLE"
    PLACEMENT_COLLISION = "PLACEMENT_COLLISION"
    PLACEMENT_RELEASE_FAIL = "PLACEMENT_RELEASE_FAIL"
    PLACEMENT_TARGET_MISS = "PLACEMENT_TARGET_MISS"
    PLACEMENT_TIPPED = "PLACEMENT_TIPPED"


FAILURE_STAGE = {
    Failure.GRASP_FAIL: "grasp",
    **{f: "execution" for f in (Failure.IK_UNREACHABLE, Failure.APPROACH_COLLISION, Failure.APPROACH_DISTURBED,
                                Failure.MOUNTED_LIFT_FAIL, Failure.TUCK_FAIL)},
    **{f: "transport" for f in (Failure.REORIENT_SLIP, Failure.TRANSPORT_SLIP, Failure.TRANSPORT_DROP)},
    **{f: "placement" for f in (Failure.PLACEMENT_UNREACHABLE, Failure.PLACEMENT_COLLISION,
                                Failure.PLACEMENT_RELEASE_FAIL, Failure.PLACEMENT_TARGET_MISS,
                                Failure.PLACEMENT_TIPPED)},
}


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CandidateRecord:
    object_id: str
    observation_id: str
    grasp: tuple[float, ...]
    source: str
    checkpoint: str
    candidate_id: str = field(default_factory=lambda: str(uuid4()))
    frame_version: str = "object-palm-6d-leap-v1"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        g = np.asarray(self.grasp, dtype=float)
        if g.shape != (GRASP_DIM,) or not np.isfinite(g).all():
            raise ValueError("grasp must contain 25 finite values")
        a, b = g[3:6], g[6:9]
        if np.linalg.norm(a) < 1e-8 or np.linalg.norm(np.cross(a, b)) < 1e-8:
            raise ValueError("degenerate 6D rotation")
        object.__setattr__(self, "grasp", tuple(float(v) for v in g))
        for name in ("object_id", "observation_id", "source", "checkpoint", "candidate_id", "frame_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        json_text(self.metadata)


@dataclass(frozen=True)
class TrialRecord:
    candidate_id: str
    scenario_id: str
    repeat_seed: int
    method: str
    simulator: str
    # None = not evaluated, not a negative label.
    grasp: bool | None = None
    execution: bool | None = None
    transport: bool | None = None
    placement: bool | None = None
    first_failure: str | None = None
    scenario: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    trial_id: str = field(default_factory=lambda: str(uuid4()))
    protocol: str = "diagnostic"

    def __post_init__(self):
        for name in ("candidate_id", "scenario_id", "method", "simulator", "trial_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        stopped = False
        failed_stage = None
        if self.protocol not in ("diagnostic", "end_to_end"):
            raise ValueError("unknown trial protocol")
        if self.protocol == "end_to_end" and self.grasp is not None:
            raise ValueError("end-to-end trials have no floating-hand grasp")
        for stage in STAGES if self.protocol == "diagnostic" else STAGES[1:]:
            value = getattr(self, stage)
            if value is not None and type(value) is not bool:
                raise ValueError("stage values must be bool or None")
            if stopped and value is not None:
                raise ValueError("a stage after a failed or missing one cannot have a result")
            if value is False:
                failed_stage = stage
            stopped |= value is not True
        if failed_stage:
            if not self.first_failure or FAILURE_STAGE[Failure(self.first_failure)] != failed_stage:
                raise ValueError("failure code must agree with the first failed stage")
        elif self.first_failure is not None:
            raise ValueError("a failure code needs a failed stage")
        for value in (self.scenario, self.diagnostics, self.provenance):
            json_text(value)

    @property
    def complete(self) -> bool:
        return self.first_failure is not None or self.placement is True

