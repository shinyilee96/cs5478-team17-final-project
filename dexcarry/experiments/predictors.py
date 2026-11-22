"""Loads the learned predictors P and E, each checked against its SHA-256. P was fitted on 473 plans of 23 pairs,
E on 236 plans of the 11 training pairs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from dexcarry.paths import WEIGHTS

# Fixed before any plan-choice test data was collected
# The robust filter's rank is applied to every accepted plan (a pair may have no plan it keeps); fixed 17 Oct
# before the predictor was fitted.
TRAIN_PAIRS = ["cube-s+tall", "stick+tall", "slim+cube-s", "tall+slab", "dgn-bottle+tall", "dgn-carton+tall",
               "ycb-sugar-box+tall", "cube-s+dgn-bottle", "bar-2cm+cube-s", "bar-4cm+tall", "stick+slab", "dgn-carton+cube-s"]
TEST_PAIRS = ["dgn-can+tall", "dgn-can+dgn-bottle", "stick+dgn-can", "bar-3cm+tall", "bar-3cm+slab", "tall-heavy+cube-s",
              "slim+tall-heavy", "cube-s+can", "stick+can", "dgn-bottle+can", "cube-s+dgn-can", "tall-heavy+slab"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_rules(folder: Path):
    """The frozen predictor and best feature of the plan-choice experiment, after checking the model's SHA-256 against the record."""
    record = json.loads((folder / "frozen.json").read_text(encoding="utf-8"))
    path = folder / record["rule_P"]["model"]
    if _sha256(path) != record["rule_P"]["sha256"]:
        raise SystemExit(f"{path} does not match its frozen SHA-256")
    return pickle.loads(path.read_bytes()), record["rule_F"]["feature"], record["rule_F"]["sign"], record


# Estimator E of the errand
# ``choose_rules`` on TRAIN_PAIRS' standard-setting labels only: no held-out object. Its output, clipped to [0, 1], is a
# plan's chance of holding both objects. What a failure leaves comes from the same labels' failure codes, pooled, with the
# refusals dropped. Calibration is reported by leave-one-pair-out on those pairs. The replay rank still
# picks the plan; E only gives its odds. New pairs have no labels, so nothing here is ever fitted on them.
ESTIMATOR_OUT = WEIGHTS / "success_estimator"


def estimate(model, X) -> np.ndarray:
    """E's chance that a plan holds both objects."""
    return np.clip(model.score(X), 0., 1.)


def frozen_estimator(folder: Path = ESTIMATOR_OUT):
    """E and its record, after checking the model's SHA-256 against the record."""
    record = json.loads((folder / "frozen.json").read_text(encoding="utf-8"))
    path = folder / record["model"]
    if _sha256(path) != record["sha256"]:
        raise SystemExit(f"{path} does not match its frozen SHA-256")
    return pickle.loads(path.read_bytes()), record

