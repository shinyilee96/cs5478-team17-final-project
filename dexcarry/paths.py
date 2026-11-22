"""Folder locations of the release (inputs, weights, results, runs), relative to the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"

# Some frozen inputs name the folder a plan library lived in when they were written. They are kept byte for byte
# (their checksums are recorded), so those names are mapped here.
STORED_FOLDERS = {
    "runs/library": INPUTS / "plan_library",
    "runs/choose": INPUTS / "plan_choice_library",
    "runs/errand_lib": INPUTS / "errand_plan_library",
    "runs/errand_lib_a": INPUTS / "errand_plan_library_2",
}


def stored(folder: str) -> Path:
    """The folder that a frozen input names as ``folder`` (for example ``runs/errand_lib``)."""
    key = folder.replace("\\", "/").rstrip("/")
    if key not in STORED_FOLDERS:
        raise KeyError(f"no input folder for {folder!r}")
    return STORED_FOLDERS[key]
