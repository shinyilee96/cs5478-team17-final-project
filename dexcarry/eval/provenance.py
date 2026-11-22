"""Saves the settings, seeds and code versions of each run next to its output."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import REPO_ROOT


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest(config: dict, *, seeds: list[int], checkpoint: str | None = None,
                 data_manifest: str | None = None) -> dict:
    def git(*args):
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "pybullet", "mujoco", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Content digest includes untracked source, since commit alone cannot identify this checkout.
    sources = {}
    for folder in ("dexcarry", "configs", "scripts"):
        for path in sorted((REPO_ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml"):
                sources[path.relative_to(REPO_ROOT).as_posix()] = file_digest(path)
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "config": config, "seeds": seeds, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "source_sha256": source_digest,
            "python": platform.python_version(), "packages": packages,
            "checkpoint": checkpoint, "checkpoint_sha256": file_digest(checkpoint) if checkpoint else None,
            "data_manifest": data_manifest, "data_manifest_sha256": file_digest(data_manifest) if data_manifest else None}


def save_manifest(path: str | Path, manifest: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def begin_run(folder: str | Path, name: str, config: dict, *, seeds: list[int], files=()) -> Path:
    """T14: one process's provenance, written as it starts, so a crash leaves a record with no end time. ``files``: the code and
    input files that matter, each hashed as it was when the run began. A resumed run writes a new file. Returns its path."""
    now = datetime.now().astimezone()
    manifest = run_manifest(config, seeds=seeds)
    hashed = {Path(path).as_posix(): file_digest(path) if Path(path).exists() else None for path in files}
    manifest.update(command=sys.argv, started=now.isoformat(timespec="seconds"), ended=None, outcome=None, files=hashed,
                    machine={"host": platform.node(), "system": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count()})
    path = Path(folder) / f"{name}-{now:%Y%m%d-%H%M%S}.json"
    save_manifest(path, manifest)
    return path


def end_run(path: str | Path, **outcome) -> None:
    """Close the record ``begin_run`` wrote."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest.update(ended=datetime.now().astimezone().isoformat(timespec="seconds"), outcome=outcome)
    save_manifest(path, manifest)
