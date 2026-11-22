"""Paths and YAML settings. File locations go through ``Paths``; ``DEXCARRY_DATA_DIR`` moves the data folder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict (empty dict for an empty file)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


@dataclass(frozen=True)
class Paths:
    """Resolved locations of the course environment, Menagerie, data and outputs."""

    repo_root: Path
    default_pnr_dir: Path
    menagerie_dir: Path
    data_dir: Path
    outputs_dir: Path
    external_dir: Path

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Paths:
        cfg = load_yaml(config_path or CONFIG_DIR / "paths.yaml")
        data_dir = os.environ.get("DEXCARRY_DATA_DIR") or cfg.get("data_dir", "data")
        outputs_dir = os.environ.get("DEXCARRY_OUTPUTS_DIR") or cfg.get("outputs_dir", "outputs")
        return cls(
            repo_root=REPO_ROOT,
            default_pnr_dir=_resolve(REPO_ROOT, cfg.get("default_pnr_dir", "other_repos/default-pnr")),
            menagerie_dir=_resolve(REPO_ROOT, cfg.get("menagerie_dir", "other_repos/mujoco_menagerie")),
            data_dir=_resolve(REPO_ROOT, data_dir),
            outputs_dir=_resolve(REPO_ROOT, outputs_dir),
            external_dir=_resolve(REPO_ROOT, cfg.get("external_dir", "data/external")),
        )

    # Frequently used sub-paths
    @property
    def fetch_urdf(self) -> Path:
        return self.default_pnr_dir / "assets" / "fetch" / "fetch.urdf"

    @property
    def urdf_dir(self) -> Path:
        return self.data_dir / "urdf"

    @property
    def mesh_cache_dir(self) -> Path:
        return self.data_dir / "mesh_cache"

    def check(self) -> list[str]:
        """Return a list of human-readable problems (empty when everything is in place)."""
        problems = []
        if not (self.default_pnr_dir / "pick_nav_reach_env.py").exists():
            problems.append(f"default project not found at {self.default_pnr_dir}")
        if not self.fetch_urdf.exists():
            problems.append(f"Fetch URDF not found at {self.fetch_urdf}")
        if not (self.menagerie_dir / "leap_hand" / "right_hand.xml").exists():
            problems.append(f"MuJoCo Menagerie not found at {self.menagerie_dir}")
        return problems


def load_hand_config(name: str) -> dict[str, Any]:
    """Load ``configs/hands/<name>.yaml`` merged with ``<name>.auto.yaml`` if script 02 wrote one.

    The auto file holds values estimated from the kinematics (palm axes, Fetch mount). It overrides
    the hand-written file key by key, so the hand-written file keeps its comments.
    """
    cfg = load_yaml(CONFIG_DIR / "hands" / f"{name}.yaml")
    auto = CONFIG_DIR / "hands" / f"{name}.auto.yaml"
    if auto.exists():
        cfg.update(load_yaml(auto))
    return cfg

