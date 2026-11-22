"""Builds data/ from the original datasets and checks it against setup/prepared_data.sha256.

    python setup/prepare_data.py --dexgraspnet datasets/DGN_2k/processed_data --scanned-objects datasets/mujoco_scanned_objects/models
    python setup/prepare_data.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import trimesh

# Run as ``python -I`` (the datasets are downloaded data), which leaves the script's folder off the import path, so the
# repository root is added here, before the package is imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dexcarry.config import Paths
from dexcarry.data.catalog import ObjectSpec, save_catalog
from dexcarry.data.mesh_tools import split_coacd_pieces
from dexcarry.data.standing import stand_mesh

RECIPES = ROOT / "setup" / "objects.json"
PREPARED = ROOT / "setup" / "prepared_data.sha256"
ASSETS = ROOT / "setup" / "robot_assets.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(recipe: dict, out: Path, dexgraspnet: Path | None, scanned: Path | None, course: Path, scratch: Path):
    """The catalog entry of one object, writing its meshes under ``out/meshes``."""
    kind, meshes = recipe["kind"], out / "meshes"
    if kind == "box":
        return ObjectSpec(recipe["id"], half_extents=tuple(recipe["half_extents"]), mass=recipe["mass"], friction=recipe["friction"])
    if kind == "cylinder":
        raw = scratch / f"{recipe['id']}.obj"
        trimesh.creation.cylinder(radius=recipe["width_m"] / 2, height=recipe["height_m"], sections=32).export(raw)
        return stand_mesh(recipe["id"], raw, [raw], meshes, mass=recipe["mass"], friction=recipe["friction"])
    if kind == "dexgraspnet":
        if dexgraspnet is None:
            raise SystemExit("--dexgraspnet is needed (the DGN_2k_processed folder)")
        folder = dexgraspnet / recipe["source"]
        visual, pieces = folder / "mesh" / "simplified.obj", sorted((folder / "urdf" / "meshes").glob("*.obj"))
    elif kind == "scanned":
        if scanned is None:
            raise SystemExit("--scanned-objects is needed (the models folder of mujoco_scanned_objects)")
        folder = scanned / recipe["source"]
        visual, pieces = folder / "model.obj", sorted(folder.glob("model_collision_*.obj"))
    elif kind == "ycb":

        folder = course / "assets" / "ycb_objects" / recipe["source"]
        visual, pieces = folder / "textured.obj", []
        for piece in split_coacd_pieces(folder / "coacd_allinone.obj", scratch / "coacd_raw" / recipe["source"]):
            hull = scratch / "collision_pieces" / recipe["source"] / piece.name
            hull.parent.mkdir(parents=True, exist_ok=True)
            trimesh.load(str(piece), force="mesh", process=False).convex_hull.export(hull)
            pieces.append(hull)
    else:
        raise ValueError(kind)
    if not visual.exists() or not pieces:
        raise SystemExit(f"{recipe['id']}: source files not found in {folder}")
    return stand_mesh(recipe["id"], visual, pieces, meshes, scale=recipe["scale"], mass=recipe["mass"], friction=recipe["friction"])


def prepare(dexgraspnet: Path | None, scanned: Path | None) -> None:
    paths = Paths.load()
    data = paths.data_dir
    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        for catalog, entries in recipes.items():
            out = data / Path(catalog).parent
            specs = [build(r, out, dexgraspnet, scanned, paths.default_pnr_dir, Path(tmp)) for r in entries]
            save_catalog(data / catalog, specs)
            print(f"{catalog}: {len(specs)} objects")


def verify() -> bool:
    """Every prepared file and robot description file against its recorded SHA-256."""
    paths = Paths.load()
    ok = True
    for listing, base, what in ((PREPARED, paths.data_dir, "prepared data"), (ASSETS, ROOT / "assets", "robot assets")):
        bad = []
        for line in listing.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            path = base / name
            if not path.exists() or sha256(path) != digest:
                bad.append(name)
        total = len(listing.read_text(encoding="utf-8").splitlines())
        print(f"{what}: {total - len(bad)} of {total} files match their checksums" + (f"; missing or different: {bad[:5]}" if bad else ""))
        ok &= not bad
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dexgraspnet", type=Path, help="the DGN_2k_processed folder (BODex object_assets/DGN_2k_processed.tar.gz, extracted)")
    parser.add_argument("--scanned-objects", type=Path, help="the models folder of kevinzakka/mujoco_scanned_objects")
    parser.add_argument("--verify", action="store_true", help="only check the prepared files and robot assets")
    args = parser.parse_args()
    if not args.verify:
        prepare(args.dexgraspnet, args.scanned_objects)
    sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
