"""Meshes made to stand on a table: ``stand_mesh`` scales a mesh, stands it on its widest stable face and writes it
with its convex collision pieces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .catalog import ObjectSpec
import trimesh


def standing_rotation(extents, centre_of_mass, bounds_centre) -> np.ndarray:
    """Rotation of the source frame that stands the object up: longest axis to +z with the end
    nearer the centre of mass down, and the narrower of the other two axes to y."""
    extents = np.asarray(extents, float)
    up = int(np.argmax(extents))
    others = [k for k in range(3) if k != up]
    narrow, wide = sorted(others, key=lambda k: extents[k])
    sign = -1. if (np.asarray(centre_of_mass, float) - np.asarray(bounds_centre, float))[up] > 0 else 1.
    R = np.zeros((3, 3))
    R[2, up], R[1, narrow], R[0, wide] = sign, 1., 1.
    if np.linalg.det(R) < 0:
        R[0] *= -1
    return R


def stand_mesh(object_id: str, visual: str | Path, pieces, out_dir: str | Path, *, scale=1., mass=.1,
               friction=.8) -> ObjectSpec:
    """Write the standing copy of a mesh and its convex pieces under ``out_dir/object_id`` and
    return its catalog entry (scale 1, paths absolute)."""

    def read(path):
        mesh = trimesh.load(str(path), force="mesh", process=False)
        return np.asarray(mesh.vertices, float) * float(scale), np.asarray(mesh.faces)

    whole, parts = read(visual), [read(path) for path in pieces]
    if not parts:
        raise ValueError("a mesh object needs convex collision pieces")
    hulls = [trimesh.Trimesh(v, f, process=False).convex_hull for v, f in parts]
    volumes = np.array([hull.volume for hull in hulls])
    centre = (volumes[:, None] * np.array([hull.center_mass for hull in hulls])).sum(axis=0) / volumes.sum()
    low, high = whole[0].min(axis=0), whole[0].max(axis=0)
    R = standing_rotation(high - low, centre, (low + high) / 2)
    folder = Path(out_dir).resolve() / object_id
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.obj"):
        # Pieces of an earlier object written under the same name must not join this one.
        stale.unlink()

    def write(name, vertices, faces):
        trimesh.Trimesh((vertices - centre) @ R.T, faces, process=False).export(folder / name)
        return str(folder / name)

    visual_path = write("visual.obj", *whole)
    piece_paths = [write(f"piece_{k:03d}.obj", *part) for k, part in enumerate(parts)]
    stood = (whole[0] - centre) @ R.T
    return ObjectSpec(object_id, half_extents=tuple(np.maximum((stood.max(axis=0) - stood.min(axis=0)) / 2, 1e-4)),
                      mass=float(mass), friction=float(friction), visual_mesh=visual_path, collision_meshes=tuple(piece_paths))
