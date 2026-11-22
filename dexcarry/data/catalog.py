"""Object catalogs: the objects a scene loads, with their meshes, mass and friction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from .mesh_tools import sample_surface


@lru_cache(maxsize=256)
def mesh_bounds(path: str, scale: float):
    return np.asarray(trimesh.load(path, force="mesh", process=False).bounds, dtype=float) * scale


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    half_extents: tuple[float, float, float] = (.03, .03, .03)
    mass: float = .15
    friction: float = .8
    visual_mesh: str | None = None
    collision_meshes: tuple[str, ...] = ()
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "half_extents", tuple(self.half_extents))
        object.__setattr__(self, "collision_meshes", tuple(self.collision_meshes))
        if not self.object_id or len(self.half_extents) != 3:
            raise ValueError("object ID and three half extents required")
        if any(c in self.object_id for c in '/\\:') or self.object_id in ('.', '..'):
            raise ValueError("object ID must not contain path characters")
        if not np.isfinite([*self.half_extents, self.mass, self.friction, self.scale]).all():
            raise ValueError("object parameters must be finite")
        if min(*self.half_extents, self.mass, self.scale) <= 0 or self.friction < 0:
            raise ValueError("invalid object parameters")
        if bool(self.visual_mesh) != bool(self.collision_meshes):
            raise ValueError("mesh objects require visual mesh and convex collision pieces")

    def create_body(self, p, position, quaternion=(0, 0, 0, 1), *, static=False):
        if self.visual_mesh:
            collision = p.createCollisionShapeArray(
                shapeTypes=[p.GEOM_MESH] * len(self.collision_meshes),
                fileNames=list(self.collision_meshes), meshScales=[[self.scale] * 3] * len(self.collision_meshes))
            visual = p.createVisualShape(p.GEOM_MESH, fileName=self.visual_mesh,
                                          meshScale=[self.scale] * 3, rgbaColor=[.9, .4, .2, 1])
        else:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=self.half_extents, rgbaColor=[.9, .4, .2, 1])
        body = p.createMultiBody(0 if static else self.mass, collision, visual,
                                 basePosition=position, baseOrientation=quaternion)
        p.changeDynamics(body, -1, lateralFriction=self.friction)
        return body

    @property
    def support_offset_z(self):
        return float(-mesh_bounds(self.visual_mesh, self.scale)[0, 2]) if self.visual_mesh else self.half_extents[2]

    def local_corners(self) -> np.ndarray:
        """(8,3) corners of the object's bounding box in its own frame."""
        low, high = (mesh_bounds(self.visual_mesh, self.scale) if self.visual_mesh else
                     (-np.asarray(self.half_extents), np.asarray(self.half_extents)))
        return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])

    def surface_cloud(self, n, seed=0):

        mesh = (trimesh.load(self.visual_mesh, force="mesh", process=False) if self.visual_mesh else
                trimesh.creation.box(extents=np.array(self.half_extents) * 2))
        vertices = np.asarray(mesh.vertices) * (self.scale if self.visual_mesh else 1.)
        return sample_surface(vertices, np.asarray(mesh.faces), n, np.random.default_rng(seed))


def load_catalog(path: str | Path) -> dict[str, ObjectSpec]:
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding="utf-8"))["objects"]
    objects = {}
    for value in entries:
        value = dict(value)
        for key in ("visual_mesh", "collision_meshes"):
            if value.get(key):
                paths = [value[key]] if key == "visual_mesh" else value[key]
                resolved = [str((path.parent / p).resolve()) for p in paths]
                if any(not Path(p).is_file() for p in resolved):
                    raise FileNotFoundError(f"missing mesh for {value['object_id']}")
                value[key] = resolved[0] if key == "visual_mesh" else resolved
        spec = ObjectSpec(**value)
        if spec.object_id in objects:
            raise ValueError("duplicate object identity")
        objects[spec.object_id] = spec
    if not objects:
        raise ValueError("empty object catalog")
    return objects


def _portable(mesh: str, folder: Path) -> str:
    """A mesh path relative to the catalog's folder, with forward slashes, so the catalog works on any
    machine and wherever the repository is cloned. ``load_catalog`` resolves it against that folder.
    A path on another drive stays absolute."""
    try:
        return Path(os.path.relpath(Path(mesh).resolve(), folder.resolve())).as_posix()
    except ValueError:
        return str(mesh)


def save_catalog(path: str | Path, objects: list[ObjectSpec]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for o in objects:
        entry = asdict(o)
        if o.visual_mesh:
            entry["visual_mesh"] = _portable(o.visual_mesh, path.parent)
            entry["collision_meshes"] = [_portable(m, path.parent) for m in o.collision_meshes]
        entries.append(entry)
    path.write_text(json.dumps({"version": 1, "objects": entries}, indent=2) + "\n", encoding="utf-8")
