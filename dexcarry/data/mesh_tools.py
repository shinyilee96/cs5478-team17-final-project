"""Small mesh helpers: split a CoACD file into convex pieces, load an OBJ, sample its surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def split_obj_groups(path: str | Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Split an OBJ into its ``o``/``g`` groups; each piece gets re-indexed vertices."""
    groups: list[tuple[str, list[list[int]]]] = []
    verts: list[list[float]] = []
    current_faces: list[list[int]] | None = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith(("o ", "g ")):
                name = line[2:].strip() or f"group_{len(groups)}"
                current_faces = []
                groups.append((name, current_faces))
            elif line.startswith("f "):
                if current_faces is None:
                    current_faces = []
                    groups.append(("group_0", current_faces))
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    current_faces.append([idx[0], idx[k], idx[k + 1]])
    verts_arr = np.asarray(verts, dtype=float)
    out = []
    for name, faces in groups:
        if not faces:
            continue
        faces_arr = np.asarray(faces, dtype=np.int64)
        used = np.unique(faces_arr)
        remap = -np.ones(len(verts_arr), dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append((name, verts_arr[used], remap[faces_arr]))
    return out


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray, name: str = "mesh") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def split_coacd_pieces(allinone_obj: str | Path, out_dir: str | Path) -> list[Path]:
    """Write one OBJ per convex piece of a CoACD all-in-one file; returns the piece paths.

    Skips work when the pieces already exist.
    """
    allinone_obj = Path(allinone_obj)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("piece_*.obj"))
    if existing:
        return existing
    pieces = split_obj_groups(allinone_obj)
    paths = []
    for k, (_, v, fcs) in enumerate(pieces):
        p = out_dir / f"piece_{k:03d}.obj"
        write_obj(p, v, fcs, name=f"piece_{k}")
        paths.append(p)
    return paths


def face_areas_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    return areas, normals


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface samples with face normals (like trimesh.sample.sample_surface)."""
    areas, normals = face_areas_normals(vertices, faces)
    probs = areas / max(areas.sum(), 1e-12)
    fidx = rng.choice(len(faces), size=n, p=probs)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    a = vertices[faces[fidx, 0]]
    b = vertices[faces[fidx, 1]]
    c = vertices[faces[fidx, 2]]
    pts = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    return pts, normals[fidx]

