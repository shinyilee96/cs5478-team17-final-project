"""Convert a MuJoCo Menagerie hand (MJCF) into a URDF that PyBullet can load."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..grasp.grasp_vector import matrix_to_rpy, quat_wxyz_to_matrix


def _floats(s: str | None, default: list[float]) -> np.ndarray:
    if s is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in s.split()], dtype=float)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.8g}" for x in np.asarray(v).reshape(-1))


@dataclass
class _Defaults:
    """Resolves MJCF default classes: class name -> tag -> attributes, with parent chain."""

    attrs: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    parent: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def parse(cls, root: ET.Element) -> _Defaults:
        d = cls()

        def walk(node: ET.Element, cname: str, parent: str | None):
            d.parent.setdefault(cname, parent)
            d.attrs.setdefault(cname, {})
            for child in node:
                if child.tag == "default":
                    walk(child, child.get("class", "main"), cname)
                else:
                    merged = dict(d.attrs[cname].get(child.tag, {}))
                    merged.update(child.attrib)
                    d.attrs[cname][child.tag] = merged

        for top in root.findall("default"):
            walk(top, top.get("class", "main"), None)
        d.attrs.setdefault("main", {})
        d.parent.setdefault("main", None)
        return d

    def resolve(self, tag: str, cname: str | None, own: dict[str, str]) -> dict[str, str]:
        chain = []
        c = cname or "main"
        while c is not None and c in self.attrs:
            chain.append(c)
            c = self.parent.get(c)
        if "main" not in chain and "main" in self.attrs:
            chain.append("main")
        out: dict[str, str] = {}
        for c in reversed(chain):
            out.update(self.attrs[c].get(tag, {}))
        out.update(own)
        return out


def _orientation_matrix(attrs: dict[str, str], angle_unit: str) -> np.ndarray:
    if "quat" in attrs:
        q = _floats(attrs["quat"], [1, 0, 0, 0])
        q = q / max(np.linalg.norm(q), 1e-12)
        return quat_wxyz_to_matrix(q)
    if "euler" in attrs:
        e = _floats(attrs["euler"], [0, 0, 0])
        if angle_unit == "degree":
            e = np.deg2rad(e)
        return Rotation.from_euler("XYZ", e).as_matrix()
    if "axisangle" in attrs:
        v = _floats(attrs["axisangle"], [0, 0, 1, 0])
        ang = v[3] if angle_unit == "radian" else np.deg2rad(v[3])
        axis = v[:3] / max(np.linalg.norm(v[:3]), 1e-12)
        return Rotation.from_rotvec(axis * ang).as_matrix()
    return np.eye(3)


@dataclass
class ConvertedHand:
    urdf_xml: str
    root_link: str
    joint_names: list[str] # in order of appearance (matches the MJCF tree order)
    joint_limits: dict[str, tuple[float, float]]
    link_names: list[str]
    body_to_link: dict[str, str]


def _geometry_xml(gattrs: dict[str, str], meshes: dict[str, tuple[Path, np.ndarray]], link_indent: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    """URDF <geometry> string, plus the geom's local pos and rotation. None if unsupported."""
    gtype = gattrs.get("type", "mesh" if "mesh" in gattrs else "sphere")
    pos = _floats(gattrs.get("pos"), [0, 0, 0])
    R = _orientation_matrix(gattrs, "radian")
    size = _floats(gattrs.get("size"), [0.0])
    if gtype == "mesh":
        mesh_name = gattrs.get("mesh")
        if mesh_name is None or mesh_name not in meshes:
            return None
        path, scale = meshes[mesh_name]
        geom = f'<mesh filename="{path.as_posix()}" scale="{_fmt(scale)}"/>'
    elif gtype == "box":
        geom = f'<box size="{_fmt(2.0 * size[:3])}"/>'
    elif gtype == "sphere":
        geom = f'<sphere radius="{size[0]:.8g}"/>'
    elif gtype == "capsule":
        if "fromto" in gattrs:
            return None
        geom = f'<capsule radius="{size[0]:.8g}" length="{2.0 * size[1]:.8g}"/>'
    elif gtype == "cylinder":
        if "fromto" in gattrs:
            return None
        geom = f'<cylinder radius="{size[0]:.8g}" length="{2.0 * size[1]:.8g}"/>'
    else:
        return None
    return geom, pos, R


def convert_mjcf_hand(
    mjcf_path: str | Path,
    *,
    prefix: str = "hand_",
    root_at_identity: bool = True,
    default_mass: float = 0.03,
    include_visual: bool = True,
) -> ConvertedHand:
    """Convert the hand MJCF at ``mjcf_path`` into URDF text.

    All mesh paths in the output are absolute, so the URDF can be written anywhere.
    """
    mjcf_path = Path(mjcf_path).resolve()
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    angle_unit = (compiler.get("angle", "degree") if compiler is not None else "degree")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_base = (mjcf_path.parent / meshdir).resolve() if not Path(meshdir).is_absolute() else Path(meshdir)

    defaults = _Defaults.parse(root)

    # Assets: meshes and materials
    meshes: dict[str, tuple[Path, np.ndarray]] = {}
    materials: dict[str, str] = {}
    for asset in root.findall("asset"):
        for m in asset.findall("mesh"):
            mattrs = defaults.resolve("mesh", m.get("class"), m.attrib)
            file = mattrs["file"]
            name = mattrs.get("name") or Path(file).stem
            scale = _floats(mattrs.get("scale"), [1, 1, 1])
            meshes[name] = ((mesh_base / file).resolve(), scale)
        for mat in asset.findall("material"):
            if "rgba" in mat.attrib:
                materials[mat.get("name")] = mat.get("rgba")

    links_xml: list[str] = []
    joints_xml: list[str] = []
    joint_names: list[str] = []
    joint_limits: dict[str, tuple[float, float]] = {}
    link_names: list[str] = []
    body_to_link: dict[str, str] = {}
    used_materials: dict[str, str] = {}

    def link_block(name: str, inertial_xml: str, geoms: list[str]) -> str:
        return f"  <link name=\"{name}\">\n{inertial_xml}{''.join(geoms)}  </link>\n"

    def inertial_xml_from(body: ET.Element, cname: str | None) -> str:
        inert = body.find("inertial")
        if inert is not None:
            iattrs = defaults.resolve("inertial", cname, inert.attrib)
            pos = _floats(iattrs.get("pos"), [0, 0, 0])
            R = _orientation_matrix(iattrs, angle_unit)
            mass = float(iattrs.get("mass", default_mass))
            if "diaginertia" in iattrs:
                d = _floats(iattrs["diaginertia"], [1e-6, 1e-6, 1e-6])
                ixx, iyy, izz, ixy, ixz, iyz = d[0], d[1], d[2], 0.0, 0.0, 0.0
            elif "fullinertia" in iattrs:
                fi = _floats(iattrs["fullinertia"], [1e-6] * 6)
                ixx, iyy, izz, ixy, ixz, iyz = fi
            else:
                ixx = iyy = izz = mass * 1e-4
                ixy = ixz = iyz = 0.0
        else:
            pos = np.zeros(3)
            R = np.eye(3)
            mass = default_mass
            ixx = iyy = izz = mass * 1e-4
            ixy = ixz = iyz = 0.0
        rpy = matrix_to_rpy(R)
        return (
            f"    <inertial>\n      <origin xyz=\"{_fmt(pos)}\" rpy=\"{_fmt(rpy)}\"/>\n"
            f"      <mass value=\"{mass:.8g}\"/>\n"
            f"      <inertia ixx=\"{ixx:.8g}\" ixy=\"{ixy:.8g}\" ixz=\"{ixz:.8g}\" iyy=\"{iyy:.8g}\" iyz=\"{iyz:.8g}\" izz=\"{izz:.8g}\"/>\n"
            f"    </inertial>\n"
        )

    def geoms_xml_from(body: ET.Element, cname: str | None) -> list[str]:
        out = []
        for g in body.findall("geom"):
            gattrs = defaults.resolve("geom", g.get("class", cname), g.attrib)
            if g.get("class") is None and cname is not None:
                # element without class: resolve through the inherited childclass
                gattrs = defaults.resolve("geom", cname, g.attrib)
            res = _geometry_xml(gattrs, meshes, "")
            if res is None:
                continue
            geom, pos, R = res
            rpy = matrix_to_rpy(R)
            contype = int(float(gattrs.get("contype", "1")))
            conaff = int(float(gattrs.get("conaffinity", "1")))
            is_collision = not (contype == 0 and conaff == 0)
            origin = f"<origin xyz=\"{_fmt(pos)}\" rpy=\"{_fmt(rpy)}\"/>"
            if include_visual:
                mat = gattrs.get("material")
                mat_xml = ""
                if mat and mat in materials:
                    used_materials[mat] = materials[mat]
                    mat_xml = f"<material name=\"{prefix}{mat}\"/>"
                out.append(f"    <visual>{origin}<geometry>{geom}</geometry>{mat_xml}</visual>\n")
            if is_collision:
                out.append(f"    <collision>{origin}<geometry>{geom}</geometry></collision>\n")
        return out

    def walk(body: ET.Element, parent_link: str | None, inherited_class: str | None, is_root: bool):
        cname = body.get("childclass", inherited_class)
        bname = body.get("name") or f"body{len(link_names)}"
        link = f"{prefix}{bname}"
        body_to_link[bname] = link
        if is_root and root_at_identity:
            pos, R = np.zeros(3), np.eye(3)
        else:
            pos = _floats(body.get("pos"), [0, 0, 0])
            R = _orientation_matrix(body.attrib, angle_unit)
        rpy = matrix_to_rpy(R)

        joints = body.findall("joint") + body.findall("freejoint")
        joints = [j for j in joints if j.tag == "joint"]
        chain_parent = parent_link
        origin_used = False
        for k, j in enumerate(joints):
            jattrs = defaults.resolve("joint", j.get("class", cname), j.attrib)
            if j.get("class") is None and cname is not None:
                jattrs = defaults.resolve("joint", cname, j.attrib)
            jpos = _floats(jattrs.get("pos"), [0, 0, 0])
            if np.linalg.norm(jpos) > 1e-9:
                raise NotImplementedError(f"joint {j.get('name')} has a non-zero pos; add an intermediate link")
            jtype = jattrs.get("type", "hinge")
            jname_raw = j.get("name") or f"{bname}_j{k}"
            jname = f"{prefix}{jname_raw}"
            axis = _floats(jattrs.get("axis"), [0, 0, 1])
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            limited = jattrs.get("limited", "auto")
            rng = _floats(jattrs.get("range"), [0.0, 0.0])
            has_range = "range" in jattrs and not np.allclose(rng, 0.0)
            if angle_unit == "degree" and jtype == "hinge":
                rng = np.deg2rad(rng)
            child_link = link if k == len(joints) - 1 else f"{link}__j{k}"
            if k < len(joints) - 1:
                links_xml.append(link_block(child_link,
                                            "    <inertial><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><mass value=\"0.001\"/>"
                                            "<inertia ixx=\"1e-7\" ixy=\"0\" ixz=\"0\" iyy=\"1e-7\" iyz=\"0\" izz=\"1e-7\"/></inertial>\n", []))
                link_names.append(child_link)
            if chain_parent is None:
                raise ValueError("root body must not carry joints for hand conversion; strip its freejoint first")
            urdf_type = "prismatic" if jtype == "slide" else ("revolute" if (has_range or limited == "true") else "continuous")
            o_xyz, o_rpy = (pos, rpy) if not origin_used else (np.zeros(3), np.zeros(3))
            origin_used = True
            limit_xml = ""
            if urdf_type in ("revolute", "prismatic"):
                lo, hi = (float(rng[0]), float(rng[1])) if has_range else (-3.1416, 3.1416)
                limit_xml = f"    <limit lower=\"{lo:.8g}\" upper=\"{hi:.8g}\" effort=\"5\" velocity=\"5\"/>\n"
                joint_limits[jname] = (lo, hi)
            damping = float(jattrs.get("damping", 0.0))
            joints_xml.append(
                f"  <joint name=\"{jname}\" type=\"{urdf_type}\">\n"
                f"    <parent link=\"{chain_parent}\"/>\n    <child link=\"{child_link}\"/>\n"
                f"    <origin xyz=\"{_fmt(o_xyz)}\" rpy=\"{_fmt(o_rpy)}\"/>\n"
                f"    <axis xyz=\"{_fmt(axis)}\"/>\n{limit_xml}"
                f"    <dynamics damping=\"{damping:.8g}\" friction=\"0\"/>\n  </joint>\n"
            )
            joint_names.append(jname)
            chain_parent = child_link
        if not joints and parent_link is not None:
            joints_xml.append(
                f"  <joint name=\"{link}_fixed\" type=\"fixed\">\n"
                f"    <parent link=\"{parent_link}\"/>\n    <child link=\"{link}\"/>\n"
                f"    <origin xyz=\"{_fmt(pos)}\" rpy=\"{_fmt(rpy)}\"/>\n  </joint>\n"
            )
        links_xml.append(link_block(link, inertial_xml_from(body, cname), geoms_xml_from(body, cname)))
        link_names.append(link)
        for child in body.findall("body"):
            walk(child, link, cname, False)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    top_bodies = worldbody.findall("body")
    if len(top_bodies) != 1:
        raise ValueError(f"expected one root body in worldbody, found {len(top_bodies)}")
    root_body = top_bodies[0]
    walk(root_body, None, root_body.get("childclass"), True)
    root_link = body_to_link[root_body.get("name")]

    mats = "".join(f"  <material name=\"{prefix}{k}\"><color rgba=\"{v}\"/></material>\n" for k, v in used_materials.items())
    urdf = (
        "<?xml version=\"1.0\"?>\n"
        f"<robot name=\"{prefix.rstrip('_') or 'hand'}\">\n{mats}{''.join(links_xml)}{''.join(joints_xml)}</robot>\n"
    )
    return ConvertedHand(urdf_xml=urdf, root_link=root_link, joint_names=joint_names,
                         joint_limits=joint_limits, link_names=link_names, body_to_link=body_to_link)
