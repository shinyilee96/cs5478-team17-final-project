"""URDF editing: mesh paths, removing the Fetch gripper, mounting the hand."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FETCH_GRIPPER_LINKS = ("gripper_link", "r_gripper_finger_link", "l_gripper_finger_link")


def absolutize_mesh_paths(root: ET.Element, base_dir: Path) -> None:
    """Rewrite every ``<mesh filename>`` relative to ``base_dir`` as an absolute posix path."""
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        if fn.startswith("package://"):
            fn = fn.split("package://", 1)[1].split("/", 1)[-1]
        p = Path(fn)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        mesh.set("filename", p.as_posix())


def remove_links(root: ET.Element, link_names: tuple[str, ...]) -> list[str]:
    """Remove links and every joint whose parent or child is one of them. Returns removed joints."""
    removed = []
    for link in list(root.findall("link")):
        if link.get("name") in link_names:
            root.remove(link)
    for joint in list(root.findall("joint")):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent in link_names or child in link_names:
            removed.append(joint.get("name"))
            root.remove(joint)
    return removed


@dataclass
class MergedRobot:
    urdf_path: Path
    hand_root_link: str
    hand_joint_names: list[str]
    mount_joint: str = "hand_mount"


def merge_hand_onto_fetch(
    fetch_urdf: str | Path,
    hand_urdf_xml: str,
    hand_root_link: str,
    hand_joint_names: list[str],
    out_path: str | Path,
    *,
    mount_link: str = "wrist_roll_link",
    mount_xyz=(0.05, 0.0, 0.0),
    mount_rpy=(0.0, 0.0, np.pi),
    remove_gripper: bool = True,
    robot_name: str = "fetch_hand",
) -> MergedRobot:
    """Write a Fetch URDF whose gripper is replaced by the converted hand.

    The hand's root link is attached to ``mount_link`` with a fixed joint at ``mount_xyz``/``mount_rpy``.
    Mesh paths of both robots are made absolute so the output can live anywhere.
    """
    fetch_urdf = Path(fetch_urdf).resolve()
    fetch_root = ET.parse(fetch_urdf).getroot()
    absolutize_mesh_paths(fetch_root, fetch_urdf.parent)
    # Optical/helper frames lack inertials in the asset. Bullet otherwise assigns
    # one kilogram per frame, altering the head/base dynamics substantially.
    for link in fetch_root.findall("link"):
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", {"value": "0.000001"})
            ET.SubElement(inertial, "inertia", {"ixx": "1e-9", "iyy": "1e-9", "izz": "1e-9",
                                                "ixy": "0", "ixz": "0", "iyz": "0"})
    if remove_gripper:
        remove_links(fetch_root, FETCH_GRIPPER_LINKS)
    fetch_root.set("name", robot_name)

    hand_root = ET.fromstring(hand_urdf_xml)
    for mat in hand_root.findall("material"):
        fetch_root.append(mat)
    for link in hand_root.findall("link"):
        fetch_root.append(link)
    mount = ET.SubElement(fetch_root, "joint", {"name": "hand_mount", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": mount_link})
    ET.SubElement(mount, "child", {"link": hand_root_link})
    ET.SubElement(mount, "origin", {"xyz": " ".join(f"{float(v):.8g}" for v in mount_xyz),
                                    "rpy": " ".join(f"{float(v):.8g}" for v in mount_rpy)})
    for joint in hand_root.findall("joint"):
        fetch_root.append(joint)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(fetch_root, space="  ")
    out_path.write_text('<?xml version="1.0"?>\n' + ET.tostring(fetch_root, encoding="unicode"), encoding="utf-8")
    return MergedRobot(urdf_path=out_path, hand_root_link=hand_root_link, hand_joint_names=list(hand_joint_names))


def write_standalone_hand_urdf(hand_urdf_xml: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Several processes start at once in sharded runs. Rewriting an identical file while
    # another process is loading it makes that load fail, so write only when it differs.
    if not out_path.exists() or out_path.read_text(encoding="utf-8") != hand_urdf_xml:
        temporary = out_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(hand_urdf_xml, encoding="utf-8")
        os.replace(temporary, out_path)
    return out_path
