"""The Fetch robot with the LEAP hand in PyBullet: scene settings (``Scenario``) and the environment (``DexCarryEnv``).
The merged robot model is built on first use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from ..config import Paths, load_hand_config
from ..data.catalog import ObjectSpec
from ..data.preprocessing import prepare_cloud
from ..grasp.grasp_vector import transform_from_pos_quat_xyzw
from ..sim.loads import anchor_friction
from .camera import CameraIntrinsics, back_project, depth_buffer_to_metric
from .fetch import FETCH_ARM_JOINTS, FETCH_BASE_JOINTS
from .mjcf_to_urdf import convert_mjcf_hand
from .urdf_tools import merge_hand_onto_fetch

READY_ARM = (0.0, -0.6, 0.0, 2.0, 0.0, 1.5, 0.0)


@dataclass(frozen=True)
class Scenario:
    seed: int = 0
    object_id: str = "box-60mm"
    object_half_extents: tuple[float, float, float] = (0.03, 0.03, 0.03)
    object_mass: float = 0.15
    object_friction: float = 0.8
    object_visual_mesh: str | None = None
    object_collision_meshes: tuple[str, ...] = ()
    object_scale: float = 1.0
    table_height: float = 0.65
    # Half the pick table's length along y. 0.4 m for every scene before 26 Oct; the errand's counter is longer
    # (``envs.errand.ErrandEnv``).
    pick_table_half_width: float = 0.4
    # The target table continues the pick table's front edge; the base strafes
    # ``travel_yaw`` (relative to its heading) by the profile distance to reach it.
    target_xy: tuple[float, float] = (0.45, 1.0)
    travel_yaw: float = float(np.pi / 2)
    # How far the base may step toward the pick table before grasping (metres along its heading),
    # tried in this order. The planner takes the first offset with a complete pick plan; (0.0,)
    # keeps the base fixed. The drive to the target starts from wherever it stood.
    standing_offsets: tuple[float, ...] = (0.0, 0.08, 0.15, 0.22)
    target_tolerance: float = 0.08
    # Optional stricter placement: the object must end within this tilt of its initial upright axis.
    max_placement_tilt_deg: float | None = None
    camera_eye: tuple[float, float, float] = (-0.45, -0.5, 1.15)
    camera_noise_std: float = 0.0
    timestep: float = 1 / 240
    substeps: int = 4
    # PyBullet's position motor asks for a joint velocity of gain x error / time step (Bullet's
    # btMultiBodyJointMotor), so at a fixed gain a smaller step makes every motor stiffer. When set,
    # the gains are scaled by timestep / gain_reference_dt, which keeps the stiffness the motors
    # have at the reference step. None (the default) uses the gains as they are, as every run before
    # did.
    gain_reference_dt: float | None = None

    @classmethod
    def for_object(cls, spec: ObjectSpec, **kwargs):
        return cls(object_id=spec.object_id, object_half_extents=spec.half_extents,
                   object_mass=spec.mass, object_friction=spec.friction,
                   object_visual_mesh=spec.visual_mesh, object_collision_meshes=spec.collision_meshes,
                   object_scale=spec.scale, **kwargs)

    def object_spec(self):
        return ObjectSpec(self.object_id, self.object_half_extents, self.object_mass, self.object_friction,
                          self.object_visual_mesh, self.object_collision_meshes, self.object_scale)

    def __post_init__(self):
        self.object_spec()
        if any(v <= 0 or not np.isfinite(v) for v in self.object_half_extents):
            raise ValueError("object dimensions must be finite and positive")
        if self.object_mass <= 0 or self.object_friction < 0 or self.timestep <= 0 or self.substeps < 1:
            raise ValueError("invalid physics parameters")
        if self.gain_reference_dt is not None and not self.gain_reference_dt > 0:
            raise ValueError("the gain reference time step must be positive")
        if self.max_placement_tilt_deg is not None and not 0 < self.max_placement_tilt_deg <= 180:
            raise ValueError("placement tilt limit must lie in (0, 180] degrees")
        object.__setattr__(self, "standing_offsets", tuple(float(v) for v in self.standing_offsets))
        if not self.standing_offsets or any(not np.isfinite(v) for v in self.standing_offsets):
            raise ValueError("at least one finite standing offset required")
        if self.target_tolerance <= 0 or self.camera_noise_std < 0:
            raise ValueError("invalid observation/success parameters")


def build_robot(paths: Paths | None = None):
    paths = paths or Paths.load()
    cfg = load_hand_config("leap")
    converted = convert_mjcf_hand(paths.menagerie_dir / cfg["menagerie_xml"], prefix="leap_",
                                  default_mass=cfg["urdf_default_mass"])
    output = merge_hand_onto_fetch(paths.fetch_urdf, converted.urdf_xml, converted.root_link,
                                 converted.joint_names, paths.urdf_dir / "fetch_leap.urdf",
                                 mount_xyz=cfg["fetch_mount"]["xyz"],
                                 mount_rpy=cfg["fetch_mount"]["rpy"]).urdf_path
    output.with_suffix(".build.json").write_text(json.dumps({"signature": robot_signature(paths)}), encoding="utf-8")
    return output


def robot_signature(paths):
    config = load_hand_config("leap")
    sources = [paths.fetch_urdf, paths.menagerie_dir / config["menagerie_xml"],
               Path(__file__).with_name("urdf_tools.py"), Path(__file__).with_name("mjcf_to_urdf.py")]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in sources:
        digest.update(path.read_bytes())
        digest.update(str(path.resolve()).encode()) # the merged model names its mesh files by absolute path: rebuild after a move
    return digest.hexdigest()


class DexCarryEnv:
    def __init__(self, scenario: Scenario | None = None, *, gui: bool = False, paths: Paths | None = None):

        self.p = p
        self.scenario = scenario or Scenario()
        self.paths = paths or Paths.load()
        urdf = self.paths.urdf_dir / "fetch_leap.urdf"
        stamp = urdf.with_suffix(".build.json")
        if not urdf.exists() or not stamp.exists() or json.loads(stamp.read_text())["signature"] != robot_signature(self.paths):
            urdf = build_robot(self.paths)
        self.hand_config = load_hand_config("leap")
        self.hand_root_link = "leap_" + self.hand_config["root_body"]
        self.hand_joints = ["leap_" + n for n in self.hand_config["joint_order"]]
        self.fingertip_links = {f: "leap_" + n for f, n in self.hand_config["fingertip_bodies"].items()}
        self.pb_physics_client = p.connect(p.GUI if gui else p.DIRECT)
        if self.pb_physics_client != 0:
            p.disconnect(self.pb_physics_client)
            raise RuntimeError("only one PyBullet environment per process is supported")
        try:
            self.robot_id = p.loadURDF(str(urdf), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
            anchor_friction(p, self.robot_id)
            self._index()
            self._collision_pairs()
            self.reset()
        except BaseException:
            self.close()
            raise

    def _index(self):
        p = self.p
        self.joint_name_to_index, self.link_names, self.joint_indices = {}, {}, []
        lows, highs, self.joint_names = [], [], []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            self.joint_name_to_index[info[1].decode()] = j
            self.link_names[info[12].decode()] = j
            if info[2] in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE):
                self.joint_indices.append(j)
                self.joint_names.append(info[1].decode())
                lo, hi = info[8:10]
                lows.append(lo if lo <= hi else -np.pi)
                highs.append(hi if lo <= hi else np.pi)
        self.name_to_dof = {n: i for i, n in enumerate(self.joint_names)}
        self.joint_lower, self.joint_upper = np.array(lows), np.array(highs)
        self.max_forces = np.full(len(lows), 200.0)
        self.position_gains = np.full(len(lows), 0.1)
        # The torso carries the whole arm and hand: at 200 N it sinks until the wrist rests on the table.
        self.max_forces[self.dof("torso_lift_joint")] = 2000.0
        for name in self.hand_joints:
            self.max_forces[self.dof(name)] = self.hand_config["pybullet_max_force"]
            self.position_gains[self.dof(name)] = self.hand_config["pybullet_position_gain"]

    def _box(self, half, position, mass=0.0, color=(0.6, 0.6, 0.6, 1)):
        p = self.p
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return p.createMultiBody(mass, c, v, basePosition=position)

    def _pick_table(self):
        """The pick table's body: a box 0.7 m deep, ``pick_table_half_width`` either side of y = 0, top at the table height. A scene
        may draw another (``envs.errand.ErrandEnv``'s disc, the round table); it is made at this point of the reset, as the box is."""
        s = self.scenario
        return self._box((0.35, s.pick_table_half_width, 0.025), (0.45, 0, s.table_height - 0.025))

    def _collision_pairs(self):
        # Same rigid component and neighboring components share designed joint geometry.
        p = self.p
        parents = {j: p.getJointInfo(self.robot_id, j)[16] for j in range(p.getNumJoints(self.robot_id))}
        components = {-1: -1}
        for j, parent in parents.items():
            components[j] = components[parent] if p.getJointInfo(self.robot_id, j)[2] == p.JOINT_FIXED else j
        neighbors = {frozenset((components[j], components[parent])) for j, parent in parents.items()}
        self.allowed_self_pairs = set()
        for a in components:
            for b in components:
                if a >= b:
                    continue
                if components[a] == components[b] or frozenset((components[a], components[b])) in neighbors:
                    self.allowed_self_pairs.add((a, b))
                    p.setCollisionFilterPair(self.robot_id, self.robot_id, a, b, 0)
        palm = self.link_index(self.hand_root_link)
        self.hand_link_indices = {palm}
        for j, parent in parents.items():
            if parent in self.hand_link_indices:
                self.hand_link_indices.add(j)
        self.support_links = {
            self.link_names[n] for n in self.link_names if n in ("base_link", "l_wheel_link", "r_wheel_link")}

    def reset(self):
        p, s = self.p, self.scenario
        self.rng = np.random.default_rng(s.seed)
        for name in ("object_id", "table_id", "target_table_id", "plane_id"):
            if hasattr(self, name):
                p.removeBody(getattr(self, name))
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(s.timestep)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)
        self.substeps, self.dt, self.step_count = s.substeps, s.timestep * s.substeps, 0
        self.gain_scale = s.timestep / s.gain_reference_dt if s.gain_reference_dt else 1.
        self.plane_id = self._box((5, 5, 0.05), (0, 0, -0.05))
        self.table_id = self._pick_table()
        tx, ty = s.target_xy
        self.target_table_id = self._box((0.25, 0.3, 0.025), (tx, ty, s.table_height - 0.025))
        self.goal_marker_id = None
        self.goal_pos = np.array([tx, ty, s.table_height + s.object_spec().support_offset_z])
        self.cube_positions = np.empty((0, 3))
        yaw = self.rng.uniform(-0.15, 0.15)
        self.object_id = s.object_spec().create_body(p, (0.45, 0, s.table_height + s.object_spec().support_offset_z + 0.001),
                                                    p.getQuaternionFromEuler((0, 0, yaw)))
        anchor_friction(p, self.object_id)
        initial = {n: 0.0 for n in self.joint_names}
        # Compact start: clear of the table, the torso and the camera's view of the object.
        initial.update(dict(zip(FETCH_ARM_JOINTS, READY_ARM, strict=True)))
        initial["root_x_axis_joint"] = -0.35
        initial["torso_lift_joint"] = 0.2
        initial.update(dict(zip(self.hand_joints, self.hand_config["open_pose"], strict=True)))
        self.set_joints(initial)
        self._hold = np.clip(self.agent_qpos, self.joint_lower, self.joint_upper)
        for _ in range(60):
            self.step(self._hold)
        self.step_count = 0
        self.initial_object_rotation = self.get_object_transform()[:3, :3]
        return self.get_object_transform()

    @property
    def agent_qpos(self):
        return np.array([v[0] for v in self.p.getJointStates(self.robot_id, self.joint_indices)])

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != self.joint_lower.shape or not np.isfinite(a).all():
            raise ValueError("action must contain one finite target per robot DoF")
        self.p.setJointMotorControlArray(self.robot_id, self.joint_indices, self.p.POSITION_CONTROL,
                                         targetPositions=np.clip(a, self.joint_lower, self.joint_upper),
                                         forces=self.max_forces, positionGains=self.position_gains * self.gain_scale)
        for _ in range(self.substeps):
            self.p.stepSimulation()
        self.step_count += 1
        if getattr(self, "_video", None) and self.step_count % 4 == 0:
            self._capture_frame()

    def start_recording(self, path, *, eye=(2.0, .5, 1.5), target=(0., .5, .75)):
        """Stream actual simulator frames, including unsuccessful executions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._camera = (list(eye), list(target))
        self._video = imageio.get_writer(path, mode="I", duration=1000 * 4 * self.substeps * self.scenario.timestep, loop=0)
        self._capture_frame()

    def stop_recording(self):
        if getattr(self, "_video", None):
            self._video.close()
            self._video = None

    def _capture_frame(self):
        p = self.p
        view = p.computeViewMatrix(*self._camera, [0., 0., 1.])
        projection = p.computeProjectionMatrixFOV(55, 640 / 480, .05, 10)
        rgba = p.getCameraImage(640, 480, view, projection, renderer=p.ER_TINY_RENDERER)[2]
        self._video.append_data(np.asarray(rgba).reshape(480, 640, 4)[:, :, :3].astype(np.uint8))

    def observe(self, n_points=1024, intr: CameraIntrinsics | None = None, *, eye=None):
        # Fixed calibrated camera: one render, known target segmentation and object pose.
        p, intr = self.p, intr or CameraIntrinsics()
        eye = np.array(self.scenario.camera_eye if eye is None else eye)
        forward = np.array([0.45, 0, self.scenario.table_height]) - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0., 0., 1.])
        view = p.computeViewMatrix(eye, eye + forward, up)
        proj = p.computeProjectionMatrixFOV(intr.fov_deg, intr.width / intr.height, intr.near, intr.far)
        _, _, _, depth, seg = p.getCameraImage(intr.width, intr.height, view, proj,
                                              renderer=p.ER_TINY_RENDERER,
                                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        depth = np.asarray(depth).reshape(intr.height, intr.width)
        seg = np.asarray(seg).reshape(depth.shape)
        mask = (seg >= 0) & ((seg & ((1 << 24) - 1)) == self.object_id) & (depth < 1)
        if mask.sum() < 30:
            raise ValueError("target not visible in the single camera observation")
        metric_depth = depth_buffer_to_metric(depth, intr.near, intr.far)
        if self.scenario.camera_noise_std:
            metric_depth = np.maximum(intr.near, metric_depth + self.rng.normal(0, self.scenario.camera_noise_std, metric_depth.shape))
        world = back_project(metric_depth, mask, intr, eye, forward, up)
        cloud = prepare_cloud(world, n_points, rng=self.rng, T_world_object=self.get_object_transform(),
                              noise_std=0.)
        return {"point_cloud": cloud, "depth_buffer": depth, "mask": mask,
                "camera_eye": eye, "T_world_object": self.get_object_transform()}

    def evaluate(self):
        p = self.p
        pos = self.object_position()
        distance = float(np.linalg.norm(pos - self.goal_pos))
        velocity = np.linalg.norm(p.getBaseVelocity(self.object_id)[0])
        contacts = p.getContactPoints(self.object_id, self.target_table_id)
        hand_contacts = p.getContactPoints(self.object_id, self.robot_id)
        success = distance <= self.scenario.target_tolerance and velocity < 0.02 and bool(contacts) and not hand_contacts
        # Orientation relative to the object's initial pose: tilt ignores rotation about the vertical.
        change = self.get_object_transform()[:3, :3] @ self.initial_object_rotation.T
        tilt = float(np.degrees(np.arccos(np.clip(change[2, 2], -1, 1))))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(change) - 1) / 2, -1, 1))))
        limit = self.scenario.max_placement_tilt_deg
        upright = limit is None or tilt <= limit
        return {"success": bool(success and upright), "dist_to_goal": distance, "released": not bool(hand_contacts),
                "supported": bool(contacts), "speed": float(velocity), "tilt_deg": tilt,
                "orientation_error_deg": angle, "upright": bool(upright)}

    def joint_index(self, name):
        return self.joint_name_to_index[name]

    def link_index(self, name):
        return self.link_names[name]

    def dof(self, name):
        return self.name_to_dof[name]

    def dofs(self, names):
        return [self.dof(n) for n in names]

    def hold_action(self):
        return self.agent_qpos.copy()

    def action_with(self, base=None, **overrides):
        a = self.hold_action() if base is None else np.array(base, copy=True)
        for name, value in overrides.items():
            a[self.dof(name)] = value
        return a

    def action_with_hand(self, hand_q, base=None):
        a = self.hold_action() if base is None else np.array(base, copy=True)
        a[self.dofs(self.hand_joints)] = hand_q
        return a

    def set_joints(self, values):
        for name, value in values.items():
            self.p.resetJointState(self.robot_id, self.joint_index(name), float(value))

    def get_joints(self, names):
        return self.agent_qpos[self.dofs(names)]

    def joint_limits(self, names):
        idx = self.dofs(names)
        return self.joint_lower[idx].copy(), self.joint_upper[idx].copy()

    def hand_joint_values(self):
        return self.get_joints(self.hand_joints)

    def hand_joint_limits(self):
        return self.joint_limits(self.hand_joints)

    def get_link_transform(self, name):
        state = self.p.getLinkState(self.robot_id, self.link_index(name), computeForwardKinematics=True)
        return transform_from_pos_quat_xyzw(state[4], state[5])

    def get_object_transform(self):
        return transform_from_pos_quat_xyzw(*self.p.getBasePositionAndOrientation(self.object_id))

    def object_position(self):
        return np.array(self.p.getBasePositionAndOrientation(self.object_id)[0])

    def get_base_pose(self):
        return self.get_joints(FETCH_BASE_JOINTS)

    def table_top_z(self):
        return self.scenario.table_height

    def finger_contacts(self, body_id):
        """Normal contact force (N) between each finger and a body, summed over the finger's links."""
        contacts = self.p.getContactPoints(self.robot_id, body_id)
        # A PyBullet link shares the index of the joint that drives it.
        return {finger: sum(c[9] for c in contacts if c[3] in {self.joint_index("leap_" + j) for j in names})
                for finger, names in self.hand_config["finger_groups"].items()}

    def summary(self):
        return {"simulator": "pybullet", "n_dofs": len(self.joint_indices), "hand_dofs": len(self.hand_joints),
                "scenario": asdict(self.scenario), "object_position": self.object_position().tolist(),
                "base_pose": self.get_base_pose().tolist(), "placement": self.evaluate()}

    def close(self):
        self.stop_recording()
        if self.p.isConnected(self.pb_physics_client):
            self.p.disconnect(self.pb_physics_client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
