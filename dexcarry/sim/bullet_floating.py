"""Floating-hand close, lift and hold tests in PyBullet."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from ..config import Paths, load_hand_config
from ..data.catalog import ObjectSpec
from ..data.lineage import CandidateRecord, TrialRecord
from ..envs.mjcf_to_urdf import convert_mjcf_hand
from ..envs.urdf_tools import write_standalone_hand_urdf
from ..grasp.force_closure import force_closure
from ..grasp.grasp_vector import (
    invert_transform,
    make_transform,
    matrix_to_quat_xyzw,
    pack_grasp,
    transform_from_pos_quat_xyzw,
    unpack_grasp,
)
from .closing import RAMP_SECONDS, SQUEEZE_SECONDS, ContactStopClose
from .loads import SLIP_ROTATION_DEG, SLIP_TRANSLATION, anchor_friction, load_directions, rotation_between_deg


class BulletFloatingHarness:
    """``pinned`` maps joint names to values those joints are held at in every test, whatever the
    grasp or the closing rule asks: a finger that is busy elsewhere (the ring finger wrapping the
    first object) and must not help hold this one."""

    def __init__(self, spec: ObjectSpec, *, paths: Paths | None = None, dt=1 / 240, hand="leap",
                 pinned: dict[str, float] | None = None):

        self.p, self.spec, self.dt = p, spec, dt
        self.pinned = dict(pinned or {})
        paths = paths or Paths.load()
        self.cfg = load_hand_config(hand)
        converted = convert_mjcf_hand(paths.menagerie_dir / self.cfg["menagerie_xml"], prefix=f"{hand}_",
                                      default_mass=self.cfg["urdf_default_mass"])
        urdf_xml, tag = self._edit_hand_urdf(converted.urdf_xml)
        path = write_standalone_hand_urdf(urdf_xml, paths.urdf_dir / f"{hand}_floating{tag}.urdf")
        self._video = self._pose_log = None
        self.client = p.connect(p.DIRECT)
        if self.client != 0:
            p.disconnect(self.client)
            raise RuntimeError("one PyBullet simulation per process required")
        try:
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(dt)
            p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)
            self.hand = p.loadURDF(str(path), basePosition=[0, 0, .8], useFixedBase=False)
            anchor_friction(p, self.hand)
            names = {p.getJointInfo(self.hand, j)[1].decode(): j for j in range(p.getNumJoints(self.hand))}
            self.joints = [names[f"{hand}_" + j] for j in self.cfg["joint_order"]]
            # Links that move with each joint: its own link and any link fixed to it. The LEAP model
            # has none of the latter; the Allegro fingertips are separate bodies fixed to the last link.
            self._rigid_links = {j: [j] for j in self.joints}
            for link in range(p.getNumJoints(self.hand)):
                if p.getJointInfo(self.hand, link)[2] != p.JOINT_FIXED:
                    continue
                owner = p.getJointInfo(self.hand, link)[16]
                while owner >= 0 and p.getJointInfo(self.hand, owner)[2] == p.JOINT_FIXED:
                    owner = p.getJointInfo(self.hand, owner)[16]
                if owner in self._rigid_links:
                    self._rigid_links[owner].append(link)
            self.lower = np.array([p.getJointInfo(self.hand, j)[8] for j in self.joints])
            self.upper = np.array([p.getJointInfo(self.hand, j)[9] for j in self.joints])
            unknown = set(self.pinned) - set(self.cfg["joint_order"])
            if unknown:
                raise ValueError(f"unknown joints to pin: {sorted(unknown)}")
            self._pinned_idx = np.array([self.cfg["joint_order"].index(n) for n in self.pinned], int)
            self._pinned_q = np.clip([self.pinned[n] for n in self.pinned], self.lower[self._pinned_idx],
                                     self.upper[self._pinned_idx]) if self.pinned else np.zeros(0)
            # PyBullet's base pose is the inertial (centre-of-mass) frame, which for the LEAP palm is
            # offset and rotated from the palm link frame that grasps are expressed in.
            self._link_to_inertial = transform_from_pos_quat_xyzw(*p.getDynamicsInfo(self.hand, -1)[3:5])
            self.constraint = p.createConstraint(self.hand, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                                  [0, 0, 0], p.getBasePositionAndOrientation(self.hand)[0])
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .6, .02])
            self.table = p.createMultiBody(0, collision, basePosition=[0, 0, -.02])
        except BaseException:
            self.close()
            raise

    def _edit_hand_urdf(self, urdf_xml: str) -> tuple[str, str]:
        """Hook for a harness that needs another collision model of the same hand: the URDF text
        to load and a suffix for its file name. The default changes nothing."""
        return urdf_xml, ""

    def start_recording(self, path, *, every=8, eye=(.42, -.42, .34), target=(0., 0., .13), size=(480, 360)):
        """Save real simulator frames of every later test to a GIF, one frame per ``every`` steps."""

        self.stop_recording()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        view = self.p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
        projection = self.p.computeProjectionMatrixFOV(50, size[0] / size[1], .05, 5)
        writer = imageio.get_writer(path, mode="I", duration=1000 * every * self.dt, loop=0)
        self._video = {"writer": writer, "every": every, "view": view, "projection": projection, "size": size, "step": 0}

    def stop_recording(self):
        if self._video:
            self._video["writer"].close()
            self._video = None

    # Bodies a pose log follows, by attribute name, in this order (the two-object harness adds "second").
    LOGGED_BODIES = ("object", "second")

    def start_pose_log(self, every=4):
        """Keep a pose log of every later step, one row per ``every`` steps: the hand's base pose (PyBullet's
        centre-of-mass frame), its 16 joint angles and each object's pose. It only reads poses, so the physics is the same with it on.
        Each ``_place_palm`` (a teleport, not motion) is marked as a "placed" event at the step it happens."""
        self._pose_log = {"every": int(every), "step": 0, "seconds": 0., "rows": [], "events": []}

    def stop_pose_log(self) -> dict | None:
        """The log so far, as arrays, and logging off: ``step`` and ``seconds`` (the simulated time, summed step by step, since the time
        step may change), ``hand`` (rows x 7: xyz, quaternion xyzw), ``joints`` (rows x 16, in the hand config's order), ``bodies``
        (rows x objects x 7, NaN where an object did not exist), ``joint_names`` and ``events``."""
        log, self._pose_log = self._pose_log, None
        if log is None:
            return None
        rows = log["rows"]
        width = max((len(r[4]) for r in rows), default=0)
        bodies = np.full((len(rows), width, 7), np.nan)
        for k, row in enumerate(rows):
            if row[4]:
                bodies[k, :len(row[4])] = row[4]
        return {"step": np.array([r[0] for r in rows], np.int64), "seconds": np.array([r[1] for r in rows]),
                "hand": np.array([r[2] for r in rows]).reshape(-1, 7), "joints": np.array([r[3] for r in rows]).reshape(-1, len(self.joints)),
                "bodies": bodies, "joint_names": list(self.cfg["joint_order"]), "events": list(log["events"]), "every": log["every"]}

    def _log_pose(self):
        """One row of the pose log, read only."""
        p, log = self.p, self._pose_log
        log["step"] += 1
        log["seconds"] += self.dt
        if log["step"] % log["every"]:
            return
        position, turn = p.getBasePositionAndOrientation(self.hand)
        bodies = [[*q, *r] for q, r in (p.getBasePositionAndOrientation(getattr(self, name)) for name in self.LOGGED_BODIES
                                         if getattr(self, name, None) is not None)]
        log["rows"].append((log["step"], log["seconds"], [*position, *turn], [p.getJointState(self.hand, j)[0] for j in self.joints], bodies))

    def close(self):
        self.stop_recording()
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _palm(self):
        """World pose of the palm link frame."""
        base = transform_from_pos_quat_xyzw(*self.p.getBasePositionAndOrientation(self.hand))
        return base @ invert_transform(self._link_to_inertial)

    def _place_palm(self, T):
        base = T @ self._link_to_inertial
        self.p.resetBasePositionAndOrientation(self.hand, base[:3, 3], matrix_to_quat_xyzw(base[:3, :3]))
        self.p.resetBaseVelocity(self.hand, [0, 0, 0], [0, 0, 0])
        if self._pose_log is not None:
            self._pose_log["events"].append({"step": self._pose_log["step"], "kind": "placed"})

    def pin(self, q):
        """``q`` with the pinned joints set to their pinned values."""
        q = np.array(q, float)
        q[self._pinned_idx] = self._pinned_q
        return q

    def _control(self, q, T):
        p = self.p
        q = self.pin(q)
        T = T @ self._link_to_inertial
        p.setJointMotorControlArray(self.hand, self.joints, p.POSITION_CONTROL, targetPositions=q,
                                    forces=[self.cfg["pybullet_max_force"]] * 16,
                                    positionGains=[self.cfg["pybullet_position_gain"]] * 16)
        p.changeConstraint(self.constraint, T[:3, 3], matrix_to_quat_xyzw(T[:3, :3]), maxForce=500)
        p.stepSimulation()
        if self._pose_log is not None:
            self._log_pose()
        video = self._video
        if video:
            video["step"] += 1
            if video["step"] % video["every"] == 0:
                width, height = video["size"]
                rgba = p.getCameraImage(width, height, video["view"], video["projection"], renderer=p.ER_TINY_RENDERER)[2]
                video["writer"].append_data(np.asarray(rgba).reshape(height, width, 4)[:, :, :3].astype(np.uint8))

    def _reset_scene(self):
        """Fresh object resting on the table with the open hand parked above; returns its pose."""
        p, spec = self.p, self.spec
        if hasattr(self, "object"):
            p.removeBody(self.object)
        p.setGravity(0, 0, -9.81)
        park = make_transform(np.eye(3), [0, 0, .8])
        self._place_palm(park)
        self.object = spec.create_body(p, [0, 0, spec.support_offset_z + .002])
        anchor_friction(p, self.object)
        q_open = np.clip(self.cfg["open_pose"], self.lower, self.upper)
        for j, q in zip(self.joints, q_open, strict=True):
            p.resetJointState(self.hand, j, q)
        for _ in range(240):
            self._control(q_open, park)
        return transform_from_pos_quat_xyzw(*p.getBasePositionAndOrientation(self.object))

    def synthesize(self, T_object_palm, preshape, closed, *, steps=300, backoff=.05):
        """Close an open preshape onto the frozen object and return the 25-D grasp at contact.

        This uses the true object geometry, so it is a privileged data-collection
        source (like the public synthesis pipelines), not a deployable generator.
        Returns None when the pose starts in collision or the thumb and at least
        one finger do not both reach the object.
        """
        p = self.p
        T_obj = self._reset_scene()
        T = T_obj @ np.asarray(T_object_palm, float)
        preshape = self.pin(np.clip(preshape, self.lower, self.upper))
        closed = self.pin(np.clip(closed, self.lower, self.upper))
        p.changeDynamics(self.object, -1, mass=0)
        self._place_palm(T)
        for j, q in zip(self.joints, preshape, strict=True):
            p.resetJointState(self.hand, j, q)
        p.performCollisionDetection()
        if (any(c[8] < -.001 for c in p.getContactPoints(self.hand, self.table))
                or any(c[8] < -.003 for c in p.getContactPoints(self.hand, self.object))):
            return None
        for alpha in np.linspace(0, 1, steps):
            self._control(preshape + alpha * (closed - preshape), T)
        for _ in range(60):
            self._control(closed, T)
        touching = {c[3] for c in p.getContactPoints(self.hand, self.object)}
        fingers = {finger for finger, names in self.cfg["finger_groups"].items()
                   if touching & set(self._finger_links(names))}
        if "thumb" not in fingers or len(fingers) < 2 or p.getContactPoints(self.hand, self.table):
            return None
        joints = np.array([p.getJointState(self.hand, j)[0] for j in self.joints])
        flex = np.array([n in self.cfg["flexion_joints"] for n in self.cfg["joint_order"]])
        joints = self.pin(np.clip(joints - backoff * flex, self.lower, self.upper))
        relative = np.asarray(T_object_palm, float)
        return pack_grasp(relative[:3, 3], relative[:3, :3], joints)

    def _finger_links(self, joint_names):
        """Link indices driven by a finger's joints (a PyBullet link shares its joint's index),
        with the links fixed to them."""
        return [link for name in joint_names for link in self._rigid_links[self.joints[self.cfg["joint_order"].index(name)]]]

    def _finger_forces(self):
        """Normal contact force (N) between each finger and the object, summed over the finger's links."""
        contacts = self.p.getContactPoints(self.hand, self.object)
        return {finger: sum(c[9] for c in contacts if c[3] in self._finger_links(names))
                for finger, names in self.cfg["finger_groups"].items()}

    def _object_contacts(self):
        """Hand-object contacts in the object's centre-of-mass frame: position, inward normal, friction."""
        p = self.p
        pose = transform_from_pos_quat_xyzw(*p.getBasePositionAndOrientation(self.object))
        rows = []
        for c in p.getContactPoints(self.hand, self.object):
            if c[8] > .001:
                continue
            # PyBullet's normal points from the object toward the hand, and friction coefficients multiply.
            mu = p.getDynamicsInfo(self.hand, c[3])[1] * p.getDynamicsInfo(self.object, -1)[1]
            rows.append([*(pose[:3, :3].T @ (np.array(c[6]) - pose[:3, 3])), *(-pose[:3, :3].T @ np.array(c[7])), mu])
        return np.array(rows).reshape(-1, 7)

    def _closure(self):
        contacts = self._object_contacts()
        return {**force_closure(contacts, float(np.linalg.norm(self.spec.half_extents))),
                "contacts": np.round(contacts, 4).tolist()}

    def run(self, candidate: CandidateRecord, seed=0, *, scenario_id="floating", jitter=.001,
            provenance=None) -> TrialRecord:
        p, spec = self.p, self.spec
        if candidate.object_id != spec.object_id:
            raise ValueError("candidate geometry identity mismatch")
        T_obj = self._reset_scene()
        t, R, joints = unpack_grasp(np.asarray(candidate.grasp), 16)
        T = T_obj @ make_transform(R, t)
        T[:3, 3] += np.random.default_rng(seed).normal(0, jitter, 3)
        flex = np.array([n in self.cfg["flexion_joints"] for n in self.cfg["joint_order"]])
        pre = self.pin(np.clip(joints - .15 * flex, self.lower, self.upper))
        target = self.pin(np.clip(joints + self.cfg["close_delta"] * flex, self.lower, self.upper))
        self._place_palm(T)
        for j, q in zip(self.joints, pre, strict=True):
            p.resetJointState(self.hand, j, q)
        p.performCollisionDetection()
        table_collision = any(c[8] < -.001 for c in p.getContactPoints(self.hand, self.table))
        penetration = max([0.] + [-c[8] for c in p.getContactPoints(self.hand, self.object)])
        success, height, drift, turned = False, 0., 0., 0.
        closing = {}
        if not table_collision and penetration <= .01:
            close = ContactStopClose(self.cfg, pre, target, self.lower, self.upper)
            proposed = self._relative()
            for alpha in np.linspace(0, 1, round(RAMP_SECONDS / self.dt)):
                self._control(close.command(alpha), T)
                close.observe(self._finger_forces(), [p.getJointState(self.hand, j)[0] for j in self.joints])
            for alpha in np.linspace(0, 1, round(SQUEEZE_SECONDS / self.dt)):
                self._control(close.squeeze(alpha), T)
            target = close.squeeze(1.)
            reference = self._relative()
            # How far closing moved the object from the proposed grasp, before any load is applied.
            closing = {"closing_rotation_deg": rotation_between_deg(proposed, reference),
                       "closing_displacement": float(np.linalg.norm(reference[:3, 3] - proposed[:3, 3])),
                       "closing_fingers": sorted(close.halted), "force_closure_at_close": self._closure()}
            start_z = p.getBasePositionAndOrientation(self.object)[0][2]
            for alpha in np.linspace(0, 1, 240):
                lifted = T.copy()
                lifted[2, 3] += .14 * alpha
                self._control(target, lifted)
            for _ in range(240):
                self._control(target, lifted)
            object_pose = transform_from_pos_quat_xyzw(*p.getBasePositionAndOrientation(self.object))
            height = float(object_pose[2, 3] - start_z)
            drift = float(np.linalg.norm(self._relative()[:3, 3] - reference[:3, 3]))
            turned = rotation_between_deg(reference, self._relative())
            success = height >= .10 and drift <= SLIP_TRANSLATION and turned <= SLIP_ROTATION_DEG
            if success:
                # Force closure of the grasp as actually held in the air, with the table gone.
                closing["force_closure"] = self._closure()
        self._held = (target, lifted) if success else None
        scenario = {"object_half_extents": spec.half_extents, "object_mass": spec.mass,
                    "object_friction": spec.friction, "object_spec": asdict(spec)}
        if self.pinned:
            scenario["pinned_joints"] = self.pinned
        return TrialRecord(candidate.candidate_id, scenario_id, seed, "floating", "pybullet",
                           grasp=bool(success), first_failure=None if success else "GRASP_FAIL", scenario=scenario,
                           diagnostics={"lift_height": height, "relative_displacement": drift,
                                        "relative_rotation_deg": turned, **closing,
                                        "table_collision": table_collision, "initial_penetration": penetration},
                           provenance=provenance or {})

    def _relative(self):
        """Object pose in the palm frame."""
        return invert_transform(self._palm()) @ transform_from_pos_quat_xyzw(
            *self.p.getBasePositionAndOrientation(self.object))

    def run_load_map(self, candidate: CandidateRecord, seed=0, *, directions=None, levels=(1., 1.5, 2., 3.),
                     scenario_id="load-map", slip_tolerance=SLIP_TRANSLATION,
                     slip_rotation_deg=SLIP_ROTATION_DEG, hold_steps=120, provenance=None) -> TrialRecord:
        """Tolerated load (g) along each palm-frame direction after a successful lift.

        The hand stays fixed and effective gravity is rotated from "down" to the
        direction at 1 g, then raised through ``levels``. A direction's tolerance is
        the last level held without the object sliding more than ``slip_tolerance``
        or pivoting more than ``slip_rotation_deg`` in the palm frame; 0 means it was lost before or at 1 g, including on the
        way there. Every direction restarts from the same saved post-lift state.
        """
        levels = [float(v) for v in levels]
        if not levels or levels[0] != 1. or any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
            raise ValueError("levels must start at 1 g and increase")
        directions = load_directions() if directions is None else np.asarray(directions, float)
        if directions.ndim != 2 or directions.shape[1] != 3 or not np.allclose(np.linalg.norm(directions, axis=1), 1):
            raise ValueError("unit palm-frame directions required")
        p = self.p
        lift = self.run(candidate, seed, provenance=provenance)
        scenario = {**lift.scenario, "directions": directions.tolist(), "levels": levels,
                    "slip_tolerance": slip_tolerance, "slip_rotation_deg": slip_rotation_deg,
                    "hold_steps": hold_steps}
        if not lift.grasp:
            return TrialRecord(candidate.candidate_id, scenario_id, seed, "load_map", "pybullet", grasp=False,
                               first_failure="GRASP_FAIL", scenario=scenario, diagnostics=lift.diagnostics,
                               provenance=provenance or {})
        target, pose = self._held
        rotation = self._palm()[:3, :3]
        reference = self._relative()
        down = np.array([0., 0., -1.])
        state = p.saveState()
        tolerance, worst = [], []
        # Control: the direction the grasp already held during the lift, run through the same
        # schedule. A loss here is creep over the longer hold, not an effect of load direction.
        lift_direction = rotation.T @ down
        try:
            for direction in [lift_direction, *directions]:
                p.restoreState(state)
                goal = rotation @ direction
                angle = float(np.arccos(np.clip(down @ goal, -1, 1)))
                axis = np.cross(down, goal)
                axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-8 else np.array([1., 0., 0.])
                held, drift = 0., 0.

                def hold(gravity, steps):
                    nonlocal drift
                    for _ in range(steps):
                        p.setGravity(*(9.81 * gravity))
                        self._control(target, pose)
                        relative = self._relative()
                        drift = max(drift, float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])))
                        if drift > slip_tolerance or rotation_between_deg(reference, relative) > slip_rotation_deg:
                            return False
                    return True

                ok = True
                for a in np.linspace(0, angle, max(2, int(240 * angle / np.pi))):
                    # Rodrigues rotation of "down" toward the target direction at constant 1 g.
                    g = down * np.cos(a) + np.cross(axis, down) * np.sin(a) + axis * (axis @ down) * (1 - np.cos(a))
                    ok = hold(g, 1)
                    if not ok:
                        break
                previous = 1.
                for level in levels if ok else []:
                    ok = all(hold(goal * (previous + f * (level - previous)), 1) for f in np.linspace(0, 1, 60)) and \
                        hold(goal * level, hold_steps)
                    if not ok:
                        break
                    held, previous = level, level
                tolerance.append(held)
                worst.append(drift)
        finally:
            p.setGravity(0, 0, -9.81)
            p.removeState(state)
        return TrialRecord(candidate.candidate_id, scenario_id, seed, "load_map", "pybullet", grasp=True,
                           scenario=scenario,
                           diagnostics={**lift.diagnostics, "tolerance_g": tolerance[1:], "max_drift": worst[1:],
                                        "control_tolerance_g": tolerance[0],
                                        "lift_direction_palm": lift_direction.tolist()},
                           provenance=provenance or {})
