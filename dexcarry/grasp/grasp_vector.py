"""The 25-number grasp vector (wrist position, 6D rotation, joint angles) and rigid-transform helpers.
PyBullet quaternions are xyzw, MuJoCo's wxyz.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

GRASP_DIM = 25
MAX_JOINTS = 16


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # gimbal lock is fine: any decomposition works
        return Rotation.from_matrix(np.asarray(R, dtype=float)).as_euler("xyz")


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened as [c0x, c0y, c0z, c1x, c1y, c1z]."""
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[:, 0], R[:, 1]])


def matrix_from_rot6d(v: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: two (possibly non-orthonormal) columns to a rotation matrix."""
    v = np.asarray(v, dtype=float).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rotation_error_vector(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) that takes R_current to R_target, expressed in the world frame."""
    R_delta = np.asarray(R_target) @ np.asarray(R_current).T
    return Rotation.from_matrix(R_delta).as_rotvec()


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def transform_from_pos_quat_xyzw(pos, quat_xyzw) -> np.ndarray:
    return make_transform(quat_xyzw_to_matrix(quat_xyzw), pos)


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points of shape (N, 3) or (3,)."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = pts.reshape(-1, 3)
    out = pts @ T[:3, :3].T + T[:3, 3]
    return out[0] if single else out


def pack_grasp(t: np.ndarray, R: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Build the 25-vector. Joints shorter than 16 are zero-padded on the right."""
    joints = np.asarray(joints, dtype=float).reshape(-1)
    if joints.size > MAX_JOINTS:
        raise ValueError(f"at most {MAX_JOINTS} joints supported, got {joints.size}")
    padded = np.zeros(MAX_JOINTS)
    padded[: joints.size] = joints
    return np.concatenate([np.asarray(t, dtype=float).reshape(3), rot6d_from_matrix(R), padded])


def unpack_grasp(vec: np.ndarray, n_joints: int = MAX_JOINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, R, joints[:n_joints])."""
    vec = np.asarray(vec, dtype=float).reshape(GRASP_DIM)
    return vec[:3].copy(), matrix_from_rot6d(vec[3:9]), vec[9 : 9 + n_joints].copy()


def grasp_to_transform(vec: np.ndarray) -> np.ndarray:
    t, R, _ = unpack_grasp(vec)
    return make_transform(R, t)


def grasp_in_world(vec: np.ndarray, T_obj_world: np.ndarray) -> np.ndarray:
    """Wrist transform in the world frame for a grasp expressed in the object frame."""
    return T_obj_world @ grasp_to_transform(vec)

