from __future__ import annotations

import numpy as np


def xyzw_to_wxyz(q):
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def look_at_quat(eye, target):
    """Compute wxyz quaternion for a camera at *eye* looking at *target*."""
    eye_a, tgt_a = np.asarray(eye, np.float64), np.asarray(target, np.float64)

    fwd = tgt_a - eye_a
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-12

    r_mat = np.column_stack([fwd, np.cross(up, fwd), up])
    tr = np.trace(r_mat)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = (
            0.25 * s,
            (r_mat[2, 1] - r_mat[1, 2]) / s,
            (r_mat[0, 2] - r_mat[2, 0]) / s,
            (r_mat[1, 0] - r_mat[0, 1]) / s,
        )
    elif r_mat[0, 0] > r_mat[1, 1] and r_mat[0, 0] > r_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r_mat[0, 0] - r_mat[1, 1] - r_mat[2, 2])
        w, x, y, z = (
            (r_mat[2, 1] - r_mat[1, 2]) / s,
            0.25 * s,
            (r_mat[0, 1] + r_mat[1, 0]) / s,
            (r_mat[0, 2] + r_mat[2, 0]) / s,
        )
    elif r_mat[1, 1] > r_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r_mat[1, 1] - r_mat[0, 0] - r_mat[2, 2])
        w, x, y, z = (
            (r_mat[0, 2] - r_mat[2, 0]) / s,
            (r_mat[0, 1] + r_mat[1, 0]) / s,
            0.25 * s,
            (r_mat[1, 2] + r_mat[2, 1]) / s,
        )
    else:
        s = 2.0 * np.sqrt(1.0 + r_mat[2, 2] - r_mat[0, 0] - r_mat[1, 1])
        w, x, y, z = (
            (r_mat[1, 0] - r_mat[0, 1]) / s,
            (r_mat[0, 2] + r_mat[2, 0]) / s,
            (r_mat[1, 2] + r_mat[2, 1]) / s,
            0.25 * s,
        )

    n = np.sqrt(w * w + x * x + y * y + z * z)
    return (float(w / n), float(x / n), float(y / n), float(z / n))


def resolve_mesh_prim(root_path, *, prim_utils, sim_utils):
    """Find the first Mesh prim under *root_path* for SDF queries."""
    from pxr import UsdGeom

    root_prim = prim_utils.get_prim_at_path(str(root_path))
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Invalid target mesh prim: {root_path}")

    query_path = str(root_path)
    if not root_prim.IsA(UsdGeom.Mesh):
        children = sim_utils.get_all_matching_child_prims(
            query_path,
            predicate=lambda p: p.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        )
        if children:
            def rank_mesh(prim):
                path = prim.GetPath().pathString.lower()
                if "/collisions/" in path or path.endswith("/collisions"):
                    return 0
                if "collision" in path:
                    return 1
                if "/visuals/" in path or path.endswith("/visuals"):
                    return 3
                return 2

            query_path = min(children, key=rank_mesh).GetPath().pathString

    query_prim = prim_utils.get_prim_at_path(query_path)
    if not query_prim or not query_prim.IsValid():
        raise RuntimeError(f"No Mesh prim found under: {root_path}")
    return query_path, query_prim


def to_numpy(x, *, dtype=None, shape=None):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    if shape is not None:
        x = x.reshape(shape)
    return x


def to_numpy_1d(x, expected):
    x = to_numpy(x, shape=(-1,))
    if x.size != expected:
        raise ValueError(f"Expected {expected} values, got {x.size}")
    return x


def obj_pose_numpy(obj) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    if obj is None:
        return pose
    root_pos = getattr(obj.data, "root_pos_w", None)
    root_quat = getattr(obj.data, "root_quat_w", None)
    if root_pos is None or root_quat is None:
        return pose
    pose[:3] = to_numpy(root_pos[0], dtype=np.float32, shape=(3,))
    pose[3:] = to_numpy(root_quat[0], dtype=np.float32, shape=(4,))
    return pose


def rigid_prim_world_pose(rp):
    """Get (pos, quat) as numpy from a RigidPrim."""
    fn = getattr(rp, "get_world_pose", None) or getattr(rp, "get_world_poses", None)
    if fn is None:
        raise AttributeError("RigidPrim has neither get_world_pose nor get_world_poses")
    pos, quat = fn()
    return to_numpy_1d(pos, 3), to_numpy_1d(quat, 4)
