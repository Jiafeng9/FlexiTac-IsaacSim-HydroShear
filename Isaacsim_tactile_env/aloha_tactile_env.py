"""Compatibility entry point for the ALOHA tactile environment."""

from aloha.cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER
from aloha.env import AlohaTactileEnv
from aloha.helpers.spatial import obj_pose_numpy, resolve_mesh_prim, rigid_prim_world_pose, to_numpy, to_numpy_1d
from aloha.tactile import TrackInfo

_resolve_mesh_prim = resolve_mesh_prim
_to_numpy_1d = to_numpy_1d
_obj_pose_numpy = obj_pose_numpy
_rigid_prim_world_pose = rigid_prim_world_pose

__all__ = [
    "AlohaTactileEnv",
    "AlohaTactileEnvCfg",
    "DATASET_JOINT_ORDER",
    "TrackInfo",
    "_resolve_mesh_prim",
    "_to_numpy_1d",
    "to_numpy",
    "_obj_pose_numpy",
    "_rigid_prim_world_pose",
]
