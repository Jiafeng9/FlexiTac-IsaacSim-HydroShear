"""ALOHA tactile environment package."""

from .camera import AlohaCameraCfg, AlohaCameraOutput
from .cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER
from .objects import AlohaObjectsCfg, AlohaObjectsOutput
from .robot import AlohaRobotCfg, AlohaRobotOutput
from .scene import AlohaSceneOutput, AlohaSimCfg
from .tactile import (
    AlohaTactileCfg,
    AlohaTactileOutput,
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
    TrackInfo,
)


def __getattr__(name: str):
    if name == "AlohaTactileEnv":
        from .env import AlohaTactileEnv

        return AlohaTactileEnv
    raise AttributeError(name)

__all__ = [
    "AlohaTactileEnv",
    "AlohaTactileEnvCfg",
    "AlohaRobotCfg",
    "AlohaRobotOutput",
    "AlohaObjectsCfg",
    "AlohaObjectsOutput",
    "AlohaCameraCfg",
    "AlohaCameraOutput",
    "AlohaSimCfg",
    "AlohaSceneOutput",
    "AlohaTactileCfg",
    "AlohaTactileOutput",
    "HydroShearTactileBackend",
    "HydroShearTactileBackendCfg",
    "DATASET_JOINT_ORDER",
    "TrackInfo",
]
