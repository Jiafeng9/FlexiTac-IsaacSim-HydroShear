"""Compatibility entry point for ALOHA tactile configuration."""

from aloha.camera import AlohaCameraCfg, AlohaCameraOutput
from aloha.cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER
from aloha.objects import AlohaObjectsCfg, AlohaObjectsOutput
from aloha.robot import AlohaRobotCfg, AlohaRobotOutput
from aloha.scene import AlohaSceneOutput, AlohaSimCfg
from aloha.tactile import (
    AlohaTactileCfg,
    AlohaTactileOutput,
    HydroShearTactileBackendCfg,
    TaxelShearTactileBackendCfg,
    TrackInfo,
    WarpSdfTactileBackendCfg,
)

__all__ = [
    "AlohaTactileEnvCfg",
    "AlohaRobotCfg",
    "AlohaRobotOutput",
    "AlohaObjectsCfg",
    "AlohaObjectsOutput",
    "AlohaTactileCfg",
    "AlohaTactileOutput",
    "WarpSdfTactileBackendCfg",
    "TaxelShearTactileBackendCfg",
    "HydroShearTactileBackendCfg",
    "AlohaCameraCfg",
    "AlohaCameraOutput",
    "AlohaSimCfg",
    "AlohaSceneOutput",
    "DATASET_JOINT_ORDER",
    "TrackInfo",
]
