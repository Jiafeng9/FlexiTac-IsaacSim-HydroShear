from __future__ import annotations

from dataclasses import dataclass, field

DATASET_JOINT_ORDER = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/left_finger",
    "left/right_finger",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/left_finger",
    "right/right_finger",
]

from .camera import AlohaCameraCfg
from .objects import AlohaObjectsCfg
from .robot import AlohaRobotCfg
from .scene import AlohaSimCfg
from .tactile import AlohaTactileCfg


@dataclass
class AlohaTactileEnvCfg:
    """Top-level ALOHA tactile environment configuration."""

    robot: AlohaRobotCfg = field(default_factory=AlohaRobotCfg)
    objects: AlohaObjectsCfg = field(default_factory=AlohaObjectsCfg)
    tactile: AlohaTactileCfg = field(default_factory=AlohaTactileCfg)
    camera: AlohaCameraCfg = field(default_factory=AlohaCameraCfg)
    sim: AlohaSimCfg = field(default_factory=AlohaSimCfg)
