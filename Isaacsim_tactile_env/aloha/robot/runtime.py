from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..helpers.structure import resolve_joint_ids


@dataclass
class AlohaRobotCfg:
    urdf_path: str = Path(__file__).resolve().parent.parent.parent / "assets" / "aloha_tactile.urdf"
    prim_path: str = "/World/Robot"
    fix_base: bool = False
    merge_fixed_joints: bool = False
    drive_stiffness: float = 400.0
    drive_damping: float = 40.0
    force_urdf_conversion: bool = True
    usd_output_dir: str | None = None


@dataclass
class AlohaRobotOutput:
    asset: object
    dataset_joint_ids: tuple[int, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray


class AlohaRobot:
    """ALOHA articulation spawning, action application, and joint readout."""

    def __init__(
        self,
        cfg,
        urdf_path: str,
        sim_utils,
        Articulation,
        ArticulationCfg,
        ImplicitActuatorCfg,
        UrdfConverterCfg,
        base_dir: str,
    ):
        self.cfg = cfg
        self.robot_cfg = cfg.robot
        self.dataset_joint_ids: list[int] = []
        self.asset = self._spawn(
            self.robot_cfg,
            urdf_path,
            sim_utils,
            Articulation,
            ArticulationCfg,
            ImplicitActuatorCfg,
            UrdfConverterCfg,
            base_dir,
        )

    def _spawn(
        self,
        cfg,
        urdf_path: str,
        sim_utils,
        Articulation,
        ArticulationCfg,
        ImplicitActuatorCfg,
        UrdfConverterCfg,
        base_dir: str,
    ):
        out_dir = cfg.usd_output_dir or os.path.join(base_dir, "output", "aloha_urdf")
        os.makedirs(out_dir, exist_ok=True)

        return Articulation(
            ArticulationCfg(
                prim_path=cfg.prim_path,
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf_path,
                    fix_base=cfg.fix_base,
                    merge_fixed_joints=cfg.merge_fixed_joints,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=cfg.drive_stiffness,
                            damping=cfg.drive_damping,
                        )
                    ),
                    usd_dir=out_dir,
                    force_usd_conversion=cfg.force_urdf_conversion,
                    activate_contact_sensors=True,
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                actuators={
                    "all": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=cfg.drive_stiffness,
                        damping=cfg.drive_damping,
                    )
                },
            )
        )

    def resolve_dataset_joint_ids(self) -> list[int]:
        from ..cfg import DATASET_JOINT_ORDER

        self.dataset_joint_ids = resolve_joint_ids(self.asset, DATASET_JOINT_ORDER)
        return self.dataset_joint_ids

    def apply_action(self, action: np.ndarray, device: str):
        import torch

        action = np.asarray(action, dtype=np.float32).reshape(16)
        action_t = torch.tensor(action, dtype=torch.float32, device=device)
        self.asset.set_joint_position_target(action_t, joint_ids=self.dataset_joint_ids)
        self.asset.write_data_to_sim()

    def reset(self):
        self.asset.write_joint_state_to_sim(
            self.asset.data.default_joint_pos.clone(),
            self.asset.data.default_joint_vel.clone(),
        )
        self.asset.reset()

    def update(self, dt: float):
        self.asset.update(dt)

    def joint_observation(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.dataset_joint_ids
        joint_pos = self.asset.data.joint_pos[0, ids].detach().cpu().numpy().astype(np.float32)
        joint_vel = self.asset.data.joint_vel[0, ids].detach().cpu().numpy().astype(np.float32)
        return joint_pos, joint_vel

    def output(self) -> AlohaRobotOutput:
        joint_pos, joint_vel = self.joint_observation()
        return AlohaRobotOutput(
            asset=self.asset,
            dataset_joint_ids=tuple(self.dataset_joint_ids),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
