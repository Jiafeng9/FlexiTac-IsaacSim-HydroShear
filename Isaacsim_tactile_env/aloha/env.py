"""Standalone gymnasium.Env for ALOHA bimanual robot with WarpSdf tactile sensors."""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium
import numpy as np

from .camera import AlohaCamera
from .cfg import AlohaTactileEnvCfg
from .helpers.structure import parse_elastomer_origins
from .objects import PlugSocketObjects
from .observation import AlohaObservationBuilder
from .robot import AlohaRobot
from .scene import AlohaScene
from .tactile import AlohaTactileSetup


class AlohaTactileEnv(gymnasium.Env):
    """ALOHA bimanual tactile environment.

    Observations:
        tactile:     (4, num_rows, num_cols) force grids
        joint_pos:   (16,) joint positions in dataset order
        joint_vel:   (16,) joint velocities in dataset order
        plug_pose:   (7,) pos + quat_wxyz (zeros if disabled)
        socket_pose: (7,) pos + quat_wxyz (zeros if disabled)
        rgb:         (H, W, 3) uint8 (if enable_camera)

    Actions:
        Box(16,) raw joint position targets.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: AlohaTactileEnvCfg, simulation_app=None):
        super().__init__()
        self._cfg = cfg
        self._simulation_app = simulation_app
        self._step_count = 0
        self._base_dir = str(Path(__file__).resolve().parent.parent)

        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api.simulation_context import SimulationContext
        from isaacsim.core.utils.extensions import enable_extension

        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.assets.articulation import ArticulationCfg
        from isaaclab.assets.rigid_object import RigidObjectCfg
        from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg
        from isaaclab.sim.converters import UrdfConverterCfg
        from isaaclab.sim.schemas import activate_contact_sensors

        self._prim_utils = prim_utils
        self._sim_utils = sim_utils
        self._math_utils = math_utils

        enable_extension("isaacsim.asset.importer.urdf")

        self._scene = AlohaScene(cfg, SimulationContext)
        self._sim = self._scene.sim
        self._device = self._scene.device
        self._scene.spawn_basic_world(sim_utils)

        self._camera_manager = AlohaCamera(cfg, sim_utils, self._base_dir)
        self._camera = self._camera_manager.camera
        self._render_output_dir = self._camera_manager.render_output_dir

        urdf_path = os.path.expanduser(cfg.robot.urdf_path)
        self._urdf_origins = parse_elastomer_origins(urdf_path)

        self._objects = PlugSocketObjects(cfg, sim_utils, RigidObject, RigidObjectCfg, self._base_dir)
        self._plug_obj = self._objects.plug_obj
        self._socket_obj = self._objects.socket_obj

        self._robot_manager = AlohaRobot(
            cfg,
            urdf_path,
            sim_utils,
            Articulation,
            ArticulationCfg,
            ImplicitActuatorCfg,
            UrdfConverterCfg,
            self._base_dir,
        )
        self._robot = self._robot_manager.asset
        activate_contact_sensors(cfg.robot.prim_path, threshold=0.0)

        from pxr import PhysxSchema, UsdPhysics

        self._UsdPhysics = UsdPhysics
        self._tactile_setup = AlohaTactileSetup(
            cfg,
            sim_utils,
            prim_utils,
            math_utils,
            self._robot,
            self._objects,
            self._urdf_origins,
            UsdPhysics,
            PhysxSchema,
            WarpSdfTactileSensor,
            WarpSdfTactileSensorCfg,
            self._device,
        )
        self._selected_links = self._tactile_setup.selected_links
        self._tactile_sensors = self._tactile_setup.sensors
        self._sensor_slot_order = self._tactile_setup.sensor_slot_order
        self._per_sensor_target_query_paths = self._tactile_setup.target_query_paths

        self._post_spawn_init()

        self._observation_builder = AlohaObservationBuilder(
            cfg,
            self._robot_manager,
            self._objects,
            self._tactile_setup,
            self._camera_manager,
        )
        self.observation_space, self.action_space = self._observation_builder.build_spaces()

    def step(self, action: np.ndarray):
        self._robot_manager.apply_action(action, self._device)

        render = not self._cfg.sim.headless or self._camera is not None
        self._sim.step(render=render)

        dt = self._cfg.sim.physics_dt
        self._robot_manager.update(dt)
        self._objects.update(dt)
        self._tactile_setup.update(dt)
        self._camera_manager.update(dt)

        obs = self._get_obs()

        if self._render_output_dir and self._camera:
            self._save_render(obs.get("rgb"), self._step_count)

        self._step_count += 1
        return obs, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._robot_manager.reset()
        self._objects.reset()

        render = not self._cfg.sim.headless or self._camera is not None
        self._sim.step(render=render)
        dt = self._cfg.sim.physics_dt
        self._robot_manager.update(dt)
        self._objects.update(dt)

        self._tactile_setup.reset()
        self._camera_manager.reset()
        self._camera_manager.update(dt)

        self._step_count = 0
        return self._get_obs(), {}

    def close(self):
        self._tactile_setup.close()

    def _post_spawn_init(self) -> None:
        self._sim.reset()
        dt = self._cfg.sim.physics_dt

        self._robot_manager.update(dt)
        self._objects.update(dt)

        self._stage = self._sim_utils.get_current_stage()
        self._tactile_setup.initialize_after_sim_reset(self._stage)
        self._per_sensor_target_prims = self._tactile_setup.per_sensor_target_prims
        self._dynamic_track_map = self._tactile_setup.dynamic_track_map

        self._dataset_joint_ids = self._robot_manager.resolve_dataset_joint_ids()

        print(
            f"[INFO] {len(self._dataset_joint_ids)} joints mapped, "
            f"{len(self._tactile_sensors)} tactile sensors",
            flush=True,
        )
        if self._camera:
            print(f"[INFO] Camera: {self._cfg.camera.width}x{self._cfg.camera.height}", flush=True)

    def _get_obs(self) -> dict:
        return self._observation_builder.build()

    def _update_target_poses(self):
        self._tactile_setup.update_target_poses()

    def _save_render(self, rgb: np.ndarray | None, step: int):
        self._camera_manager.save_render(rgb, step)
