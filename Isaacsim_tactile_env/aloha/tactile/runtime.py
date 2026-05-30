from __future__ import annotations

import numpy as np

from .backend import (
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
    TaxelShearTactileBackend,
    TaxelShearTactileBackendCfg,
    WarpSdfTactileBackend,
    WarpSdfTactileBackendCfg,
)
from .binding import AlohaTactileBinding
from .output import AlohaTactileOutput
from .patch import AlohaPatchTransform
from .target_tracking import AlohaTargetTracker


class AlohaTactileSetup:
    """ALOHA adapter for selectable tactile backends."""

    def __init__(
        self,
        cfg,
        sim_utils,
        prim_utils,
        math_utils,
        robot_asset,
        objects,
        urdf_origins: dict,
        UsdPhysics,
        PhysxSchema,
        WarpSdfTactileSensor,
        WarpSdfTactileSensorCfg,
        device: str,
    ):
        self.cfg = cfg
        self.sim_utils = sim_utils
        self.robot_asset = robot_asset
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self._device = device

        self.binding = AlohaTactileBinding(cfg, sim_utils, prim_utils, objects, UsdPhysics, PhysxSchema)
        self.selected_links, self.target_root_paths, self.target_query_paths = self.binding.build()

        self.patch_transform = AlohaPatchTransform(self.tactile_cfg, urdf_origins, math_utils)
        self.backend = self._create_backend(
            WarpSdfTactileSensor,
            WarpSdfTactileSensorCfg,
        )
        self.sensors, self.sensor_slot_order = self.backend.create_sensors(
            self.selected_links,
            self.target_query_paths,
        )
        self.hydro_normal_backend = None
        self.hydro_normal_sensors = []
        self.hydro_normal_sensor_slot_order = []
        if bool(getattr(self.tactile_cfg, "enable_hydro_normal_observation", False)):
            self.hydro_normal_backend = HydroShearTactileBackend(
                self.cfg,
                self.patch_transform,
                self.robot_asset,
                self._device,
                backend_cfg=self.tactile_cfg.hydro_normal_backend,
                output_key=self.tactile_cfg.hydro_normal_output_key,
            )
            self.hydro_normal_sensors, self.hydro_normal_sensor_slot_order = (
                self.hydro_normal_backend.create_sensors(self.selected_links, self.target_query_paths)
            )

        self.target_tracker = AlohaTargetTracker(self.target_query_paths, sim_utils, math_utils, UsdPhysics, device)
        self.per_sensor_target_prims = []
        self.dynamic_track_map = {}

    def _create_backend(self, WarpSdfTactileSensor, WarpSdfTactileSensorCfg):
        backend_cfg = self.tactile_cfg.backend
        if isinstance(backend_cfg, TaxelShearTactileBackendCfg):
            return TaxelShearTactileBackend(
                self.cfg,
                self.patch_transform,
                WarpSdfTactileSensor,
                WarpSdfTactileSensorCfg,
                self.robot_asset,
                self._device,
            )
        if isinstance(backend_cfg, WarpSdfTactileBackendCfg):
            return WarpSdfTactileBackend(self.cfg, self.patch_transform, WarpSdfTactileSensor, WarpSdfTactileSensorCfg)
        if isinstance(backend_cfg, HydroShearTactileBackendCfg):
            return HydroShearTactileBackend(self.cfg, self.patch_transform, self.robot_asset, self._device)
        raise TypeError(f"Unsupported tactile backend cfg: {type(backend_cfg).__name__}")

    def initialize_after_sim_reset(self, stage):
        self.target_tracker.initialize(stage)
        self.per_sensor_target_prims = self.target_tracker.per_sensor_target_prims
        self.dynamic_track_map = self.target_tracker.dynamic_track_map
        self.backend.initialize_after_sim_reset(self.sensors, stage, self.target_tracker)
        if self.hydro_normal_backend is not None:
            self.hydro_normal_backend.initialize_after_sim_reset(
                self.hydro_normal_sensors,
                stage,
                self.target_tracker,
            )

    def update_target_poses(self):
        if isinstance(self.backend, WarpSdfTactileBackend):
            self.target_tracker.update_target_poses(self.sensors)

    def update(self, dt: float):
        self.backend.update(dt, self.sensors, self.target_tracker)
        if self.hydro_normal_backend is not None:
            self.hydro_normal_backend.update(dt, self.hydro_normal_sensors, self.target_tracker)

    def reset(self):
        self.backend.reset(self.sensors)
        if self.hydro_normal_backend is not None:
            self.hydro_normal_backend.reset(self.hydro_normal_sensors)

    def close(self):
        self.backend.close(self.sensors)
        if self.hydro_normal_backend is not None:
            self.hydro_normal_backend.close(self.hydro_normal_sensors)

    def normal_force_grid(self) -> np.ndarray:
        return self.backend.observations(self.sensors, self.sensor_slot_order)[self.tactile_cfg.output_key]

    def output(self) -> AlohaTactileOutput:
        observations = self.backend.observations(self.sensors, self.sensor_slot_order)
        if self.hydro_normal_backend is not None:
            observations.update(
                self.hydro_normal_backend.observations(
                    self.hydro_normal_sensors,
                    self.hydro_normal_sensor_slot_order,
                )
            )
        return AlohaTactileOutput(
            observations=observations,
            selected_links=tuple(self.selected_links),
            target_query_paths=tuple(self.target_query_paths),
            sensor_slot_order=tuple(self.sensor_slot_order),
        )
