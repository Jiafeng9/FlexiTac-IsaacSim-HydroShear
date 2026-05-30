from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..helpers.structure import sensor_slot
from tactile.backend import (
    HydroShearTactileBackend as CoreHydroShearTactileBackend,
    HydroShearTactileBackendCfg as CoreHydroShearTactileBackendCfg,
)
from tactile.elastomer import FlatPatchElastomerSdfCfg, inverse_transform_points, quat_apply_wxyz, quat_conjugate_wxyz
from tactile.hydroshear import SurfacePointHydroShearCfg
from tactile.readout import (
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
    create_taxel_grid_points,
)
from tactile.surface import ObjectSurfaceSampler, ObjectSurfaceSamplerCfg
from tactile.taxel_shear import TaxelShearOutput, TaxelShearTracker, TaxelShearTrackerCfg


@dataclass
class WarpSdfTactileBackendCfg:
    """Backend parameters for IsaacLab's WarpSDF tactile sensor."""

    mesh_max_dist: float = 0.20
    mesh_signed: bool = True
    mesh_signed_distance_method: str = "winding"
    mesh_shell_thickness: float = 0.001
    stiffness: float = 5_000.0
    max_force: float = 10.0
    normalize_forces: bool = True
    debug_vis: bool = False


@dataclass
class TaxelShearTactileBackendCfg(WarpSdfTactileBackendCfg):
    """WarpSDF normal readout plus per-taxel tangential shear state."""

    shear_stiffness: float = 500.0
    shear_friction_coefficient: float = 0.5
    shear_contact_threshold: float = 1.0e-6
    shear_decay: float = 0.0
    shear_reset_on_contact_loss: bool = True
    shear_eps: float = 1.0e-8
    shear_force_sign: float = 1.0
    include_force_observations: bool = True
    include_taxel_shear_debug_observations: bool = True


@dataclass
class HydroShearTactileBackendCfg:
    """ALOHA adapter config for the surface-point HydroShear tactile backend."""

    dilation_decay: float = 1.0
    shear_decay: float = 1.0
    normal_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    friction_coefficient: float = 0.5
    area_mode: str = "unit"
    normal_motion_deadband: float = 0.0
    max_frame_displacement: float | None = None
    readout_ema_alpha: float = 1.0
    surface_point_count: int = 2048
    surface_sample_seed: int | None = 0
    surface_smooth_normals: bool = False
    projected_displacement_lambda_d: float = 1.0
    projected_displacement_decay: float = 0.0
    projected_displacement_max: float | None = None
    projected_displacement_include_normal: bool = True
    projection_lambda_s: float = 10_800.0
    projection_weight_by_penetration: bool | None = None
    normal_projection_weight_by_penetration: bool = False
    shear_projection_weight_by_penetration: bool = True
    projection_use_3d_distance: bool = True
    normalize_projection_weights: bool = False
    projection_chunk_size: int | None = None
    output_mode: str = "marker_field"
    marker_lambda_s: float = 10_800.0
    marker_lambda_d: float = 20_000.0
    marker_shear_scale: float = 1000.0 / 0.065
    marker_dilation_scale: float = 1000.0 / 0.065
    marker_sdf_query_chunk_size: int | None = 64
    include_marker_observations: bool = True
    normal_readout_scale: float = 1.0
    shear_readout_scale: float = 1.0
    shear_axis_signs: tuple[float, float] = (1.0, 1.0)
    shear_axis_signs_by_slot: tuple[tuple[float, float], ...] | None = (
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    )
    use_elastomer_mesh_sdf: bool = True
    elastomer_sdf_query_chunk_size: int | None = 2048
    include_force_observations: bool = False
    debug_print: bool = False


@dataclass
class HydroShearSensorState:
    link_path: str
    target_query_path: str
    slot: int
    patch_pos_b: tuple[float, float, float]
    patch_quat_b: tuple[float, float, float, float]
    body_index: int | None = None
    samples: Any | None = None
    core: CoreHydroShearTactileBackend | None = None
    cores: list[CoreHydroShearTactileBackend] | None = None
    elastomer_vertices_p: torch.Tensor | None = None
    elastomer_faces: torch.Tensor | None = None
    last_output: Any | None = None
    last_outputs: list[Any] | None = None


@dataclass
class TaxelShearSensorState:
    link_path: str
    slot: int
    patch_pos_b: tuple[float, float, float]
    patch_quat_b: tuple[float, float, float, float]
    body_index: int | None = None
    taxel_positions_e: torch.Tensor | None = None
    patch_quat_e: torch.Tensor | None = None
    tracker: TaxelShearTracker | None = None
    last_output: TaxelShearOutput | None = None
    last_shear_vec_w: torch.Tensor | None = None


class WarpSdfTactileBackend:
    """ALOHA binding adapter for IsaacLab's WarpSDF tactile sensor."""

    def __init__(self, cfg, patch_transform, WarpSdfTactileSensor, WarpSdfTactileSensorCfg):
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self.backend_cfg = cfg.tactile.backend
        self.patch_transform = patch_transform
        self.WarpSdfTactileSensor = WarpSdfTactileSensor
        self.WarpSdfTactileSensorCfg = WarpSdfTactileSensorCfg

    def create_sensors(self, selected_links: list[str], target_query_paths: list[str]) -> tuple[list, list[int]]:
        sensors: list = []
        slot_order: list[int] = []
        tactile_cfg = self.tactile_cfg
        backend_cfg = self.backend_cfg

        for i, link_path in enumerate(selected_links):
            patch_pos, patch_quat = self.patch_transform.compute(link_path)

            sensor_cfg = self.WarpSdfTactileSensorCfg(
                prim_path=self.robot_cfg.prim_path,
                elastomer_prim_paths=[link_path],
                num_rows=tactile_cfg.num_rows,
                num_cols=tactile_cfg.num_cols,
                point_distance=tactile_cfg.point_distance,
                normal_axis=tactile_cfg.normal_axis,
                normal_offset=tactile_cfg.normal_offset,
                patch_offset_pos_b=patch_pos,
                patch_offset_quat_b=patch_quat,
                target_mesh_prim_path=target_query_paths[i],
                mesh_max_dist=backend_cfg.mesh_max_dist,
                mesh_use_signed_distance=backend_cfg.mesh_signed,
                mesh_signed_distance_method=backend_cfg.mesh_signed_distance_method,
                mesh_smooth_normals=True,
                mesh_shell_thickness=backend_cfg.mesh_shell_thickness,
                stiffness=backend_cfg.stiffness,
                max_force=backend_cfg.max_force,
                normalize_forces=backend_cfg.normalize_forces,
                debug_vis=backend_cfg.debug_vis,
            )
            sensors.append(self.WarpSdfTactileSensor(sensor_cfg))

            slot = sensor_slot(link_path)
            slot_order.append(slot if slot is not None else i)

        return sensors, slot_order

    def initialize_after_sim_reset(self, sensors: list, stage, target_tracker):
        pass

    def update(self, dt: float, sensors: list, target_tracker):
        target_tracker.update_target_poses(sensors)
        for sensor in sensors:
            sensor.update(dt=dt)

    def reset(self, sensors: list):
        for sensor in sensors:
            sensor.reset()

    def close(self, sensors: list):
        sensors.clear()

    def observations(self, sensors: list, sensor_slot_order: list[int]) -> dict[str, np.ndarray]:
        tactile_cfg = self.tactile_cfg
        tactile = np.zeros((4, tactile_cfg.num_rows, tactile_cfg.num_cols), dtype=np.float32)
        for i, sensor in enumerate(sensors):
            data = sensor.data.tactile_points_w
            if data is None:
                continue
            forces = data[0, :, 3].detach().cpu().numpy().astype(np.float32)
            tactile[sensor_slot_order[i]] = forces.reshape(tactile_cfg.num_rows, tactile_cfg.num_cols)
        return {tactile_cfg.output_key: tactile}


class TaxelShearTactileBackend(WarpSdfTactileBackend):
    """ALOHA adapter for WarpSDF normal forces plus taxel-level shear tracking."""

    def __init__(self, cfg, patch_transform, WarpSdfTactileSensor, WarpSdfTactileSensorCfg, robot_asset, device: str):
        super().__init__(cfg, patch_transform, WarpSdfTactileSensor, WarpSdfTactileSensorCfg)
        self.robot_asset = robot_asset
        self.device = device
        self._states: list[TaxelShearSensorState] = []

    def create_sensors(self, selected_links: list[str], target_query_paths: list[str]) -> tuple[list, list[int]]:
        sensors, slot_order = super().create_sensors(selected_links, target_query_paths)
        self._states = []
        for i, link_path in enumerate(selected_links):
            patch_pos, patch_quat = self.patch_transform.compute(link_path)
            slot = sensor_slot(link_path)
            self._states.append(
                TaxelShearSensorState(
                    link_path=link_path,
                    slot=slot if slot is not None else i,
                    patch_pos_b=patch_pos,
                    patch_quat_b=patch_quat,
                )
            )
        return sensors, slot_order

    def initialize_after_sim_reset(self, sensors: list, stage, target_tracker):
        del stage, target_tracker
        for sensor, state in zip(sensors, self._states, strict=False):
            state.body_index = self._resolve_robot_body_index(state.link_path)
            state.taxel_positions_e, state.patch_quat_e = self._make_taxel_geometry(state, sensor=sensor)
            state.tracker = TaxelShearTracker(
                TaxelShearTrackerCfg(
                    shear_stiffness=float(self.backend_cfg.shear_stiffness),
                    friction_coefficient=float(self.backend_cfg.shear_friction_coefficient),
                    shear_decay=float(self.backend_cfg.shear_decay),
                    reset_on_contact_loss=bool(self.backend_cfg.shear_reset_on_contact_loss),
                    force_sign=float(self.backend_cfg.shear_force_sign),
                    eps=float(self.backend_cfg.shear_eps),
                )
            )
            state.last_output = None
            state.last_shear_vec_w = None

    def update(self, dt: float, sensors: list, target_tracker):
        super().update(dt, sensors, target_tracker)
        for i, (sensor, state) in enumerate(zip(sensors, self._states, strict=False)):
            if state.tracker is None:
                continue
            if getattr(sensor, "_points_local_per_sensor", None) is not None:
                state.taxel_positions_e, state.patch_quat_e = self._make_taxel_geometry(state, sensor=sensor)
            elif state.taxel_positions_e is None or state.patch_quat_e is None:
                state.taxel_positions_e, state.patch_quat_e = self._make_taxel_geometry(state, sensor=sensor)
            if state.taxel_positions_e is None or state.patch_quat_e is None:
                continue

            tactile_points = sensor.data.tactile_points_w
            if tactile_points is None:
                continue
            normal = tactile_points[..., 3].to(device=self.device, dtype=torch.float32)
            normal_for_limit = normal * float(self.backend_cfg.max_force) if self.backend_cfg.normalize_forces else normal
            contact = normal_for_limit > float(self.backend_cfg.shear_contact_threshold)

            body_pos_w, body_quat_w = self._sensor_body_pose(sensor, state, num_envs=normal.shape[0])
            target_pos_w, target_quat_w = self._sensor_target_pose(sensor, target_tracker, sensor_index=i)
            object_pos_e, object_quat_e = _relative_pose(
                parent_pos_w=body_pos_w,
                parent_quat_w=body_quat_w,
                child_pos_w=target_pos_w,
                child_quat_w=target_quat_w,
            )

            state.last_output = state.tracker.update(
                taxel_positions_e=state.taxel_positions_e,
                object_pos_e=object_pos_e,
                object_quat_e=object_quat_e,
                normal_force=normal_for_limit,
                contact_mask=contact,
                patch_quat_e=state.patch_quat_e,
                normal_axis=int(self.tactile_cfg.normal_axis),
            )
            state.last_shear_vec_w = self._shear_to_world(
                state.last_output.shear_force_uv,
                body_quat_w=body_quat_w,
                patch_quat_e=state.patch_quat_e,
            )

    def reset(self, sensors: list):
        super().reset(sensors)
        for state in self._states:
            if state.tracker is not None:
                state.tracker.reset()
            state.last_output = None

    def close(self, sensors: list):
        super().close(sensors)
        self._states.clear()

    def observations(self, sensors: list, sensor_slot_order: list[int]) -> dict[str, np.ndarray]:
        obs = super().observations(sensors, sensor_slot_order)
        if not bool(getattr(self.backend_cfg, "include_force_observations", False)):
            return obs

        tactile_cfg = self.tactile_cfg
        rows, cols = tactile_cfg.num_rows, tactile_cfg.num_cols
        tactile = obs[tactile_cfg.output_key]
        tactile_force = np.zeros((4, rows, cols, 3), dtype=np.float32)
        tactile_shear = np.zeros((4, rows, cols, 2), dtype=np.float32)
        normal_force_grid = (
            tactile * float(self.backend_cfg.max_force) if self.backend_cfg.normalize_forces else tactile
        )
        tactile_force[..., 0] = normal_force_grid

        for i, state in enumerate(self._states):
            if state.last_output is None:
                continue
            slot = sensor_slot_order[i]
            shear = state.last_output.shear_force_uv
            if shear.ndim == 3:
                shear = shear[0]
            tactile_shear[slot] = _to_numpy(shear, shape=(rows, cols, 2))
            tactile_force[slot, :, :, 1:] = tactile_shear[slot]

        obs[f"{tactile_cfg.output_key}_force"] = tactile_force
        obs[f"{tactile_cfg.output_key}_shear"] = tactile_shear
        if not bool(getattr(self.backend_cfg, "include_taxel_shear_debug_observations", False)):
            return obs

        slip_ratio = np.zeros((4, rows, cols), dtype=np.float32)
        shear_vec_w = np.zeros((4, rows, cols, 3), dtype=np.float32)
        for i, state in enumerate(self._states):
            if state.last_output is None:
                continue
            slot = sensor_slot_order[i]
            slip = state.last_output.slip_ratio
            vec_w = state.last_shear_vec_w
            if slip.ndim == 2:
                slip = slip[0]
            if vec_w is not None and vec_w.ndim == 3:
                vec_w = vec_w[0]
            slip_ratio[slot] = _to_numpy(slip, shape=(rows, cols))
            if vec_w is not None:
                shear_vec_w[slot] = _to_numpy(vec_w, shape=(rows, cols, 3))
        obs[f"{tactile_cfg.output_key}_slip_ratio"] = slip_ratio
        obs[f"{tactile_cfg.output_key}_shear_vector_w"] = shear_vec_w
        return obs

    def _make_taxel_geometry(self, state: TaxelShearSensorState, *, sensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        points = getattr(sensor, "_points_local_per_sensor", None)
        if points is not None:
            points = points[0].to(device=self.device, dtype=torch.float32)
            return points, _to_tensor(state.patch_quat_b, device=self.device)

        grid = create_taxel_grid_points(
            TaxelGridCfg(
                num_rows=self.tactile_cfg.num_rows,
                num_cols=self.tactile_cfg.num_cols,
                point_distance=self.tactile_cfg.point_distance,
                normal_axis=self.tactile_cfg.normal_axis,
                normal_offset=self.tactile_cfg.normal_offset,
                device=self.device,
            )
        )
        patch_pos_b = _to_tensor(state.patch_pos_b, device=self.device)
        patch_quat_b = _to_tensor(state.patch_quat_b, device=self.device)
        points_e = quat_apply_wxyz(patch_quat_b.unsqueeze(0).expand(grid.shape[0], -1), grid) + patch_pos_b
        return points_e, patch_quat_b

    def _sensor_body_pose(self, sensor, state: TaxelShearSensorState, *, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        pose_sensors = getattr(sensor, "_pose_sensors", None)
        if pose_sensors:
            pose_sensor = pose_sensors[0]
            if getattr(pose_sensor, "is_initialized", False):
                data = pose_sensor.data
                if data.pos_w is not None and data.quat_w is not None:
                    return (
                        data.pos_w[:num_envs, 0].to(device=self.device, dtype=torch.float32),
                        data.quat_w[:num_envs, 0].to(device=self.device, dtype=torch.float32),
                    )

        if state.body_index is None:
            raise RuntimeError(f"TaxelShear sensor is not initialized: {state.link_path}")
        body_pos_w = self.robot_asset.data.body_pos_w[:num_envs, state.body_index].to(
            device=self.device,
            dtype=torch.float32,
        )
        body_quat_w = self.robot_asset.data.body_quat_w[:num_envs, state.body_index].to(
            device=self.device,
            dtype=torch.float32,
        )
        return body_pos_w, body_quat_w

    def _sensor_target_pose(self, sensor, target_tracker, *, sensor_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mesh_pos = getattr(sensor, "_mesh_pos_w", None)
        mesh_quat = getattr(sensor, "_mesh_quat_w", None)
        if mesh_pos is not None and mesh_quat is not None:
            return (
                mesh_pos.to(device=self.device, dtype=torch.float32),
                mesh_quat.to(device=self.device, dtype=torch.float32),
            )

        target_pos_w, target_quat_w = target_tracker.target_pose_for_sensor(sensor_index)
        return _to_tensor(target_pos_w, device=self.device), _to_tensor(target_quat_w, device=self.device)

    def _shear_to_world(
        self,
        shear_uv: torch.Tensor,
        *,
        body_quat_w: torch.Tensor,
        patch_quat_e: torch.Tensor,
    ) -> torch.Tensor:
        shear_uv = shear_uv.to(device=self.device, dtype=torch.float32)
        num_envs, num_points = shear_uv.shape[:2]
        shear_patch = torch.zeros((num_envs, num_points, 3), device=self.device, dtype=torch.float32)
        tangent_axes = [0, 1, 2]
        tangent_axes.remove(int(self.tactile_cfg.normal_axis))
        shear_patch[..., tangent_axes] = shear_uv
        patch_quat = patch_quat_e.view(1, 1, 4).expand(num_envs, num_points, -1)
        shear_body = quat_apply_wxyz(patch_quat, shear_patch)
        body_quat = body_quat_w.view(num_envs, 1, 4).expand(num_envs, num_points, -1)
        return quat_apply_wxyz(body_quat, shear_body)

    def _resolve_robot_body_index(self, link_path: str) -> int:
        body_names = [str(n) for n in self.robot_asset.body_names]
        body_names_l = [n.lower() for n in body_names]
        link_l = link_path.lower()
        leaf = link_l.rsplit("/", 1)[-1]

        for token in (leaf, leaf.replace("_link", "")):
            if token in body_names_l:
                return body_names_l.index(token)

        cands = [
            i
            for i, name in enumerate(body_names_l)
            if name in link_l or leaf in name or name.endswith(leaf) or leaf.endswith(name)
        ]
        if not cands:
            raise RuntimeError(f"Cannot map elastomer prim to robot body: {link_path}. Bodies: {body_names}")
        return min(cands, key=lambda i: len(body_names_l[i]))


class HydroShearTactileBackend:
    """ALOHA runtime adapter for the generic surface-point HydroShear backend."""

    def __init__(self, cfg, patch_transform, robot_asset, device: str, *, backend_cfg=None, output_key: str | None = None):
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self.backend_cfg = backend_cfg if backend_cfg is not None else cfg.tactile.backend
        self.output_key = output_key if output_key is not None else cfg.tactile.output_key
        self.patch_transform = patch_transform
        self.robot_asset = robot_asset
        self.device = device
        self._stage = None

    def create_sensors(self, selected_links: list[str], target_query_paths: list[str]) -> tuple[list, list[int]]:
        sensors: list[HydroShearSensorState] = []
        slot_order: list[int] = []

        for i, link_path in enumerate(selected_links):
            patch_pos, patch_quat = self.patch_transform.compute(link_path)
            slot = sensor_slot(link_path)
            slot = slot if slot is not None else i
            sensors.append(
                HydroShearSensorState(
                    link_path=link_path,
                    target_query_path=target_query_paths[i],
                    slot=slot,
                    patch_pos_b=patch_pos,
                    patch_quat_b=patch_quat,
                )
            )
            slot_order.append(slot)

        return sensors, slot_order

    def initialize_after_sim_reset(self, sensors: list[HydroShearSensorState], stage, target_tracker):
        self._stage = stage
        for i, state in enumerate(sensors):
            state.body_index = self._resolve_robot_body_index(state.link_path)
            state.samples = self._sample_target_mesh(stage, state.target_query_path, sensor_index=i)
            if bool(getattr(self.backend_cfg, "use_elastomer_mesh_sdf", True)):
                state.elastomer_vertices_p, state.elastomer_faces = self._sample_elastomer_mesh(stage, state)
            state.core = self._make_core_backend(
                slot=state.slot,
                elastomer_vertices_p=state.elastomer_vertices_p,
                elastomer_faces=state.elastomer_faces,
            )
            state.cores = [state.core]
            state.last_output = None
            state.last_outputs = None
            if self.backend_cfg.debug_print:
                print(
                    f"[HydroShear] sensor={i} slot={state.slot} body={state.body_index} "
                    f"samples={state.samples.num_points} target={state.target_query_path} "
                    f"elastomer_sdf={'mesh' if state.elastomer_vertices_p is not None else 'flat'}",
                    flush=True,
                )

    def update(self, dt: float, sensors: list[HydroShearSensorState], target_tracker):
        del dt
        for i, state in enumerate(sensors):
            if state.samples is None or state.core is None:
                continue

            patch_pos_w, patch_quat_w = self._patch_world_pose(state)
            target_pos_w, target_quat_w = target_tracker.target_pose_for_sensor(i)
            target_pos_w = _to_tensor(target_pos_w, device=self.device)
            target_quat_w = _to_tensor(target_quat_w, device=self.device)

            target_pos_w, target_quat_w = _match_pose_batch(
                parent_pos=patch_pos_w,
                parent_quat=patch_quat_w,
                child_pos=target_pos_w,
                child_quat=target_quat_w,
            )

            object_pos_e, object_quat_e = _relative_pose(
                parent_pos_w=patch_pos_w,
                parent_quat_w=patch_quat_w,
                child_pos_w=target_pos_w,
                child_quat_w=target_quat_w,
            )
            if object_pos_e.ndim == 1:
                outputs = [
                    state.core.update(
                        state.samples,
                        object_pos_e=object_pos_e,
                        object_quat_e=object_quat_e,
                    )
                ]
            else:
                num_envs = int(object_pos_e.shape[0])
                if state.cores is None:
                    state.cores = []
                while len(state.cores) < num_envs:
                    state.cores.append(
                        self._make_core_backend(
                            slot=state.slot,
                            elastomer_vertices_p=state.elastomer_vertices_p,
                            elastomer_faces=state.elastomer_faces,
                        )
                    )
                outputs = []
                for env_id in range(num_envs):
                    outputs.append(
                        state.cores[env_id].update(
                            state.samples,
                            object_pos_e=object_pos_e[env_id],
                            object_quat_e=object_quat_e[env_id],
                        )
                    )
            state.last_outputs = outputs
            state.last_output = outputs[0] if outputs else None

    def reset(self, sensors: list[HydroShearSensorState]):
        for state in sensors:
            if state.core is not None:
                state.core.reset()
            if state.cores is not None:
                for core in state.cores:
                    core.reset()
            state.last_output = None
            state.last_outputs = None

    def close(self, sensors: list[HydroShearSensorState]):
        sensors.clear()

    def observations(self, sensors: list[HydroShearSensorState], sensor_slot_order: list[int]) -> dict[str, np.ndarray]:
        tactile_cfg = self.tactile_cfg
        rows, cols = tactile_cfg.num_rows, tactile_cfg.num_cols
        primary_shape = _hydroshear_primary_shape(self.backend_cfg, rows, cols)
        num_envs = max((len(state.last_outputs or ([state.last_output] if state.last_output is not None else [])) for state in sensors), default=1)
        batched = num_envs > 1
        tactile_shape = ((num_envs, 4) if batched else (4,)) + primary_shape
        tactile = np.zeros(tactile_shape, dtype=np.float32)

        include_force = bool(getattr(self.backend_cfg, "include_force_observations", False))
        include_marker = bool(getattr(self.backend_cfg, "include_marker_observations", False))
        force_shape = ((num_envs, 4) if batched else (4,)) + (rows, cols, 3)
        shear_shape = ((num_envs, 4) if batched else (4,)) + (rows, cols, 2)
        tactile_force = np.zeros(force_shape, dtype=np.float32) if include_force else None
        tactile_shear = np.zeros(shear_shape, dtype=np.float32) if include_force else None
        tactile_marker = np.zeros(force_shape, dtype=np.float32) if include_marker else None
        tactile_marker_dilation = np.zeros(force_shape, dtype=np.float32) if include_marker else None
        tactile_marker_shear = np.zeros(force_shape, dtype=np.float32) if include_marker else None

        for i, state in enumerate(sensors):
            outputs = state.last_outputs or ([state.last_output] if state.last_output is not None else [])
            if not outputs:
                continue
            slot = sensor_slot_order[i]
            for env_id, out in enumerate(outputs):
                index = (env_id, slot) if batched else slot
                tactile[index] = _to_numpy(out.observations[self.output_key], shape=primary_shape)
                if include_force:
                    tactile_force[index] = _to_numpy(out.observations[f"{self.output_key}_force"], shape=(rows, cols, 3))
                    tactile_shear[index] = _to_numpy(out.observations[f"{self.output_key}_shear"], shape=(rows, cols, 2))
                if include_marker:
                    tactile_marker[index] = _to_numpy(out.observations[f"{self.output_key}_marker"], shape=(rows, cols, 3))
                    tactile_marker_dilation[index] = _to_numpy(
                        out.observations[f"{self.output_key}_marker_dilation"],
                        shape=(rows, cols, 3),
                    )
                    tactile_marker_shear[index] = _to_numpy(
                        out.observations[f"{self.output_key}_marker_shear"],
                        shape=(rows, cols, 3),
                    )

        obs = {self.output_key: tactile}
        if include_force:
            obs[f"{self.output_key}_force"] = tactile_force
            obs[f"{self.output_key}_shear"] = tactile_shear
        if include_marker:
            obs[f"{self.output_key}_marker"] = tactile_marker
            obs[f"{self.output_key}_marker_dilation"] = tactile_marker_dilation
            obs[f"{self.output_key}_marker_shear"] = tactile_marker_shear
        return obs

    def _make_core_backend(
        self,
        *,
        slot: int | None = None,
        elastomer_vertices_p: torch.Tensor | None = None,
        elastomer_faces: torch.Tensor | None = None,
    ) -> CoreHydroShearTactileBackend:
        tactile_cfg = self.tactile_cfg
        backend_cfg = self.backend_cfg
        normal_weight_by_penetration = bool(backend_cfg.normal_projection_weight_by_penetration)
        shear_weight_by_penetration = bool(backend_cfg.shear_projection_weight_by_penetration)
        if backend_cfg.projection_weight_by_penetration is not None:
            normal_weight_by_penetration = bool(backend_cfg.projection_weight_by_penetration)
            shear_weight_by_penetration = bool(backend_cfg.projection_weight_by_penetration)
        shear_axis_signs = self._shear_axis_signs_for_slot(slot)
        cfg = CoreHydroShearTactileBackendCfg(
            grid=TaxelGridCfg(
                num_rows=tactile_cfg.num_rows,
                num_cols=tactile_cfg.num_cols,
                point_distance=tactile_cfg.point_distance,
                normal_axis=tactile_cfg.normal_axis,
                normal_offset=tactile_cfg.normal_offset,
                device=self.device,
            ),
            elastomer=FlatPatchElastomerSdfCfg(
                normal_axis=tactile_cfg.normal_axis,
                surface_offset=tactile_cfg.normal_offset,
                half_extent_u=float(tactile_cfg.point_distance) * float(tactile_cfg.num_rows) * 0.5,
                half_extent_v=float(tactile_cfg.point_distance) * float(tactile_cfg.num_cols) * 0.5,
            ),
            elastomer_vertices_p=elastomer_vertices_p,
            elastomer_faces=elastomer_faces,
            elastomer_sdf_query_chunk_size=backend_cfg.elastomer_sdf_query_chunk_size,
            hydroshear=SurfacePointHydroShearCfg(
                normal_stiffness=backend_cfg.normal_stiffness,
                shear_stiffness=backend_cfg.shear_stiffness,
                friction_coefficient=backend_cfg.friction_coefficient,
                normal_axis=tactile_cfg.normal_axis,
                area_mode=backend_cfg.area_mode,
                normal_decay=backend_cfg.dilation_decay,
                shear_decay=backend_cfg.shear_decay,
                motion_deadband=backend_cfg.normal_motion_deadband,
                max_frame_displacement=backend_cfg.max_frame_displacement,
            ),
            projected_surface=ProjectedSurfacePointTrackerCfg(
                lambda_d=backend_cfg.projected_displacement_lambda_d,
                decay=backend_cfg.projected_displacement_decay,
                max_displacement=backend_cfg.projected_displacement_max,
                include_normal_displacement=backend_cfg.projected_displacement_include_normal,
            ),
            projection=SurfacePointForceProjectorCfg(
                lambda_s=backend_cfg.projection_lambda_s,
                weight_by_penetration=backend_cfg.projection_weight_by_penetration,
                normal_weight_by_penetration=normal_weight_by_penetration,
                shear_weight_by_penetration=shear_weight_by_penetration,
                use_3d_distance=backend_cfg.projection_use_3d_distance,
                normalize_weights=backend_cfg.normalize_projection_weights,
                normal_scale=backend_cfg.normal_readout_scale,
                shear_scale=backend_cfg.shear_readout_scale,
                shear_axis_signs=shear_axis_signs,
                chunk_size=backend_cfg.projection_chunk_size,
            ),
            marker_projection=HydroShearMarkerReadoutCfg(
                lambda_s=backend_cfg.marker_lambda_s,
                lambda_d=backend_cfg.marker_lambda_d,
                shear_scale=backend_cfg.marker_shear_scale,
                dilation_scale=backend_cfg.marker_dilation_scale,
                shear_axis_signs=shear_axis_signs,
                use_3d_distance=backend_cfg.projection_use_3d_distance,
                chunk_size=backend_cfg.projection_chunk_size,
                sdf_query_chunk_size=backend_cfg.marker_sdf_query_chunk_size,
            ),
            output_mode=backend_cfg.output_mode,
            readout_ema_alpha=backend_cfg.readout_ema_alpha,
            output_key=self.output_key,
        )
        return CoreHydroShearTactileBackend(cfg)

    def _shear_axis_signs_for_slot(self, slot: int | None) -> tuple[float, float]:
        signs_by_slot = self.backend_cfg.shear_axis_signs_by_slot
        if signs_by_slot is not None and slot is not None and 0 <= int(slot) < len(signs_by_slot):
            signs = signs_by_slot[int(slot)]
            return float(signs[0]), float(signs[1])
        signs = self.backend_cfg.shear_axis_signs
        return float(signs[0]), float(signs[1])

    def _sample_target_mesh(self, stage, target_query_path: str, *, sensor_index: int):
        vertices, faces = _usd_mesh_vertices_faces(stage, target_query_path)
        try:
            scale = np.asarray(_usd_prim_world_scale(stage, target_query_path), dtype=np.float64).reshape(3)
            vertices = vertices * scale.reshape(1, 3)
        except Exception:
            pass
        seed = self.backend_cfg.surface_sample_seed
        if seed is not None:
            seed = int(seed) + int(sensor_index)
        sampler = ObjectSurfaceSampler(
            ObjectSurfaceSamplerCfg(
                num_points=int(self.backend_cfg.surface_point_count),
                seed=seed,
                smooth_normals=bool(self.backend_cfg.surface_smooth_normals),
                device=self.device,
            )
        )
        return sampler.sample_arrays(vertices, faces)

    def _sample_elastomer_mesh(self, stage, state: HydroShearSensorState) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        try:
            vertices_b, faces = _usd_mesh_vertices_faces_in_root(stage, state.link_path)
        except Exception as e:
            print(f"[WARN] HydroShear elastomer mesh SDF fallback to flat patch for {state.link_path}: {e}", flush=True)
            return None, None

        vertices_b_t = torch.as_tensor(vertices_b, dtype=torch.float32, device=self.device)
        patch_pos_b = _to_tensor(state.patch_pos_b, device=self.device)
        patch_quat_b = _to_tensor(state.patch_quat_b, device=self.device)
        vertices_p = inverse_transform_points(vertices_b_t, patch_pos_b, patch_quat_b)
        return vertices_p, torch.as_tensor(faces, dtype=torch.long, device=self.device)

    def _resolve_robot_body_index(self, link_path: str) -> int:
        body_names = [str(n) for n in self.robot_asset.body_names]
        body_names_l = [n.lower() for n in body_names]
        link_l = link_path.lower()
        leaf = link_l.rsplit("/", 1)[-1]

        for token in (leaf, leaf.replace("_link", "")):
            if token in body_names_l:
                return body_names_l.index(token)

        cands = [
            i
            for i, name in enumerate(body_names_l)
            if name in link_l or leaf in name or name.endswith(leaf) or leaf.endswith(name)
        ]
        if not cands:
            raise RuntimeError(f"Cannot map elastomer prim to robot body: {link_path}. Bodies: {body_names}")
        return min(cands, key=lambda i: len(body_names_l[i]))

    def _patch_world_pose(self, state: HydroShearSensorState) -> tuple[torch.Tensor, torch.Tensor]:
        if state.body_index is None:
            raise RuntimeError(f"HydroShear sensor is not initialized: {state.link_path}")

        body_pos_w = self.robot_asset.data.body_pos_w[:, state.body_index].to(device=self.device, dtype=torch.float32)
        body_quat_w = self.robot_asset.data.body_quat_w[:, state.body_index].to(device=self.device, dtype=torch.float32)
        patch_pos_b = _to_tensor(state.patch_pos_b, device=self.device)
        patch_quat_b = _to_tensor(state.patch_quat_b, device=self.device)

        patch_pos_w = body_pos_w + quat_apply_wxyz(body_quat_w, patch_pos_b.view(1, 3).expand_as(body_pos_w))
        patch_quat_w = _quat_mul_wxyz(body_quat_w, patch_quat_b.view(1, 4).expand_as(body_quat_w))
        return patch_pos_w, patch_quat_w


def _usd_mesh_vertices_faces(stage, mesh_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"HydroShear target query path is not a Mesh prim: {mesh_prim_path}")

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    vertices = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

    faces: list[list[int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + int(count)]
        offset += int(count)
        if count < 3:
            continue
        for k in range(1, int(count) - 1):
            faces.append([int(face[0]), int(face[k]), int(face[k + 1])])

    if vertices.size == 0 or not faces:
        raise RuntimeError(f"HydroShear target mesh has no triangles: {mesh_prim_path}")
    return vertices, np.asarray(faces, dtype=np.int64)


def _usd_mesh_vertices_faces_in_root(stage, root_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid mesh root prim path: {root_prim_path}")

    mesh_prim = None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
    if mesh_prim is None:
        raise RuntimeError(f"No Mesh prim found under {root_prim_path}")

    vertices, faces = _usd_mesh_vertices_faces(stage, mesh_prim.GetPath().pathString)
    mesh_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    root_world = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    root_world_inv = root_world.GetInverse()

    vertices_root = []
    for vertex in vertices:
        point_world = mesh_world.Transform(Gf.Vec3d(float(vertex[0]), float(vertex[1]), float(vertex[2])))
        point_root = root_world_inv.Transform(point_world)
        vertices_root.append([float(point_root[0]), float(point_root[1]), float(point_root[2])])
    return np.asarray(vertices_root, dtype=np.float64), faces


def _usd_prim_world_scale(stage, prim_path: str) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for scale query: {prim_path}")
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(float(v.GetLength()) for v in world_transform.ExtractRotationMatrix())


def _to_tensor(x, *, device: str) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _to_numpy(x, *, shape):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(shape)


def _match_pose_batch(
    *,
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parent_pos.ndim == 2 and child_pos.ndim == 1:
        child_pos = child_pos.view(1, 3).expand(parent_pos.shape[0], -1)
        child_quat = child_quat.view(1, 4).expand(parent_pos.shape[0], -1)
    elif parent_pos.ndim == 1 and child_pos.ndim == 2:
        parent_pos = parent_pos.view(1, 3).expand(child_pos.shape[0], -1)
        parent_quat = parent_quat.view(1, 4).expand(child_pos.shape[0], -1)
    elif parent_pos.ndim == 2 and child_pos.ndim == 2 and parent_pos.shape[0] != child_pos.shape[0]:
        if parent_pos.shape[0] == 1:
            parent_pos = parent_pos.expand(child_pos.shape[0], -1)
            parent_quat = parent_quat.expand(child_pos.shape[0], -1)
        elif child_pos.shape[0] == 1:
            child_pos = child_pos.expand(parent_pos.shape[0], -1)
            child_quat = child_quat.expand(parent_pos.shape[0], -1)
        else:
            raise ValueError(
                f"Cannot match pose batch sizes: parent={parent_pos.shape[0]} child={child_pos.shape[0]}"
            )
    return child_pos, child_quat


def _hydroshear_primary_shape(backend_cfg, rows: int, cols: int) -> tuple[int, ...]:
    if getattr(backend_cfg, "output_mode", "force_grid") == "marker_field":
        return (rows, cols, 3)
    return (rows, cols)


def _relative_pose(
    *,
    parent_pos_w: torch.Tensor,
    parent_quat_w: torch.Tensor,
    child_pos_w: torch.Tensor,
    child_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent_quat_inv = quat_conjugate_wxyz(parent_quat_w)
    child_pos_parent = quat_apply_wxyz(parent_quat_inv, child_pos_w - parent_pos_w)
    child_quat_parent = _quat_mul_wxyz(parent_quat_inv, child_quat_w)
    return child_pos_parent, child_quat_parent


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
