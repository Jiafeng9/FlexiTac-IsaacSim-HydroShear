from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any

import numpy as np
import torch

from ..helpers.structure import sensor_slot
from tactile.backend import (
    HydroShearTactileBackend as CoreHydroShearTactileBackend,
    HydroShearTactileBackendCfg as CoreHydroShearTactileBackendCfg,
)
from tactile.elastomer import inverse_transform_points, quat_apply_wxyz, quat_conjugate_wxyz
from tactile.hydroshear import BumpHydroShearCfg, SurfacePointHydroShearCfg
from tactile.readout import (
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
)
from tactile.surface import ObjectSurfaceSampler, ObjectSurfaceSamplerCfg


@dataclass
class HydroShearTactileBackendCfg:
    """ALOHA adapter config for the surface-point HydroShear tactile backend."""

    dilation_decay: float = 1.0
    shear_decay: float = 0.1
    E: float = 1.0
    K: float = 1.0
    A: float = 1.0
    mu: float = 10_000.0
    normal_stiffness: float | None = None
    shear_stiffness: float | None = None
    friction_coefficient: float | None = None
    area_mode: str = "unit"
    normal_motion_deadband: float = 1.0e-6
    max_frame_displacement: float | None = None
    readout_ema_alpha: float = 1.0
    surface_point_count: int = 2048
    surface_sample_seed: int | None = 0
    surface_poisson_radius: float = 0.00075
    surface_poisson_initial_num_points: int = 50_000
    taxel_surface_margin: float = 0.003
    taxel_surface_local_z_dir: float = -1.0
    normal_direction: float | None = None
    bump_enabled: bool = False
    bump_rows: int = 6
    bump_cols: int = 16
    bump_shape_from_active_area: bool = False
    bump_pitch_mm: float | tuple[float, float] = 4.0
    bump_active_width_mm: float = 26.0
    bump_active_length_mm: float = 66.65
    bump_sim_active_width_mm: float | None = None
    bump_sim_active_length_mm: float | None = None
    bump_center_source: str = "aloha_bump_pad"
    bump_center_surface_band: float | None = None
    bump_center_surface_band_ratio: float = 0.05
    bump_contact_skin_depth: float | None = None
    bump_normal_damping: float = 0.0
    bump_aggregation_mode: str = "weighted_mean"
    bump_reset_on_contact_loss: bool = True
    projected_displacement_lambda_d: float = 1.0
    projected_displacement_decay: float = 0.0
    projected_displacement_max: float | None = None
    projected_displacement_include_normal: bool = True
    projection_lambda_s: float = 300.0
    projection_weight_by_penetration: bool | None = None
    normal_projection_weight_by_penetration: bool = False
    shear_projection_weight_by_penetration: bool = True
    projection_use_3d_distance: bool = True
    normalize_projection_weights: bool = False
    projection_chunk_size: int | None = None
    output_mode: str = "marker_field"
    marker_lambda_s: float = 300.0
    marker_lambda_d: float = 700.0
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
    sdf_resolution: float = 0.001
    sdf_cache_dir: str | None = None
    sdf_clean_cache: bool = False
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
    taxel_positions_p: torch.Tensor | None = None
    last_output: Any | None = None
    last_outputs: list[Any] | None = None


def bump_pitch_pair_mm(backend_cfg) -> tuple[float, float]:
    """Return row/column real-world bump pitch in mm."""

    pitch = getattr(backend_cfg, "bump_pitch_mm", 4.0)
    if isinstance(pitch, (tuple, list)):
        if len(pitch) != 2:
            raise ValueError("bump_pitch_mm tuple must contain (row_pitch_mm, col_pitch_mm)")
        pitch_u_mm, pitch_v_mm = float(pitch[0]), float(pitch[1])
    else:
        pitch_u_mm = pitch_v_mm = float(pitch)
    if pitch_u_mm <= 0.0 or pitch_v_mm <= 0.0:
        raise ValueError("bump_pitch_mm must be positive")
    return pitch_u_mm, pitch_v_mm


def bump_real_active_pair_mm(backend_cfg) -> tuple[float, float]:
    width_mm = float(getattr(backend_cfg, "bump_active_width_mm", 26.0))
    length_mm = float(getattr(backend_cfg, "bump_active_length_mm", 66.65))
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("bump real active width/length must be positive")
    return width_mm, length_mm


def bump_sim_active_pair_mm(backend_cfg) -> tuple[float, float]:
    real_width_mm, real_length_mm = bump_real_active_pair_mm(backend_cfg)
    sim_width = getattr(backend_cfg, "bump_sim_active_width_mm", None)
    sim_length = getattr(backend_cfg, "bump_sim_active_length_mm", None)
    sim_width_mm = real_width_mm if sim_width is None else float(sim_width)
    sim_length_mm = real_length_mm if sim_length is None else float(sim_length)
    if sim_width_mm <= 0.0 or sim_length_mm <= 0.0:
        raise ValueError("bump sim active width/length must be positive")
    return sim_width_mm, sim_length_mm


def bump_sim_pitch_pair_mm(backend_cfg) -> tuple[float, float]:
    real_pitch_u_mm, real_pitch_v_mm = bump_pitch_pair_mm(backend_cfg)
    real_width_mm, real_length_mm = bump_real_active_pair_mm(backend_cfg)
    sim_width_mm, sim_length_mm = bump_sim_active_pair_mm(backend_cfg)
    return real_pitch_u_mm * sim_width_mm / real_width_mm, real_pitch_v_mm * sim_length_mm / real_length_mm


def resolve_bump_grid_shape(backend_cfg) -> tuple[int, int]:
    """Resolve bump rows/cols from either explicit counts or active area."""

    if bool(getattr(backend_cfg, "bump_shape_from_active_area", False)):
        pitch_u_mm, pitch_v_mm = bump_pitch_pair_mm(backend_cfg)
        active_width_mm, active_length_mm = bump_real_active_pair_mm(backend_cfg)
        rows = max(1, int(np.floor(active_width_mm / pitch_u_mm + 1.0e-9)))
        cols = max(1, int(np.floor(active_length_mm / pitch_v_mm + 1.0e-9)))
    else:
        rows = int(getattr(backend_cfg, "bump_rows", 6))
        cols = int(getattr(backend_cfg, "bump_cols", 16))
    if rows <= 0 or cols <= 0:
        raise ValueError("bump rows and cols must be positive")
    return rows, cols


def resolve_tactile_grid_shape(backend_cfg, tactile_cfg) -> tuple[int, int]:
    if getattr(backend_cfg, "bump_enabled", False):
        return resolve_bump_grid_shape(backend_cfg)
    return int(tactile_cfg.num_rows), int(tactile_cfg.num_cols)


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

    def _output_grid_shape(self) -> tuple[int, int]:
        return resolve_tactile_grid_shape(self.backend_cfg, self.tactile_cfg)

    def _core_output_mode(self) -> str:
        output_mode = str(getattr(self.backend_cfg, "output_mode", "force_grid"))
        if bool(getattr(self.backend_cfg, "bump_enabled", False)) and output_mode == "marker_field":
            return "bump_force_grid"
        return output_mode

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
            state.elastomer_vertices_p, state.elastomer_faces = self._sample_elastomer_mesh(stage, state)
            if bool(getattr(self.backend_cfg, "bump_enabled", False)):
                state.taxel_positions_p = None
            else:
                state.taxel_positions_p = self._sample_elastomer_taxel_points(state)
            state.core = self._make_core_backend(
                slot=state.slot,
                elastomer_vertices_p=state.elastomer_vertices_p,
                elastomer_faces=state.elastomer_faces,
                elastomer_sdf_object_name=state.link_path,
                taxel_positions_p=state.taxel_positions_p,
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
                            elastomer_sdf_object_name=state.link_path,
                            taxel_positions_p=state.taxel_positions_p,
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
        rows, cols = self._output_grid_shape()
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
        scalar_shape = (num_envs, 4) if batched else (4,)
        contact_count = np.zeros(scalar_shape, dtype=np.float32)
        bump_contact_count = np.zeros(scalar_shape, dtype=np.float32)
        max_penetration = np.zeros(scalar_shape, dtype=np.float32)
        min_sdf = np.zeros(scalar_shape, dtype=np.float32)

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
                contact_count[index] = float(out.contact.contact_mask.sum().detach().cpu())
                if out.bump_surface is not None:
                    bump_contact_count[index] = float(out.bump_surface.contact_mask.sum().detach().cpu())
                    max_penetration[index] = float(out.bump_surface.penetration.max().detach().cpu())
                else:
                    bump_contact_count[index] = contact_count[index]
                    max_penetration[index] = float(out.contact.penetration.max().detach().cpu())
                min_sdf[index] = float(out.contact.sdf.min().detach().cpu())

        obs = {
            self.output_key: tactile,
            f"{self.output_key}_contact_count": contact_count,
            f"{self.output_key}_bump_contact_count": bump_contact_count,
            f"{self.output_key}_max_penetration": max_penetration,
            f"{self.output_key}_min_sdf": min_sdf,
        }
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
        elastomer_sdf_object_name: str | None = None,
        taxel_positions_p: torch.Tensor | None = None,
    ) -> CoreHydroShearTactileBackend:
        tactile_cfg = self.tactile_cfg
        backend_cfg = self.backend_cfg
        rows, cols = self._output_grid_shape()
        normal_stiffness, shear_stiffness, friction_coefficient = self._official_force_params(backend_cfg)
        normal_weight_by_penetration = bool(backend_cfg.normal_projection_weight_by_penetration)
        shear_weight_by_penetration = bool(backend_cfg.shear_projection_weight_by_penetration)
        if backend_cfg.projection_weight_by_penetration is not None:
            normal_weight_by_penetration = bool(backend_cfg.projection_weight_by_penetration)
            shear_weight_by_penetration = bool(backend_cfg.projection_weight_by_penetration)
        shear_axis_signs = self._shear_axis_signs_for_slot(slot)
        normal_direction = self._normal_direction(backend_cfg)
        bump_contact_skin_depth = backend_cfg.bump_contact_skin_depth
        if bump_contact_skin_depth is None:
            bump_contact_skin_depth = abs(float(tactile_cfg.normal_offset))
        bump_centers_p = self._bump_centers_p_for_backend(
            backend_cfg,
            tactile_cfg,
            elastomer_vertices_p,
            normal_direction=normal_direction,
        )
        elastomer_sdf_name = elastomer_sdf_object_name or f"elastomer_{slot if slot is not None else 0}"
        cfg = CoreHydroShearTactileBackendCfg(
            grid=TaxelGridCfg(
                num_rows=rows,
                num_cols=cols,
                point_distance=tactile_cfg.point_distance,
                normal_axis=tactile_cfg.normal_axis,
                normal_offset=tactile_cfg.normal_offset,
                device=self.device,
            ),
            taxel_positions_p=taxel_positions_p,
            elastomer_vertices_p=elastomer_vertices_p,
            elastomer_faces=elastomer_faces,
            elastomer_sdf_query_chunk_size=backend_cfg.elastomer_sdf_query_chunk_size,
            elastomer_sdf_resolution=float(backend_cfg.sdf_resolution),
            elastomer_sdf_cache_path=self._sdf_cache_path(elastomer_sdf_name),
            elastomer_sdf_object_name=elastomer_sdf_name,
            elastomer_sdf_clean_cache=bool(backend_cfg.sdf_clean_cache),
            hydroshear=SurfacePointHydroShearCfg(
                normal_stiffness=normal_stiffness,
                shear_stiffness=shear_stiffness,
                friction_coefficient=friction_coefficient,
                normal_axis=tactile_cfg.normal_axis,
                normal_direction=normal_direction,
                area_mode=backend_cfg.area_mode,
                normal_decay=backend_cfg.dilation_decay,
                shear_decay=backend_cfg.shear_decay,
                motion_deadband=backend_cfg.normal_motion_deadband,
                max_frame_displacement=backend_cfg.max_frame_displacement,
            ),
            bump=BumpHydroShearCfg(
                enabled=bool(getattr(backend_cfg, "bump_enabled", False)),
                num_rows=rows,
                num_cols=cols,
                centers_p=bump_centers_p,
                center_source=backend_cfg.bump_center_source,
                center_surface_band=backend_cfg.bump_center_surface_band,
                center_surface_band_ratio=backend_cfg.bump_center_surface_band_ratio,
                normal_stiffness=normal_stiffness,
                normal_damping=float(backend_cfg.bump_normal_damping),
                shear_stiffness=shear_stiffness,
                friction_coefficient=friction_coefficient,
                normal_axis=tactile_cfg.normal_axis,
                normal_direction=normal_direction,
                contact_skin_depth=float(bump_contact_skin_depth),
                area_mode=backend_cfg.area_mode,
                aggregation_mode=backend_cfg.bump_aggregation_mode,
                shear_decay=backend_cfg.shear_decay,
                reset_on_contact_loss=bool(backend_cfg.bump_reset_on_contact_loss),
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
            output_mode=self._core_output_mode(),
            readout_ema_alpha=backend_cfg.readout_ema_alpha,
            output_key=self.output_key,
        )
        return CoreHydroShearTactileBackend(cfg)

    @staticmethod
    def _bump_centers_p_for_backend(
        backend_cfg,
        tactile_cfg,
        elastomer_vertices_p: torch.Tensor | None,
        *,
        normal_direction: float,
    ) -> torch.Tensor | None:
        if not bool(getattr(backend_cfg, "bump_enabled", False)):
            return None
        source = str(getattr(backend_cfg, "bump_center_source", "mesh_surface"))
        if source == "mesh_surface":
            return None
        if source != "aloha_bump_pad":
            raise ValueError("bump_center_source must be 'aloha_bump_pad' or 'mesh_surface'")
        if elastomer_vertices_p is None:
            raise RuntimeError("ALOHA bump center source requires elastomer_vertices_p")

        vertices = torch.as_tensor(elastomer_vertices_p, dtype=torch.float32)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("elastomer_vertices_p must have shape (num_vertices, 3)")

        normal_axis = int(tactile_cfg.normal_axis)
        if normal_axis != 0:
            raise ValueError("aloha_bump_pad center source expects ALOHA patch normal_axis=0; use mesh_surface otherwise")
        tangent_axes = [0, 1, 2]
        tangent_axes.remove(normal_axis)
        axis_u, axis_v = tangent_axes
        rows, cols = resolve_bump_grid_shape(backend_cfg)

        normal_values = vertices[:, normal_axis]
        bounds = torch.stack((vertices.min(dim=0).values, vertices.max(dim=0).values), dim=0)
        center = 0.5 * (bounds[0] + bounds[1])

        pitch_u_mm, pitch_v_mm = bump_sim_pitch_pair_mm(backend_cfg)
        active_width_mm, active_length_mm = bump_sim_active_pair_mm(backend_cfg)

        def axis_centers(count: int, pitch_mm: float, active_extent_mm: float, axis: int) -> torch.Tensor:
            if count <= 0:
                raise ValueError("bump rows and cols must be positive")
            if pitch_mm <= 0.0:
                raise ValueError("bump pitch must be positive")
            span_mm = float(count - 1) * float(pitch_mm)
            if span_mm > float(active_extent_mm) + 1.0e-6:
                raise ValueError(
                    f"bump grid span {span_mm:.3f} mm exceeds active extent {float(active_extent_mm):.3f} mm"
                )
            offsets_mm = (torch.arange(count, dtype=vertices.dtype, device=vertices.device) - (count - 1) / 2.0) * float(pitch_mm)
            return center[axis] + offsets_mm * 0.001

        u = axis_centers(rows, pitch_u_mm, active_width_mm, axis_u)
        v = axis_centers(cols, pitch_v_mm, active_length_mm, axis_v)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        centers = torch.zeros((rows * cols, 3), dtype=vertices.dtype, device=vertices.device)
        centers[:, axis_u] = uu.reshape(-1)
        centers[:, axis_v] = vv.reshape(-1)
        centers[:, normal_axis] = normal_values.max() if normal_direction >= 0.0 else normal_values.min()
        return centers

    @staticmethod
    def _normal_direction(backend_cfg) -> float:
        normal_direction = getattr(backend_cfg, "normal_direction", None)
        if normal_direction is None:
            normal_direction = getattr(backend_cfg, "taxel_surface_local_z_dir", 1.0)
        return 1.0 if float(normal_direction) >= 0.0 else -1.0

    @staticmethod
    def _official_force_params(backend_cfg) -> tuple[float, float, float]:
        normal_stiffness = backend_cfg.normal_stiffness
        if normal_stiffness is None:
            normal_stiffness = float(backend_cfg.E) * float(backend_cfg.A)
        shear_stiffness = backend_cfg.shear_stiffness
        if shear_stiffness is None:
            shear_stiffness = float(backend_cfg.K) * float(backend_cfg.A)
        friction_coefficient = backend_cfg.friction_coefficient
        if friction_coefficient is None:
            friction_coefficient = float(backend_cfg.mu)
        return float(normal_stiffness), float(shear_stiffness), float(friction_coefficient)

    def _sdf_cache_path(self, object_name: str) -> str | None:
        cache_dir = self.backend_cfg.sdf_cache_dir
        if not cache_dir:
            return None
        return str(Path(os.path.expanduser(str(cache_dir))) / f"{_safe_sdf_name(object_name)}_cached_sdf.pkl")

    def _robot_usd_output_dir(self) -> Path:
        out_dir = getattr(self.robot_cfg, "usd_output_dir", None)
        if out_dir:
            return Path(os.path.expanduser(str(out_dir))).resolve()
        urdf_path = Path(os.path.expanduser(str(self.robot_cfg.urdf_path))).resolve()
        return urdf_path.parents[1] / "output" / "aloha_urdf"

    def _shear_axis_signs_for_slot(self, slot: int | None) -> tuple[float, float]:
        signs_by_slot = self.backend_cfg.shear_axis_signs_by_slot
        if signs_by_slot is not None and slot is not None and 0 <= int(slot) < len(signs_by_slot):
            signs = signs_by_slot[int(slot)]
            return float(signs[0]), float(signs[1])
        signs = self.backend_cfg.shear_axis_signs
        return float(signs[0]), float(signs[1])

    def _sample_target_mesh(self, stage, target_query_path: str, *, sensor_index: int):
        vertices, faces = _usd_mesh_vertices_faces(stage, target_query_path)
        scale = np.asarray(_usd_prim_world_scale(stage, target_query_path), dtype=np.float64).reshape(3)
        vertices = vertices * scale.reshape(1, 3)
        seed = self.backend_cfg.surface_sample_seed
        if seed is not None:
            seed = int(seed) + int(sensor_index)
        sampler = ObjectSurfaceSampler(
            ObjectSurfaceSamplerCfg(
                num_points=int(self.backend_cfg.surface_point_count),
                seed=seed,
                poisson_radius=float(self.backend_cfg.surface_poisson_radius),
                poisson_initial_num_points=int(self.backend_cfg.surface_poisson_initial_num_points),
                sdf_resolution=float(self.backend_cfg.sdf_resolution),
                sdf_cache_path=self._sdf_cache_path(f"target_{sensor_index}_{target_query_path}"),
                sdf_object_name=f"target_{sensor_index}_{target_query_path}",
                sdf_clean_cache=bool(self.backend_cfg.sdf_clean_cache),
                device=self.device,
            )
        )
        return sampler.sample_arrays(vertices, faces)

    def _sample_elastomer_mesh(self, stage, state: HydroShearSensorState) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        try:
            vertices_b, faces = _usd_mesh_vertices_faces_in_root(stage, state.link_path)
        except RuntimeError as exc:
            if "No Mesh prim found" not in str(exc):
                raise
            vertices_b, faces = _usd_elastomer_collider_mesh_vertices_faces(
                stage,
                state.link_path,
                converted_usd_dir=self._robot_usd_output_dir(),
            )
            if self.backend_cfg.debug_print:
                print(
                    f"[HydroShear] USD mesh missing under {state.link_path}; "
                    "using converted USD collider mesh",
                    flush=True,
                )

        faces = _ensure_outward_triangle_winding(vertices_b, faces)
        vertices_b_t = torch.as_tensor(vertices_b, dtype=torch.float32, device=self.device)
        patch_pos_b = _to_tensor(state.patch_pos_b, device=self.device)
        patch_quat_b = _to_tensor(state.patch_quat_b, device=self.device)
        vertices_p = inverse_transform_points(vertices_b_t, patch_pos_b, patch_quat_b)
        return vertices_p, torch.as_tensor(faces, dtype=torch.long, device=self.device)

    def _sample_elastomer_taxel_points(self, state: HydroShearSensorState) -> torch.Tensor:
        if state.elastomer_vertices_p is None or state.elastomer_faces is None:
            raise RuntimeError(f"HydroShear taxel surface points require elastomer mesh: {state.link_path}")
        return _raycast_taxel_points_on_mesh(
            state.elastomer_vertices_p.detach().cpu().numpy(),
            state.elastomer_faces.detach().cpu().numpy(),
            num_rows=int(self.tactile_cfg.num_rows),
            num_cols=int(self.tactile_cfg.num_cols),
            normal_axis=int(self.tactile_cfg.normal_axis),
            margin=float(self.backend_cfg.taxel_surface_margin),
            local_z_dir=float(self.backend_cfg.taxel_surface_local_z_dir),
            device=self.device,
        )

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


def _ensure_outward_triangle_winding(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip closed meshes with inward winding before building a signed SDF."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or faces.ndim != 2 or faces.shape[-1] != 3:
        return faces
    if faces.size == 0:
        return faces
    tri = vertices[faces]
    signed_volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    bounds = vertices.max(axis=0) - vertices.min(axis=0)
    scale_volume = float(np.prod(np.maximum(bounds, 1.0e-12)))
    if signed_volume < -max(1.0e-15, 1.0e-6 * scale_volume):
        return faces[:, [0, 2, 1]].copy()
    return faces


def _usd_mesh_vertices_faces_in_root(stage, root_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh_prim_path = _first_mesh_prim_path_under(stage, root_prim_path)
    return _usd_mesh_vertices_faces_relative_to_root(stage, mesh_prim_path, root_prim_path)


def _usd_elastomer_collider_mesh_vertices_faces(
    stage,
    link_prim_path: str,
    *,
    converted_usd_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the URDF-converter collider mesh for an elastomer link.

    IsaacLab's URDF converter keeps the articulation link at e.g.
    `/World/Robot/left_arm_elastomer_left`, but places converted geometry under
    sibling collections such as `/World/Robot/colliders/<link>/.../mesh`.
    """

    robot_root, link_name = link_prim_path.rsplit("/", 1)
    parent_root = robot_root.rsplit("/", 1)[0] if "/" in robot_root.strip("/") else ""
    base_roots = []
    for base in (robot_root, parent_root, ""):
        if base not in base_roots:
            base_roots.append(base)

    errors: list[str] = []
    for base in base_roots:
        candidate_root = f"{base}/colliders/{link_name}" if base else f"/colliders/{link_name}"
        try:
            mesh_prim_path = _first_mesh_prim_path_under(stage, candidate_root)
            return _usd_mesh_vertices_faces_relative_to_root(stage, mesh_prim_path, link_prim_path)
        except RuntimeError as exc:
            errors.append(str(exc))

    if converted_usd_dir is not None:
        try:
            return _converted_usd_elastomer_collider_mesh_vertices_faces(converted_usd_dir, link_name)
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError("; ".join(errors))


def _converted_usd_elastomer_collider_mesh_vertices_faces(
    converted_usd_dir: Path,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd

    usd_dir = Path(converted_usd_dir)
    candidates: list[Path] = []
    for pattern in (
        "configuration/*_base.usd",
        "configuration/*.usd",
        "*.usd",
    ):
        for path in sorted(usd_dir.glob(pattern)):
            if path not in candidates:
                candidates.append(path)
    for path in (
        usd_dir / "configuration" / "aloha_tactile_base.usd",
        usd_dir / "aloha_tactile.usd",
    ):
        if path not in candidates:
            candidates.append(path)

    errors: list[str] = []
    for usd_path in candidates:
        if not usd_path.is_file():
            errors.append(f"missing converted USD: {usd_path}")
            continue
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            errors.append(f"cannot open converted USD: {usd_path}")
            continue
        collider_root = f"/colliders/{link_name}"
        try:
            mesh_prim_path = _first_mesh_prim_path_under(stage, collider_root)
            return _usd_mesh_vertices_faces(stage, mesh_prim_path)
        except RuntimeError as exc:
            errors.append(f"{usd_path}: {exc}")

    raise RuntimeError("; ".join(errors))


def _first_mesh_prim_path_under(stage, root_prim_path: str) -> str:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid mesh root prim path: {root_prim_path}")

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            return prim.GetPath().pathString
    raise RuntimeError(f"No Mesh prim found under {root_prim_path}")


def _usd_mesh_vertices_faces_relative_to_root(
    stage,
    mesh_prim_path: str,
    root_prim_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(root_prim_path)
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid mesh root prim path: {root_prim_path}")
    if not mesh_prim or not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")

    vertices, faces = _usd_mesh_vertices_faces(stage, mesh_prim_path)
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


def _raycast_taxel_points_on_mesh(
    vertices,
    faces,
    *,
    num_rows: int,
    num_cols: int,
    normal_axis: int,
    margin: float,
    local_z_dir: float,
    device: str,
) -> torch.Tensor:
    import open3d as o3d

    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces_np.shape}")

    bbox_min = vertices_np.min(axis=0)
    bbox_max = vertices_np.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    dims = bbox_max - bbox_min
    slim_axis = int(np.argmin(dims))
    if slim_axis != int(normal_axis):
        raise ValueError(
            f"HydroShear elastomer slim axis {slim_axis} does not match tactile normal_axis {normal_axis}"
        )

    axis_idxs = [0, 1, 2]
    axis_idxs.remove(slim_axis)
    div_sz = (dims[axis_idxs] - float(margin) * 2.0) / (np.array([num_rows, num_cols], dtype=np.float32) + 1.0)
    tactile_points_dx = float(np.min(div_sz))
    if tactile_points_dx <= 0.0:
        raise ValueError(f"Invalid elastomer taxel spacing {tactile_points_dx}; check mesh size and margin")

    u = np.linspace(
        center[axis_idxs[0]] - tactile_points_dx * (float(num_rows) + 1.0) / 2.0,
        center[axis_idxs[0]] + tactile_points_dx * (float(num_rows) + 1.0) / 2.0,
        num_rows + 2,
        dtype=np.float32,
    )[1:-1]
    v = np.linspace(
        center[axis_idxs[1]] - tactile_points_dx * (float(num_cols) + 1.0) / 2.0,
        center[axis_idxs[1]] + tactile_points_dx * (float(num_cols) + 1.0) / 2.0,
        num_cols + 2,
        dtype=np.float32,
    )[1:-1]
    uu, vv = np.meshgrid(u, v, indexing="ij")
    origins = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    origins[:, axis_idxs[0]] = uu.reshape(-1)
    origins[:, axis_idxs[1]] = vv.reshape(-1)
    origins[:, slim_axis] = center[slim_axis]

    ray_dir = np.zeros((origins.shape[0], 3), dtype=np.float32)
    ray_dir[:, slim_axis] = float(local_z_dir)
    rays = np.concatenate((origins, ray_dir), axis=-1)

    legacy_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_np.astype(np.float64)),
        o3d.utility.Vector3iVector(faces_np),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans["t_hit"].numpy().astype(np.float32)
    if not np.all(np.isfinite(t_hit)):
        missing = int(np.size(t_hit) - np.count_nonzero(np.isfinite(t_hit)))
        raise RuntimeError(f"HydroShear taxel raycast missed elastomer mesh for {missing} points")

    points = origins.copy()
    points[:, slim_axis] = origins[:, slim_axis] + t_hit * float(local_z_dir)
    return torch.as_tensor(points, dtype=torch.float32, device=device)


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
    if getattr(backend_cfg, "bump_enabled", False):
        output_mode = getattr(backend_cfg, "output_mode", "force_grid")
        if output_mode == "marker_field":
            output_mode = "bump_force_grid"
        if output_mode == "bump_force_grid":
            return (rows, cols, 3)
    if getattr(backend_cfg, "output_mode", "force_grid") == "marker_field":
        return (rows, cols, 3)
    if getattr(backend_cfg, "output_mode", "force_grid") == "bump_force_grid":
        return (rows, cols, 3)
    return (rows, cols)


def _safe_sdf_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return safe or "mesh"


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
