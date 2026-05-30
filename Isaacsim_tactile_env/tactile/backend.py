from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .contact import SurfacePointContactQuery, SurfacePointContactState
from .elastomer import (
    FlatPatchElastomerSdf,
    FlatPatchElastomerSdfCfg,
    MeshPatchElastomerSdf,
    inverse_transform_points,
    transform_points,
)
from .hydroshear import SurfacePointHydroShearCfg, SurfacePointHydroShearOutput, SurfacePointHydroShearTracker
from .readout import (
    HydroShearMarkerProjector,
    HydroShearMarkerReadout,
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointOutput,
    ProjectedSurfacePointTracker,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjector,
    SurfacePointForceProjectorCfg,
    TaxelForceReadout,
    TaxelGridCfg,
)
from .surface import ObjectSurfaceSamples


@dataclass
class HydroShearTactileBackendCfg:
    grid: TaxelGridCfg = field(default_factory=TaxelGridCfg)
    elastomer: FlatPatchElastomerSdfCfg = field(default_factory=FlatPatchElastomerSdfCfg)
    elastomer_vertices_p: torch.Tensor | None = None
    elastomer_faces: torch.Tensor | None = None
    elastomer_sdf_query_chunk_size: int | None = 2048
    hydroshear: SurfacePointHydroShearCfg = field(default_factory=SurfacePointHydroShearCfg)
    projected_surface: ProjectedSurfacePointTrackerCfg = field(default_factory=ProjectedSurfacePointTrackerCfg)
    projection: SurfacePointForceProjectorCfg = field(default_factory=SurfacePointForceProjectorCfg)
    marker_projection: HydroShearMarkerReadoutCfg = field(default_factory=HydroShearMarkerReadoutCfg)
    output_mode: str = "force_grid"
    readout_ema_alpha: float = 1.0
    output_key: str = "tactile"


@dataclass
class HydroShearTactileBackendOutput:
    contact: SurfacePointContactState
    surface: SurfacePointHydroShearOutput
    projected_surface: ProjectedSurfacePointOutput | None
    readout: TaxelForceReadout
    marker_readout: HydroShearMarkerReadout
    observations: dict[str, torch.Tensor]


class HydroShearTactileBackend:
    """Surface-point HydroShear tactile pipeline.

    The caller owns robot/object pose acquisition. This backend expects object
    pose expressed in elastomer frame and returns taxel-grid force readouts.
    """

    def __init__(self, cfg: HydroShearTactileBackendCfg | None = None):
        self.cfg = cfg or HydroShearTactileBackendCfg()
        self.contact_query = SurfacePointContactQuery(self._make_elastomer_sdf())
        self.tracker = SurfacePointHydroShearTracker(self.cfg.hydroshear)
        self.projected_surface_tracker = ProjectedSurfacePointTracker(
            normal_axis=self.cfg.grid.normal_axis,
            cfg=self.cfg.projected_surface,
        )
        self.projector = SurfacePointForceProjector(self.cfg.grid, self.cfg.projection)
        self.marker_projector = HydroShearMarkerProjector(self.cfg.grid, self.cfg.marker_projection)
        self._prev_tactile_force: torch.Tensor | None = None

    def _make_elastomer_sdf(self):
        if self.cfg.elastomer_vertices_p is not None and self.cfg.elastomer_faces is not None:
            return MeshPatchElastomerSdf(
                vertices_p=self.cfg.elastomer_vertices_p,
                faces=self.cfg.elastomer_faces,
                chunk_size=self.cfg.elastomer_sdf_query_chunk_size,
            )
        return FlatPatchElastomerSdf(self.cfg.elastomer)

    def reset(self):
        self.tracker.reset()
        self.projected_surface_tracker.reset()
        self._prev_tactile_force = None

    def update(
        self,
        samples: ObjectSurfaceSamples,
        *,
        object_pos_e: torch.Tensor,
        object_quat_e: torch.Tensor,
        patch_pos_e: torch.Tensor | None = None,
        patch_quat_e: torch.Tensor | None = None,
        projected_surface_points_p: torch.Tensor | None = None,
    ) -> HydroShearTactileBackendOutput:
        contact = self.contact_query.compute(
            samples,
            object_pos_e=object_pos_e,
            object_quat_e=object_quat_e,
            patch_pos_e=patch_pos_e,
            patch_quat_e=patch_quat_e,
        )
        surface = self.tracker.update(samples, contact)
        projected_surface = None
        if projected_surface_points_p is None and float(self.cfg.projected_surface.lambda_d) != 0.0:
            projected_surface = self.projected_surface_tracker.update(
                surface_points_p=contact.points_p,
                displacement_e=surface.displacement_e,
                contact_mask=surface.contact_mask,
                patch_quat_e=patch_quat_e,
            )
            projected_surface_points_p = projected_surface.projected_points_p

        readout = self.projector.project(
            surface_points_p=contact.points_p,
            penetration=surface.penetration,
            normal_force=surface.normal_force,
            shear_force_e=surface.shear_force_e,
            patch_quat_e=patch_quat_e,
            projected_surface_points_p=projected_surface_points_p,
        )
        readout = self._smooth_readout(readout)
        marker_object_sdf = self._query_marker_object_sdf(
            samples,
            object_pos_e=object_pos_e,
            object_quat_e=object_quat_e,
            patch_pos_e=patch_pos_e,
            patch_quat_e=patch_quat_e,
            device=contact.points_p.device,
            dtype=contact.points_p.dtype,
        )
        marker_readout = self.marker_projector.project(
            surface_points_p=contact.points_p,
            penetration=surface.penetration,
            displacement_e=surface.displacement_e,
            patch_quat_e=patch_quat_e,
            projected_surface_points_p=projected_surface_points_p,
            marker_object_sdf=marker_object_sdf,
        )

        observations = {
            self.cfg.output_key: self._primary_observation(readout, marker_readout),
            f"{self.cfg.output_key}_force": readout.tactile_force,
            f"{self.cfg.output_key}_shear": readout.tactile_shear,
            f"{self.cfg.output_key}_marker": marker_readout.marker_field,
            f"{self.cfg.output_key}_marker_dilation": marker_readout.dilation_field,
            f"{self.cfg.output_key}_marker_shear": marker_readout.shear_field,
        }
        return HydroShearTactileBackendOutput(
            contact=contact,
            surface=surface,
            projected_surface=projected_surface,
            readout=readout,
            marker_readout=marker_readout,
            observations=observations,
        )

    def _primary_observation(
        self,
        readout: TaxelForceReadout,
        marker_readout: HydroShearMarkerReadout,
    ) -> torch.Tensor:
        if self.cfg.output_mode == "force_grid":
            return readout.tactile
        if self.cfg.output_mode == "marker_field":
            return marker_readout.marker_field
        raise ValueError("output_mode must be 'force_grid' or 'marker_field'")

    def _query_marker_object_sdf(
        self,
        samples: ObjectSurfaceSamples,
        *,
        object_pos_e,
        object_quat_e,
        patch_pos_e,
        patch_quat_e,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if samples.vertices_o is None or samples.faces is None:
            return None

        taxel_positions_p = self.marker_projector.taxel_positions_p.to(device=device, dtype=dtype)
        if patch_pos_e is None and patch_quat_e is None:
            taxel_positions_e = taxel_positions_p
        elif patch_pos_e is not None and patch_quat_e is not None:
            taxel_positions_e = transform_points(
                taxel_positions_p,
                torch.as_tensor(patch_pos_e, dtype=dtype, device=device),
                torch.as_tensor(patch_quat_e, dtype=dtype, device=device),
            )
        else:
            raise ValueError("patch_pos_e and patch_quat_e must be provided together")

        taxel_positions_o = inverse_transform_points(
            taxel_positions_e,
            torch.as_tensor(object_pos_e, dtype=dtype, device=device),
            torch.as_tensor(object_quat_e, dtype=dtype, device=device),
        )
        return samples.query_sdf(
            taxel_positions_o,
            chunk_size=self.cfg.marker_projection.sdf_query_chunk_size,
        ).sdf

    def _smooth_readout(self, readout: TaxelForceReadout) -> TaxelForceReadout:
        alpha = float(self.cfg.readout_ema_alpha)
        if alpha >= 1.0:
            self._prev_tactile_force = readout.tactile_force.detach().clone()
            return readout
        if alpha <= 0.0:
            alpha = 0.0

        current = readout.tactile_force
        if self._prev_tactile_force is None or self._prev_tactile_force.shape != current.shape:
            smoothed = current
        else:
            prev = self._prev_tactile_force.to(device=current.device, dtype=current.dtype)
            smoothed = alpha * current + (1.0 - alpha) * prev

        self._prev_tactile_force = smoothed.detach().clone()
        return TaxelForceReadout(
            taxel_positions_p=readout.taxel_positions_p,
            tactile_force=smoothed,
            normal_force=smoothed[..., 0],
            shear_force_uv=smoothed[..., 1:],
            weight_sum=readout.weight_sum,
        )
