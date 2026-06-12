from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .geometry import (
    MeshPatchElastomerSdf,
    ObjectSurfaceSamples,
    SurfacePointContactQuery,
    SurfacePointContactState,
    inverse_transform_points,
    transform_points,
)
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


# ---------------------------------------------------------------------------
# Hydroshear
# ---------------------------------------------------------------------------

@dataclass
class SurfacePointHydroShearCfg:
    normal_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    friction_coefficient: float = 10_000.0
    normal_mode: str = "fixed_axis"
    normal_axis: int = 2
    normal_direction: float = 1.0
    area_mode: str = "unit"
    normal_decay: float = 1.0
    shear_decay: float = 1.0
    motion_deadband: float = 0.0
    max_frame_displacement: float | None = None
    eps: float = 1.0e-8


@dataclass
class SurfacePointHydroShearState:
    prev_points_e: torch.Tensor
    prev_sdf: torch.Tensor
    normal_displacement: torch.Tensor
    shear_displacement_e: torch.Tensor
    initialized: torch.Tensor


@dataclass
class SurfacePointHydroShearOutput:
    displacement_e: torch.Tensor
    normal_displacement: torch.Tensor
    shear_displacement_e: torch.Tensor
    force_e: torch.Tensor
    normal_force: torch.Tensor
    shear_force_e: torch.Tensor
    alpha: torch.Tensor
    contact_mask: torch.Tensor
    penetration: torch.Tensor
    state: SurfacePointHydroShearState


class SurfacePointHydroShearTracker:
    """HydroShear-style recurrent displacement state per object surface point."""

    def __init__(self, cfg: SurfacePointHydroShearCfg | None = None):
        self.cfg = cfg or SurfacePointHydroShearCfg()
        self.state: SurfacePointHydroShearState | None = None

    def reset(self):
        self.state = None

    def update(
        self,
        samples: ObjectSurfaceSamples,
        contact: SurfacePointContactState,
    ) -> SurfacePointHydroShearOutput:
        area_scale = self._area_scale(samples, contact)

        if self.state is None or self.state.prev_points_e.shape != contact.points_e.shape:
            self.state = self._new_state(contact)

        assert self.state is not None
        prev = self.state

        alpha = contact_segment_fraction(prev.prev_sdf, contact.sdf, eps=float(self.cfg.eps))
        alpha = torch.where(prev.initialized, alpha, torch.zeros_like(alpha))

        if self.cfg.normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2")

        # Follow the HydroShear update convention: track the displacement of
        # the previous in-contact surface point toward the current one.
        d_total = prev.prev_points_e - contact.points_e
        d_contact = alpha.unsqueeze(-1) * d_total
        d_contact = self._stabilize_frame_displacement(d_contact)

        normal_axis = int(self.cfg.normal_axis)
        normal_direction = 1.0 if float(self.cfg.normal_direction) >= 0.0 else -1.0
        normal_mode = str(self.cfg.normal_mode)
        if normal_mode == "sdf_normal":
            if contact.elastomer_normals_e is None:
                raise RuntimeError("normal_mode='sdf_normal' requires elastomer SDF normals")
            normal_basis_e = contact.elastomer_normals_e.to(device=d_contact.device, dtype=d_contact.dtype)
            normal_basis_e = normal_basis_e / normal_basis_e.norm(
                dim=-1, keepdim=True
            ).clamp_min(float(self.cfg.eps))

            fixed_basis = torch.zeros_like(normal_basis_e)
            fixed_basis[..., normal_axis] = normal_direction
            flip = (normal_basis_e * fixed_basis).sum(dim=-1, keepdim=True) < 0.0
            normal_basis_e = torch.where(flip, -normal_basis_e, normal_basis_e)

            d_n_scalar = (d_contact * normal_basis_e).sum(dim=-1)
            d_tangent = d_contact - d_n_scalar.unsqueeze(-1) * normal_basis_e
        elif normal_mode == "fixed_axis":
            normal_basis_e = torch.zeros_like(contact.points_e)
            normal_basis_e[..., normal_axis] = normal_direction
            d_n_scalar = normal_direction * d_contact[..., normal_axis]
            d_tangent = d_contact.clone()
            d_tangent[..., normal_axis] = 0.0
        else:
            raise ValueError("normal_mode must be 'fixed_axis' or 'sdf_normal'")

        prev_normal = float(self.cfg.normal_decay) * prev.normal_displacement
        prev_shear_e = float(self.cfg.shear_decay) * prev.shear_displacement_e

        normal_candidate = prev_normal + float(self.cfg.normal_stiffness) * area_scale * d_n_scalar
        normal_displacement = normal_candidate.clamp_min(0.0)

        shear_candidate = prev_shear_e + float(self.cfg.shear_stiffness) * area_scale.unsqueeze(-1) * d_tangent
        limit = float(self.cfg.friction_coefficient) * normal_displacement
        shear_norm = shear_candidate.norm(dim=-1)
        shear_scale = torch.minimum(torch.ones_like(limit), limit / (shear_norm + float(self.cfg.eps)))
        shear_displacement_e = shear_candidate * shear_scale.unsqueeze(-1)

        normal_displacement = torch.where(contact.contact_mask, normal_displacement, torch.zeros_like(normal_displacement))
        shear_displacement_e = torch.where(
            contact.contact_mask.unsqueeze(-1),
            shear_displacement_e,
            torch.zeros_like(shear_displacement_e),
        )
        normal_displacement_e = normal_displacement.unsqueeze(-1) * normal_basis_e
        displacement_e = normal_displacement_e + shear_displacement_e

        self.state = SurfacePointHydroShearState(
            prev_points_e=contact.points_e.detach().clone(),
            prev_sdf=contact.sdf.detach().clone(),
            normal_displacement=normal_displacement.detach().clone(),
            shear_displacement_e=shear_displacement_e.detach().clone(),
            initialized=torch.ones_like(contact.sdf, dtype=torch.bool),
        )

        return SurfacePointHydroShearOutput(
            displacement_e=displacement_e,
            normal_displacement=normal_displacement,
            shear_displacement_e=shear_displacement_e,
            force_e=displacement_e,
            normal_force=normal_displacement,
            shear_force_e=shear_displacement_e,
            alpha=alpha,
            contact_mask=contact.contact_mask,
            penetration=contact.penetration,
            state=self.state,
        )

    def _stabilize_frame_displacement(self, d_contact: torch.Tensor) -> torch.Tensor:
        deadband = float(self.cfg.motion_deadband)
        if deadband > 0.0:
            norm = d_contact.norm(dim=-1, keepdim=True)
            d_contact = torch.where(norm >= deadband, d_contact, torch.zeros_like(d_contact))

        max_displacement = self.cfg.max_frame_displacement
        if max_displacement is not None and max_displacement > 0.0:
            norm = d_contact.norm(dim=-1, keepdim=True)
            scale = torch.minimum(
                torch.ones_like(norm),
                torch.as_tensor(float(max_displacement), dtype=norm.dtype, device=norm.device)
                / norm.clamp_min(float(self.cfg.eps)),
            )
            d_contact = d_contact * scale

        return d_contact

    def _area_scale(self, samples: ObjectSurfaceSamples, contact: SurfacePointContactState) -> torch.Tensor:
        if str(self.cfg.area_mode) == "unit":
            return torch.ones_like(contact.sdf)
        if str(self.cfg.area_mode) == "mesh_area":
            area = samples.area.to(device=contact.points_e.device, dtype=contact.points_e.dtype)
            if area.shape != contact.sdf.shape:
                raise ValueError(f"area shape {tuple(area.shape)} must match sdf shape {tuple(contact.sdf.shape)}")
            return area
        raise ValueError("area_mode must be 'unit' or 'mesh_area'")

    @staticmethod
    def _new_state(contact: SurfacePointContactState) -> SurfacePointHydroShearState:
        return SurfacePointHydroShearState(
            prev_points_e=contact.points_e.detach().clone(),
            prev_sdf=contact.sdf.detach().clone(),
            normal_displacement=torch.zeros_like(contact.sdf),
            shear_displacement_e=torch.zeros_like(contact.points_e),
            initialized=torch.zeros_like(contact.sdf, dtype=torch.bool),
        )


def contact_segment_fraction(prev_sdf: torch.Tensor, curr_sdf: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Original HydroFOTS contact-internal displacement fraction.

    This mirrors:
    `-(relu(-prev_sdf) - relu(-curr_sdf)) / (prev_sdf - curr_sdf)`,
    with the original NaN fill value based on the current SDF sign.
    """

    del eps
    alpha = -torch.divide(torch.relu(-prev_sdf) - torch.relu(-curr_sdf), prev_sdf - curr_sdf)
    nan_fill = -0.5 * (torch.sign(curr_sdf) - 1.0)
    alpha = torch.where(torch.isnan(alpha), nan_fill, alpha)
    return alpha.clamp(0.0, 1.0)

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@dataclass
class HydroShearTactileBackendCfg:
    grid: TaxelGridCfg = field(default_factory=TaxelGridCfg)
    taxel_positions_p: torch.Tensor | None = None
    elastomer_vertices_p: torch.Tensor | None = None
    elastomer_faces: torch.Tensor | None = None
    elastomer_sdf_query_chunk_size: int | None = 2048
    elastomer_sdf_resolution: float = 0.001
    elastomer_sdf_cache_path: str | None = None
    elastomer_sdf_object_name: str = "elastomer"
    elastomer_sdf_clean_cache: bool = False
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
        self.projector = SurfacePointForceProjector(
            self.cfg.grid,
            self.cfg.projection,
            taxel_positions_p=self.cfg.taxel_positions_p,
        )
        self.marker_projector = HydroShearMarkerProjector(
            self.cfg.grid,
            self.cfg.marker_projection,
            taxel_positions_p=self.cfg.taxel_positions_p,
        )
        self._prev_tactile_force: torch.Tensor | None = None

    def _make_elastomer_sdf(self):
        if self.cfg.elastomer_vertices_p is None or self.cfg.elastomer_faces is None:
            raise RuntimeError("HydroShearTactileBackend requires elastomer mesh vertices/faces")
        return MeshPatchElastomerSdf(
            vertices_p=self.cfg.elastomer_vertices_p,
            faces=self.cfg.elastomer_faces,
            chunk_size=self.cfg.elastomer_sdf_query_chunk_size,
            sdf_resolution=self.cfg.elastomer_sdf_resolution,
            sdf_cache_path=self.cfg.elastomer_sdf_cache_path,
            sdf_object_name=self.cfg.elastomer_sdf_object_name,
            sdf_clean_cache=self.cfg.elastomer_sdf_clean_cache,
        )

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
