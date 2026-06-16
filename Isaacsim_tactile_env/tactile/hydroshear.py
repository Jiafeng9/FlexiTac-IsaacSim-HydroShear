from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .geometry import (
    MeshPatchElastomerSdf,
    ObjectSurfaceSamples,
    SurfacePointContactQuery,
    SurfacePointContactState,
    inverse_transform_points,
    rotate_vectors,
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
    tangential_axes,
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
    shear_decay: float = 0.1
    motion_deadband: float = 1.0e-6
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
        prev_shear_raw_e = prev.shear_displacement_e.to(device=d_tangent.device, dtype=d_tangent.dtype)
        prev_shear_e = float(self.cfg.shear_decay) * prev_shear_raw_e

        normal_candidate = prev_normal + float(self.cfg.normal_stiffness) * area_scale * d_n_scalar
        normal_displacement = normal_candidate.clamp_min(0.0)

        shear_update_e = prev_shear_e + float(self.cfg.shear_stiffness) * area_scale.unsqueeze(-1) * d_tangent
        tangent_motion = d_tangent.norm(dim=-1, keepdim=True) > float(self.cfg.eps)
        shear_candidate = torch.where(tangent_motion, shear_update_e, prev_shear_raw_e)
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


@dataclass
class BumpHydroShearCfg:
    """Per-bump HydroShear state with penetration-based normal force."""

    enabled: bool = False
    num_rows: int = 4
    num_cols: int = 8
    centers_p: torch.Tensor | None = None
    center_source: str = "mesh_surface"
    center_surface_band: float | None = None
    center_surface_band_ratio: float = 0.05
    normal_stiffness: float = 1.0
    normal_damping: float = 0.0
    shear_stiffness: float = 1.0
    friction_coefficient: float = 10_000.0
    normal_axis: int = 2
    normal_direction: float = 1.0
    contact_skin_depth: float = 0.0
    area_mode: str = "unit"
    aggregation_mode: str = "weighted_mean"
    shear_decay: float = 0.1
    reset_on_contact_loss: bool = True
    motion_deadband: float = 1.0e-6
    max_frame_displacement: float | None = None
    eps: float = 1.0e-8


@dataclass
class BumpHydroShearState:
    prev_points_e: torch.Tensor
    prev_sdf: torch.Tensor
    prev_penetration: torch.Tensor
    prev_bump_ids: torch.Tensor
    prev_contact_mask: torch.Tensor
    normal_force: torch.Tensor
    shear_force_e: torch.Tensor
    initialized: torch.Tensor


@dataclass
class BumpHydroShearOutput:
    centers_p: torch.Tensor
    bump_ids: torch.Tensor
    alpha: torch.Tensor
    contact_mask: torch.Tensor
    penetration: torch.Tensor
    normal_force: torch.Tensor
    shear_force_e: torch.Tensor
    force_e: torch.Tensor
    delta_shear_e: torch.Tensor
    state: BumpHydroShearState


class BumpHydroShearTracker:
    """HydroShear-style force memory attached to discrete bump cells.

    Object surface points still provide frame-to-frame motion increments, but
    normal and shear force state are stored per bump. This prevents a surface
    point from carrying shear history from one bump to another.
    """

    def __init__(self, cfg: BumpHydroShearCfg | None = None):
        self.cfg = cfg or BumpHydroShearCfg()
        self.state: BumpHydroShearState | None = None

    @property
    def num_bumps(self) -> int:
        return int(self.cfg.num_rows) * int(self.cfg.num_cols)

    def reset(self):
        self.state = None

    def update(
        self,
        samples: ObjectSurfaceSamples,
        contact: SurfacePointContactState,
        *,
        centers_p: torch.Tensor,
        patch_quat_e: torch.Tensor | None = None,
    ) -> BumpHydroShearOutput:
        centers_p = torch.as_tensor(centers_p, dtype=contact.points_p.dtype, device=contact.points_p.device)
        if centers_p.ndim != 2 or centers_p.shape[-1] != 3:
            raise ValueError("centers_p must have shape (num_bumps, 3)")
        if centers_p.shape[0] != self.num_bumps:
            raise ValueError(f"centers_p has {centers_p.shape[0]} centers, expected {self.num_bumps}")

        area_scale = self._area_scale(samples, contact)
        bump_ids = self._assign_bumps(contact.points_p, centers_p)

        if (
            self.state is None
            or self.state.prev_points_e.shape != contact.points_e.shape
            or self.state.normal_force.shape[0] != centers_p.shape[0]
        ):
            self.state = self._new_state(contact, bump_ids, centers_p.shape[0])

        assert self.state is not None
        prev = self.state

        alpha = contact_segment_fraction(prev.prev_sdf, contact.sdf, eps=float(self.cfg.eps))
        alpha = torch.where(prev.initialized, alpha, torch.zeros_like(alpha))

        d_total = prev.prev_points_e - contact.points_e
        d_contact = alpha.unsqueeze(-1) * d_total
        d_contact = self._stabilize_frame_displacement(d_contact)

        normal_basis_e = self._normal_basis_e(contact.points_e, patch_quat_e)
        d_normal = (d_contact * normal_basis_e).sum(dim=-1)
        d_tangent = d_contact - d_normal.unsqueeze(-1) * normal_basis_e

        bump_contact = self._effective_contact_mask(contact)
        bump_penetration = self._effective_penetration(contact)

        num_bumps = centers_p.shape[0]
        bump_contact_mask = self._scatter_any(bump_ids, bump_contact, num_bumps)

        penetration_b = self._aggregate_bump_values(
            bump_ids,
            bump_penetration,
            area_scale,
            num_bumps,
            mask=bump_contact,
        )

        damping = torch.zeros_like(penetration_b)
        if float(self.cfg.normal_damping) != 0.0:
            same_bump = bump_ids == prev.prev_bump_ids.to(device=bump_ids.device)
            valid_damping = bump_contact & prev.initialized & same_bump
            damping = self._aggregate_bump_values(
                bump_ids,
                bump_penetration - prev.prev_penetration,
                area_scale,
                num_bumps,
                mask=valid_damping,
            )

        normal_force = (
            float(self.cfg.normal_stiffness) * penetration_b
            + float(self.cfg.normal_damping) * damping
        ).clamp_min(0.0)

        same_bump = bump_ids == prev.prev_bump_ids.to(device=bump_ids.device)
        valid_shear = bump_contact & prev.prev_contact_mask & prev.initialized & same_bump
        delta_shear_e = self._aggregate_bump_values(
            bump_ids,
            d_tangent,
            area_scale,
            num_bumps,
            mask=valid_shear,
        )

        prev_shear_force_e = prev.shear_force_e.to(device=delta_shear_e.device, dtype=delta_shear_e.dtype)
        shear_update_e = float(self.cfg.shear_decay) * prev_shear_force_e + float(self.cfg.shear_stiffness) * delta_shear_e
        point_tangent_motion = valid_shear & (d_tangent.norm(dim=-1) > float(self.cfg.eps))
        bump_tangent_motion = self._scatter_any(bump_ids, point_tangent_motion, num_bumps)
        shear_candidate = torch.where(bump_tangent_motion.unsqueeze(-1), shear_update_e, prev_shear_force_e)
        limit = float(self.cfg.friction_coefficient) * normal_force
        shear_norm = shear_candidate.norm(dim=-1)
        shear_scale = torch.minimum(torch.ones_like(limit), limit / (shear_norm + float(self.cfg.eps)))
        shear_force_e = shear_candidate * shear_scale.unsqueeze(-1)
        if bool(self.cfg.reset_on_contact_loss):
            shear_force_e = torch.where(bump_contact_mask.unsqueeze(-1), shear_force_e, torch.zeros_like(shear_force_e))
        else:
            shear_force_e = torch.where(
                bump_contact_mask.unsqueeze(-1),
                shear_force_e,
                float(self.cfg.shear_decay) * prev_shear_force_e.to(device=shear_force_e.device, dtype=shear_force_e.dtype),
            )

        force_e = normal_force.unsqueeze(-1) * self._normal_basis_e(centers_p, patch_quat_e) + shear_force_e

        self.state = BumpHydroShearState(
            prev_points_e=contact.points_e.detach().clone(),
            prev_sdf=contact.sdf.detach().clone(),
            prev_penetration=bump_penetration.detach().clone(),
            prev_bump_ids=bump_ids.detach().clone(),
            prev_contact_mask=bump_contact.detach().clone(),
            normal_force=normal_force.detach().clone(),
            shear_force_e=shear_force_e.detach().clone(),
            initialized=torch.ones_like(contact.sdf, dtype=torch.bool),
        )

        return BumpHydroShearOutput(
            centers_p=centers_p,
            bump_ids=bump_ids,
            alpha=alpha,
            contact_mask=bump_contact_mask,
            penetration=penetration_b,
            normal_force=normal_force,
            shear_force_e=shear_force_e,
            force_e=force_e,
            delta_shear_e=delta_shear_e,
            state=self.state,
        )

    def _assign_bumps(self, points_p: torch.Tensor, centers_p: torch.Tensor) -> torch.Tensor:
        axis_u, axis_v = tangential_axes(int(self.cfg.normal_axis))
        diff = points_p[:, None, (axis_u, axis_v)] - centers_p[None, :, (axis_u, axis_v)]
        dist2 = (diff * diff).sum(dim=-1)
        return torch.argmin(dist2, dim=-1).to(dtype=torch.long)

    def _effective_penetration(self, contact: SurfacePointContactState) -> torch.Tensor:
        skin_depth = max(0.0, float(self.cfg.contact_skin_depth))
        if skin_depth <= 0.0:
            return contact.penetration
        return (torch.as_tensor(skin_depth, dtype=contact.sdf.dtype, device=contact.sdf.device) - contact.sdf).clamp_min(0.0)

    def _effective_contact_mask(self, contact: SurfacePointContactState) -> torch.Tensor:
        skin_depth = max(0.0, float(self.cfg.contact_skin_depth))
        if skin_depth <= 0.0:
            return contact.contact_mask
        return contact.sdf < torch.as_tensor(skin_depth, dtype=contact.sdf.dtype, device=contact.sdf.device)

    def _normal_basis_e(self, points: torch.Tensor, patch_quat_e: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2")
        normal_direction = 1.0 if float(self.cfg.normal_direction) >= 0.0 else -1.0
        normal_basis_p = torch.zeros_like(points)
        normal_basis_p[..., int(self.cfg.normal_axis)] = normal_direction
        if patch_quat_e is None:
            return normal_basis_p
        return rotate_vectors(normal_basis_p, torch.as_tensor(patch_quat_e, dtype=points.dtype, device=points.device))

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
    def _scatter_sum(
        ids: torch.Tensor,
        values: torch.Tensor,
        num_bumps: int,
        *,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ids = ids.to(dtype=torch.long, device=values.device)
        mask = mask.to(dtype=torch.bool, device=values.device)
        out = torch.zeros((num_bumps,) + values.shape[1:], dtype=values.dtype, device=values.device)
        if bool(mask.any()):
            out.index_add_(0, ids[mask], values[mask])
        return out

    def _aggregate_bump_values(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
        weights: torch.Tensor,
        num_bumps: int,
        *,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights.to(device=values.device, dtype=values.dtype)
        weighted_values = values * weights.reshape(weights.shape + (1,) * (values.ndim - weights.ndim))
        total = self._scatter_sum(ids, weighted_values, num_bumps, mask=mask)
        mode = str(self.cfg.aggregation_mode)
        if mode == "sum":
            return total
        if mode == "weighted_mean":
            weight_sum = self._scatter_sum(ids, weights, num_bumps, mask=mask)
            return total / weight_sum.clamp_min(float(self.cfg.eps)).reshape(
                weight_sum.shape + (1,) * (total.ndim - weight_sum.ndim)
            )
        raise ValueError("aggregation_mode must be 'weighted_mean' or 'sum'")

    @staticmethod
    def _scatter_any(ids: torch.Tensor, mask: torch.Tensor, num_bumps: int) -> torch.Tensor:
        values = torch.ones_like(ids, dtype=torch.int64)
        out = torch.zeros(num_bumps, dtype=torch.int64, device=ids.device)
        active = mask.to(dtype=torch.bool, device=ids.device)
        if bool(active.any()):
            out.index_add_(0, ids[active].to(dtype=torch.long), values[active])
        return out > 0

    @staticmethod
    def _new_state(contact: SurfacePointContactState, bump_ids: torch.Tensor, num_bumps: int) -> BumpHydroShearState:
        return BumpHydroShearState(
            prev_points_e=contact.points_e.detach().clone(),
            prev_sdf=contact.sdf.detach().clone(),
            prev_penetration=contact.penetration.detach().clone(),
            prev_bump_ids=bump_ids.detach().clone(),
            prev_contact_mask=torch.zeros_like(contact.contact_mask, dtype=torch.bool),
            normal_force=torch.zeros(num_bumps, dtype=contact.sdf.dtype, device=contact.sdf.device),
            shear_force_e=torch.zeros((num_bumps, 3), dtype=contact.points_e.dtype, device=contact.points_e.device),
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


def create_bump_grid_centers(
    cfg: BumpHydroShearCfg,
    grid_cfg: TaxelGridCfg,
    elastomer_vertices_p: torch.Tensor | None,
) -> torch.Tensor:
    """Create row-major bump centers in patch frame from elastomer mesh geometry."""

    if elastomer_vertices_p is None:
        raise RuntimeError("Bump HydroShear requires elastomer_vertices_p or explicit bump centers")
    if int(cfg.num_rows) <= 0 or int(cfg.num_cols) <= 0:
        raise ValueError("bump num_rows and num_cols must be positive")

    vertices = torch.as_tensor(elastomer_vertices_p, dtype=torch.float32)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("elastomer_vertices_p must have shape (num_vertices, 3)")

    source = str(cfg.center_source)
    if source == "mesh_surface":
        centers = _mesh_surface_bump_centers(cfg, grid_cfg, vertices)
        if centers is not None:
            return centers
        raise RuntimeError("Could not infer bump centers from sensing-surface vertices; pass centers_p explicitly")
    raise ValueError("bump center_source must be 'mesh_surface'")


def _mesh_surface_bump_centers(
    cfg: BumpHydroShearCfg,
    grid_cfg: TaxelGridCfg,
    vertices: torch.Tensor,
) -> torch.Tensor | None:
    normal_axis = int(grid_cfg.normal_axis)
    axis_u, axis_v = tangential_axes(normal_axis)
    normal_direction = 1.0 if float(cfg.normal_direction) >= 0.0 else -1.0
    signed_normal = vertices[:, normal_axis] * normal_direction
    normal_span = signed_normal.max() - signed_normal.min()
    band = cfg.center_surface_band
    if band is None:
        band = float(normal_span) * float(cfg.center_surface_band_ratio)
    band = max(float(band), 1.0e-9)
    surface_mask = signed_normal >= signed_normal.max() - band
    surface_vertices = vertices[surface_mask]
    if surface_vertices.shape[0] < int(cfg.num_rows) * int(cfg.num_cols):
        return None

    u = _cluster_axis_centers(surface_vertices[:, axis_u], int(cfg.num_rows), eps=float(cfg.eps))
    v = _cluster_axis_centers(surface_vertices[:, axis_v], int(cfg.num_cols), eps=float(cfg.eps))
    if u is None or v is None:
        return None

    uu, vv = torch.meshgrid(u, v, indexing="ij")
    centers = torch.zeros((int(cfg.num_rows) * int(cfg.num_cols), 3), dtype=vertices.dtype, device=vertices.device)
    centers[:, axis_u] = uu.reshape(-1)
    centers[:, axis_v] = vv.reshape(-1)
    centers[:, normal_axis] = surface_vertices[:, normal_axis].mean()
    return centers


def _cluster_axis_centers(values: torch.Tensor, count: int, *, eps: float) -> torch.Tensor | None:
    if count == 1:
        return values.mean().reshape(1)
    if values.numel() < count:
        return None

    lo = values.min()
    hi = values.max()
    span = hi - lo
    if float(span) <= eps:
        return None

    centers = torch.linspace(float(lo), float(hi), count, dtype=values.dtype, device=values.device)
    for _ in range(32):
        dist = (values.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        ids = torch.argmin(dist, dim=-1)
        next_centers = centers.clone()
        for i in range(count):
            mask = ids == i
            if not bool(mask.any()):
                return None
            next_centers[i] = values[mask].mean()
        if bool((next_centers - centers).abs().max() <= max(eps, float(span) * 1.0e-6)):
            centers = next_centers
            break
        centers = next_centers

    centers = torch.sort(centers).values
    if bool(torch.any((centers[1:] - centers[:-1]) <= max(eps, float(span) * 1.0e-3))):
        return None
    return centers

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
    bump: BumpHydroShearCfg = field(default_factory=BumpHydroShearCfg)
    projected_surface: ProjectedSurfacePointTrackerCfg = field(default_factory=ProjectedSurfacePointTrackerCfg)
    projection: SurfacePointForceProjectorCfg = field(default_factory=SurfacePointForceProjectorCfg)
    marker_projection: HydroShearMarkerReadoutCfg = field(default_factory=HydroShearMarkerReadoutCfg)
    output_mode: str = "force_grid"
    readout_ema_alpha: float = 1.0
    output_key: str = "tactile"


@dataclass
class HydroShearTactileBackendOutput:
    contact: SurfacePointContactState
    surface: SurfacePointHydroShearOutput | BumpHydroShearOutput
    bump_surface: BumpHydroShearOutput | None
    projected_surface: ProjectedSurfacePointOutput | None
    readout: TaxelForceReadout
    marker_readout: HydroShearMarkerReadout | None
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
        self.bump_tracker = BumpHydroShearTracker(self.cfg.bump)
        self.bump_centers_p = self._make_bump_centers_p()
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
        self.bump_tracker.reset()
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
        if bool(self.cfg.bump.enabled):
            return self._update_bump(contact, samples, patch_quat_e=patch_quat_e)

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
            bump_surface=None,
            projected_surface=projected_surface,
            readout=readout,
            marker_readout=marker_readout,
            observations=observations,
        )

    def _update_bump(
        self,
        contact: SurfacePointContactState,
        samples: ObjectSurfaceSamples,
        *,
        patch_quat_e: torch.Tensor | None,
    ) -> HydroShearTactileBackendOutput:
        if self.bump_centers_p is None:
            raise RuntimeError("Bump HydroShear requires bump centers")

        bump = self.bump_tracker.update(
            samples,
            contact,
            centers_p=self.bump_centers_p.to(device=contact.points_p.device, dtype=contact.points_p.dtype),
            patch_quat_e=patch_quat_e,
        )
        readout = self._bump_readout(bump, patch_quat_e=patch_quat_e)
        readout = self._smooth_readout(readout)
        marker_readout = self._empty_marker_readout(readout)

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
            surface=bump,
            bump_surface=bump,
            projected_surface=None,
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
            if bool(self.cfg.bump.enabled):
                return readout.tactile_force
            return marker_readout.marker_field
        if self.cfg.output_mode == "bump_force_grid":
            return readout.tactile_force
        raise ValueError("output_mode must be 'force_grid', 'marker_field', or 'bump_force_grid'")

    def _make_bump_centers_p(self) -> torch.Tensor | None:
        if not bool(self.cfg.bump.enabled):
            return None
        if self.cfg.bump.centers_p is not None:
            centers = torch.as_tensor(self.cfg.bump.centers_p, dtype=torch.float32)
            expected = int(self.cfg.bump.num_rows) * int(self.cfg.bump.num_cols)
            if centers.shape != (expected, 3):
                raise ValueError(f"bump centers must have shape ({expected}, 3), got {tuple(centers.shape)}")
            return centers
        return create_bump_grid_centers(
            self.cfg.bump,
            self.cfg.grid,
            self.cfg.elastomer_vertices_p,
        )

    def _bump_readout(
        self,
        bump: BumpHydroShearOutput,
        *,
        patch_quat_e: torch.Tensor | None,
    ) -> TaxelForceReadout:
        axis_u, axis_v = tangential_axes(int(self.cfg.grid.normal_axis))
        shear_force_p = SurfacePointForceProjector._shear_force_to_patch(bump.shear_force_e, patch_quat_e)
        shear_uv = shear_force_p[:, (axis_u, axis_v)]
        signs = torch.as_tensor(self.cfg.projection.shear_axis_signs, dtype=shear_uv.dtype, device=shear_uv.device)
        shear_uv = shear_uv * float(self.cfg.projection.shear_scale) * signs
        normal = bump.normal_force * float(self.cfg.projection.normal_scale)
        tactile_force = torch.cat((normal.unsqueeze(-1), shear_uv), dim=-1)
        shape = (int(self.cfg.bump.num_rows), int(self.cfg.bump.num_cols))
        return TaxelForceReadout(
            taxel_positions_p=bump.centers_p,
            tactile_force=tactile_force.reshape(shape + (3,)),
            normal_force=normal.reshape(shape),
            shear_force_uv=shear_uv.reshape(shape + (2,)),
            weight_sum=bump.contact_mask.to(dtype=normal.dtype).reshape(shape),
        )

    @staticmethod
    def _empty_marker_readout(readout: TaxelForceReadout) -> HydroShearMarkerReadout:
        zeros = torch.zeros_like(readout.tactile_force)
        return HydroShearMarkerReadout(
            taxel_positions_p=readout.taxel_positions_p,
            marker_field=zeros,
            dilation_field=zeros,
            shear_field=zeros,
            weight_sum=readout.weight_sum,
        )

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
