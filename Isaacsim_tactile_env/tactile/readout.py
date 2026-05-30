from __future__ import annotations

from dataclasses import dataclass

import torch

from .elastomer import quat_apply_wxyz, quat_conjugate_wxyz


@dataclass
class TaxelGridCfg:
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002
    normal_axis: int = 2
    normal_offset: float = 0.0
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"


@dataclass
class SurfacePointForceProjectorCfg:
    lambda_s: float = 10_800.0
    normalize_weights: bool = False
    weight_by_penetration: bool | None = None
    normal_weight_by_penetration: bool = False
    shear_weight_by_penetration: bool = True
    use_3d_distance: bool = True
    normal_scale: float = 1.0
    shear_scale: float = 1.0
    shear_axis_signs: tuple[float, float] = (1.0, 1.0)
    chunk_size: int | None = None
    eps: float = 1.0e-12

    def __post_init__(self):
        if self.weight_by_penetration is not None:
            self.normal_weight_by_penetration = bool(self.weight_by_penetration)
            self.shear_weight_by_penetration = bool(self.weight_by_penetration)


@dataclass
class ProjectedSurfacePointTrackerCfg:
    lambda_d: float = 1.0
    decay: float = 0.0
    max_displacement: float | None = None
    include_normal_displacement: bool = True


@dataclass
class ProjectedSurfacePointState:
    displacement_p: torch.Tensor


@dataclass
class ProjectedSurfacePointOutput:
    projected_points_p: torch.Tensor
    displacement_p: torch.Tensor
    state: ProjectedSurfacePointState


@dataclass
class TaxelForceReadout:
    taxel_positions_p: torch.Tensor
    tactile_force: torch.Tensor
    normal_force: torch.Tensor
    shear_force_uv: torch.Tensor
    weight_sum: torch.Tensor

    @property
    def tactile(self) -> torch.Tensor:
        return self.normal_force

    @property
    def tactile_shear(self) -> torch.Tensor:
        return self.shear_force_uv


@dataclass
class HydroShearMarkerReadoutCfg:
    lambda_s: float = 10_800.0
    lambda_d: float = 20_000.0
    shear_weight_by_penetration: bool = False
    dilation_weight_by_penetration: bool = False
    normalize_shear_weights: bool = False
    normalize_dilation_weights: bool = False
    shear_scale: float = 1.0
    dilation_scale: float = 1.0
    shear_axis_signs: tuple[float, float] = (1.0, 1.0)
    use_3d_distance: bool = True
    chunk_size: int | None = None
    sdf_query_chunk_size: int | None = 64
    eps: float = 1.0e-12


@dataclass
class HydroShearMarkerReadout:
    taxel_positions_p: torch.Tensor
    marker_field: torch.Tensor
    dilation_field: torch.Tensor
    shear_field: torch.Tensor
    weight_sum: torch.Tensor

    @property
    def tactile(self) -> torch.Tensor:
        return self.marker_field


def tangential_axes(normal_axis: int) -> tuple[int, int]:
    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1, or 2")
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    return axes[0], axes[1]


def create_taxel_grid_points(cfg: TaxelGridCfg) -> torch.Tensor:
    if cfg.num_rows <= 0 or cfg.num_cols <= 0:
        raise ValueError("num_rows and num_cols must be positive")
    if cfg.point_distance <= 0.0:
        raise ValueError("point_distance must be positive")

    axis_u, axis_v = tangential_axes(int(cfg.normal_axis))
    u = torch.linspace(
        -cfg.point_distance * (cfg.num_rows + 1) / 2.0,
        +cfg.point_distance * (cfg.num_rows + 1) / 2.0,
        steps=cfg.num_rows + 2,
        device=cfg.device,
        dtype=cfg.dtype,
    )[1:-1]
    v = torch.linspace(
        -cfg.point_distance * (cfg.num_cols + 1) / 2.0,
        +cfg.point_distance * (cfg.num_cols + 1) / 2.0,
        steps=cfg.num_cols + 2,
        device=cfg.device,
        dtype=cfg.dtype,
    )[1:-1]

    uu, vv = torch.meshgrid(u, v, indexing="ij")
    points = torch.zeros((cfg.num_rows * cfg.num_cols, 3), device=cfg.device, dtype=cfg.dtype)
    points[:, axis_u] = uu.reshape(-1)
    points[:, axis_v] = vv.reshape(-1)
    points[:, int(cfg.normal_axis)] = float(cfg.normal_offset)
    return points


class SurfacePointForceProjector:
    """Project HydroShear surface-point state to a taxel grid."""

    def __init__(
        self,
        grid_cfg: TaxelGridCfg,
        cfg: SurfacePointForceProjectorCfg | None = None,
        *,
        taxel_positions_p: torch.Tensor | None = None,
    ):
        self.grid_cfg = grid_cfg
        self.cfg = cfg or SurfacePointForceProjectorCfg()
        self.taxel_positions_p = taxel_positions_p if taxel_positions_p is not None else create_taxel_grid_points(grid_cfg)

    def project(
        self,
        *,
        surface_points_p: torch.Tensor,
        penetration: torch.Tensor,
        normal_force: torch.Tensor,
        shear_force_e: torch.Tensor,
        patch_quat_e: torch.Tensor | None = None,
        projected_surface_points_p: torch.Tensor | None = None,
    ) -> TaxelForceReadout:
        """Return `[fn, ftu, ftv]` per taxel.

        `surface_points_p` are the current object surface points in patch frame.
        `projected_surface_points_p` is the future hook for HydroShear's
        projected displacement point `o_hat`; when omitted, current points are
        used directly.
        """

        source_points_p = projected_surface_points_p if projected_surface_points_p is not None else surface_points_p
        taxel_positions_p = self.taxel_positions_p.to(device=source_points_p.device, dtype=source_points_p.dtype)

        penetration = penetration.to(device=source_points_p.device, dtype=source_points_p.dtype)
        normal_force = normal_force.to(device=source_points_p.device, dtype=source_points_p.dtype)
        shear_force_e = shear_force_e.to(device=source_points_p.device, dtype=source_points_p.dtype)

        if source_points_p.ndim != 2 or source_points_p.shape[-1] != 3:
            raise ValueError("surface points must have shape (num_points, 3)")
        if penetration.shape != normal_force.shape or penetration.shape != source_points_p.shape[:1]:
            raise ValueError("penetration and normal_force must have shape (num_points,)")
        if shear_force_e.shape != source_points_p.shape:
            raise ValueError("shear_force_e must have shape (num_points, 3)")

        axis_u, axis_v = tangential_axes(int(self.grid_cfg.normal_axis))
        taxel_uv = taxel_positions_p[:, (axis_u, axis_v)]
        source_uv = source_points_p[:, (axis_u, axis_v)]
        weight_taxel_points = taxel_positions_p if self.cfg.use_3d_distance else taxel_uv
        weight_source_points = source_points_p if self.cfg.use_3d_distance else source_uv
        shear_force_p = self._shear_force_to_patch(shear_force_e, patch_quat_e)
        shear_uv = shear_force_p[:, (axis_u, axis_v)]
        normal_grid, shear_grid_uv, weight_sum = self._project_weighted_forces(
            taxel_points=weight_taxel_points,
            source_points=weight_source_points,
            penetration=penetration,
            normal_force=normal_force,
            shear_uv=shear_uv,
        )

        normal_grid = normal_grid * float(self.cfg.normal_scale)
        signs = torch.as_tensor(self.cfg.shear_axis_signs, dtype=shear_grid_uv.dtype, device=shear_grid_uv.device)
        shear_grid_uv = shear_grid_uv * float(self.cfg.shear_scale) * signs

        tactile_force = torch.cat((normal_grid.unsqueeze(-1), shear_grid_uv), dim=-1)
        shape = (self.grid_cfg.num_rows, self.grid_cfg.num_cols)
        return TaxelForceReadout(
            taxel_positions_p=taxel_positions_p,
            tactile_force=tactile_force.reshape(shape + (3,)),
            normal_force=normal_grid.reshape(shape),
            shear_force_uv=shear_grid_uv.reshape(shape + (2,)),
            weight_sum=weight_sum.reshape(shape),
        )

    def _project_weighted_forces(
        self,
        *,
        taxel_points: torch.Tensor,
        source_points: torch.Tensor,
        penetration: torch.Tensor,
        normal_force: torch.Tensor,
        shear_uv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = self.cfg.chunk_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= taxel_points.shape[0]:
            normal_weights = self._weights(
                taxel_points,
                source_points,
                penetration,
                weight_by_penetration=bool(self.cfg.normal_weight_by_penetration),
            )
            shear_weights = self._weights(
                taxel_points,
                source_points,
                penetration,
                weight_by_penetration=bool(self.cfg.shear_weight_by_penetration),
            )
            return normal_weights @ normal_force, shear_weights @ shear_uv, normal_weights.sum(dim=-1)

        normal_parts = []
        shear_parts = []
        weight_parts = []
        for start in range(0, taxel_points.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), taxel_points.shape[0])
            normal_weights = self._weights(
                taxel_points[start:end],
                source_points,
                penetration,
                weight_by_penetration=bool(self.cfg.normal_weight_by_penetration),
            )
            shear_weights = self._weights(
                taxel_points[start:end],
                source_points,
                penetration,
                weight_by_penetration=bool(self.cfg.shear_weight_by_penetration),
            )
            normal_parts.append(normal_weights @ normal_force)
            shear_parts.append(shear_weights @ shear_uv)
            weight_parts.append(normal_weights.sum(dim=-1))
        return torch.cat(normal_parts, dim=0), torch.cat(shear_parts, dim=0), torch.cat(weight_parts, dim=0)

    def _weights(
        self,
        taxel_points: torch.Tensor,
        source_points: torch.Tensor,
        penetration: torch.Tensor,
        *,
        weight_by_penetration: bool,
    ) -> torch.Tensor:
        diff = taxel_points[:, None, :] - source_points[None, :, :]
        dist2 = (diff * diff).sum(dim=-1)
        weights = torch.exp(-float(self.cfg.lambda_s) * dist2)
        if weight_by_penetration:
            weights = penetration.unsqueeze(0) * weights
        if self.cfg.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(self.cfg.eps))
        return weights

    @staticmethod
    def _shear_force_to_patch(shear_force_e: torch.Tensor, patch_quat_e: torch.Tensor | None) -> torch.Tensor:
        if patch_quat_e is None:
            return shear_force_e

        quat = torch.as_tensor(patch_quat_e, dtype=shear_force_e.dtype, device=shear_force_e.device)
        while quat.ndim < shear_force_e.ndim:
            quat = quat.unsqueeze(-2)
        quat = quat.expand(shear_force_e.shape[:-1] + (4,))
        return quat_apply_wxyz(quat_conjugate_wxyz(quat), shear_force_e)


class HydroShearMarkerProjector:
    """Project HydroShear displacement state to marker displacement channels.

    The output channel order is `[normal, tangent_u, tangent_v]`. When
    `marker_object_sdf` is supplied, dilation follows the official HydroShear
    form: marker points query the object SDF, `height = relu(-sdf)`, then
    `height * dvec * exp(-lambda_d * ||dvec||^2)` is summed over markers.
    """

    def __init__(
        self,
        grid_cfg: TaxelGridCfg,
        cfg: HydroShearMarkerReadoutCfg | None = None,
        *,
        taxel_positions_p: torch.Tensor | None = None,
    ):
        self.grid_cfg = grid_cfg
        self.cfg = cfg or HydroShearMarkerReadoutCfg()
        self.taxel_positions_p = taxel_positions_p if taxel_positions_p is not None else create_taxel_grid_points(grid_cfg)

    def project(
        self,
        *,
        surface_points_p: torch.Tensor,
        penetration: torch.Tensor,
        displacement_e: torch.Tensor,
        patch_quat_e: torch.Tensor | None = None,
        projected_surface_points_p: torch.Tensor | None = None,
        marker_object_sdf: torch.Tensor | None = None,
    ) -> HydroShearMarkerReadout:
        source_points_p = projected_surface_points_p if projected_surface_points_p is not None else surface_points_p
        taxel_positions_p = self.taxel_positions_p.to(device=surface_points_p.device, dtype=surface_points_p.dtype)
        surface_points_p = surface_points_p.to(device=taxel_positions_p.device, dtype=taxel_positions_p.dtype)
        source_points_p = source_points_p.to(device=taxel_positions_p.device, dtype=taxel_positions_p.dtype)
        penetration = penetration.to(device=taxel_positions_p.device, dtype=taxel_positions_p.dtype)
        displacement_e = displacement_e.to(device=taxel_positions_p.device, dtype=taxel_positions_p.dtype)

        if surface_points_p.ndim != 2 or surface_points_p.shape[-1] != 3:
            raise ValueError("surface_points_p must have shape (num_points, 3)")
        if source_points_p.shape != surface_points_p.shape:
            raise ValueError("projected_surface_points_p must match surface_points_p shape")
        if displacement_e.shape != surface_points_p.shape:
            raise ValueError("displacement_e must match surface_points_p shape")
        if penetration.shape != surface_points_p.shape[:1]:
            raise ValueError("penetration must have shape (num_points,)")
        if marker_object_sdf is not None:
            marker_object_sdf = torch.as_tensor(marker_object_sdf, dtype=taxel_positions_p.dtype, device=taxel_positions_p.device)
            if marker_object_sdf.shape != taxel_positions_p.shape[:1]:
                raise ValueError("marker_object_sdf must have shape (num_taxels,)")

        displacement_p = SurfacePointForceProjector._shear_force_to_patch(displacement_e, patch_quat_e)
        marker_vec, dilation_vec, shear_vec, weight_sum = self._project_vectors(
            taxel_positions_p=taxel_positions_p,
            surface_points_p=surface_points_p,
            source_points_p=source_points_p,
            penetration=penetration,
            displacement_p=displacement_p,
            marker_object_sdf=marker_object_sdf,
        )

        marker_field = self._vector_to_channels(marker_vec)
        dilation_field = self._vector_to_channels(dilation_vec)
        shear_field = self._vector_to_channels(shear_vec)
        shape = (self.grid_cfg.num_rows, self.grid_cfg.num_cols)
        return HydroShearMarkerReadout(
            taxel_positions_p=taxel_positions_p,
            marker_field=marker_field.reshape(shape + (3,)),
            dilation_field=dilation_field.reshape(shape + (3,)),
            shear_field=shear_field.reshape(shape + (3,)),
            weight_sum=weight_sum.reshape(shape),
        )

    def _project_vectors(
        self,
        *,
        taxel_positions_p: torch.Tensor,
        surface_points_p: torch.Tensor,
        source_points_p: torch.Tensor,
        penetration: torch.Tensor,
        displacement_p: torch.Tensor,
        marker_object_sdf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = self.cfg.chunk_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= taxel_positions_p.shape[0]:
            return self._project_vectors_chunk(
                taxel_positions_p=taxel_positions_p,
                surface_points_p=surface_points_p,
                source_points_p=source_points_p,
                penetration=penetration,
                displacement_p=displacement_p,
                all_taxel_positions_p=taxel_positions_p,
                marker_object_sdf=marker_object_sdf,
            )

        marker_parts = []
        dilation_parts = []
        shear_parts = []
        weight_parts = []
        for start in range(0, taxel_positions_p.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), taxel_positions_p.shape[0])
            marker, dilation, shear, weight_sum = self._project_vectors_chunk(
                taxel_positions_p=taxel_positions_p[start:end],
                surface_points_p=surface_points_p,
                source_points_p=source_points_p,
                penetration=penetration,
                displacement_p=displacement_p,
                all_taxel_positions_p=taxel_positions_p,
                marker_object_sdf=marker_object_sdf,
            )
            marker_parts.append(marker)
            dilation_parts.append(dilation)
            shear_parts.append(shear)
            weight_parts.append(weight_sum)
        return (
            torch.cat(marker_parts, dim=0),
            torch.cat(dilation_parts, dim=0),
            torch.cat(shear_parts, dim=0),
            torch.cat(weight_parts, dim=0),
        )

    def _project_vectors_chunk(
        self,
        *,
        taxel_positions_p: torch.Tensor,
        surface_points_p: torch.Tensor,
        source_points_p: torch.Tensor,
        penetration: torch.Tensor,
        displacement_p: torch.Tensor,
        all_taxel_positions_p: torch.Tensor,
        marker_object_sdf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shear_query, shear_source = self._distance_points(taxel_positions_p, source_points_p)
        shear_weights = self._gaussian_weights(
            shear_query,
            shear_source,
            penetration,
            lambda_value=float(self.cfg.lambda_s),
            weight_by_penetration=bool(self.cfg.shear_weight_by_penetration),
            normalize=bool(self.cfg.normalize_shear_weights),
        )
        normal_axis = int(self.grid_cfg.normal_axis)
        shear_height = displacement_p[..., normal_axis].clamp_min(0.0)
        shear_displacement_p = displacement_p.clone()
        shear_displacement_p[..., normal_axis] = 0.0
        shear_source_vec = -shear_height.unsqueeze(-1) * shear_displacement_p
        shear_vec = shear_weights @ shear_source_vec

        if marker_object_sdf is None:
            dilation_vec = self._sampled_surface_dilation(
                taxel_positions_p=taxel_positions_p,
                surface_points_p=surface_points_p,
                penetration=penetration,
            )
        else:
            dilation_vec = self._sdf_marker_dilation(
                taxel_positions_p=taxel_positions_p,
                all_taxel_positions_p=all_taxel_positions_p,
                marker_object_sdf=marker_object_sdf,
            )

        shear_vec = self._apply_shear_signs(shear_vec) * float(self.cfg.shear_scale)
        dilation_vec = dilation_vec * float(self.cfg.dilation_scale)
        return dilation_vec + shear_vec, dilation_vec, shear_vec, shear_weights.sum(dim=-1)

    def _sampled_surface_dilation(
        self,
        *,
        taxel_positions_p: torch.Tensor,
        surface_points_p: torch.Tensor,
        penetration: torch.Tensor,
    ) -> torch.Tensor:
        dilation_query, dilation_source = self._distance_points(taxel_positions_p, surface_points_p)
        dilation_weights = self._gaussian_weights(
            dilation_query,
            dilation_source,
            penetration,
            lambda_value=float(self.cfg.lambda_d),
            weight_by_penetration=bool(self.cfg.dilation_weight_by_penetration),
            normalize=bool(self.cfg.normalize_dilation_weights),
        )
        dvec = taxel_positions_p[:, None, :] - surface_points_p[None, :, :]
        return (dilation_weights.unsqueeze(-1) * dvec).sum(dim=1)

    def _sdf_marker_dilation(
        self,
        *,
        taxel_positions_p: torch.Tensor,
        all_taxel_positions_p: torch.Tensor,
        marker_object_sdf: torch.Tensor,
    ) -> torch.Tensor:
        query_points, source_points = self._distance_points(taxel_positions_p, all_taxel_positions_p)
        diff = query_points[:, None, :] - source_points[None, :, :]
        dist2 = (diff * diff).sum(dim=-1)
        weights = torch.exp(-float(self.cfg.lambda_d) * dist2)
        if self.cfg.normalize_dilation_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(self.cfg.eps))
        height = (-marker_object_sdf).clamp_min(0.0)
        dvec = taxel_positions_p[:, None, :] - all_taxel_positions_p[None, :, :]
        return (weights.unsqueeze(-1) * height.view(1, -1, 1) * dvec).sum(dim=1)

    def _distance_points(self, taxel_positions_p: torch.Tensor, source_points_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.use_3d_distance:
            return taxel_positions_p, source_points_p
        axis_u, axis_v = tangential_axes(int(self.grid_cfg.normal_axis))
        return taxel_positions_p[:, (axis_u, axis_v)], source_points_p[:, (axis_u, axis_v)]

    def _gaussian_weights(
        self,
        taxel_points: torch.Tensor,
        source_points: torch.Tensor,
        penetration: torch.Tensor,
        *,
        lambda_value: float,
        weight_by_penetration: bool,
        normalize: bool,
    ) -> torch.Tensor:
        diff = taxel_points[:, None, :] - source_points[None, :, :]
        dist2 = (diff * diff).sum(dim=-1)
        weights = torch.exp(-lambda_value * dist2)
        if weight_by_penetration:
            weights = penetration.unsqueeze(0) * weights
        if normalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(self.cfg.eps))
        return weights

    def _apply_shear_signs(self, shear_vec: torch.Tensor) -> torch.Tensor:
        axis_u, axis_v = tangential_axes(int(self.grid_cfg.normal_axis))
        signs = torch.as_tensor(self.cfg.shear_axis_signs, dtype=shear_vec.dtype, device=shear_vec.device)
        out = shear_vec.clone()
        out[..., axis_u] = out[..., axis_u] * signs[0]
        out[..., axis_v] = out[..., axis_v] * signs[1]
        return out

    def _vector_to_channels(self, vec_p: torch.Tensor) -> torch.Tensor:
        axis_u, axis_v = tangential_axes(int(self.grid_cfg.normal_axis))
        return torch.stack(
            (
                vec_p[..., int(self.grid_cfg.normal_axis)],
                vec_p[..., axis_u],
                vec_p[..., axis_v],
            ),
            dim=-1,
        )


class ProjectedSurfacePointTracker:
    """Track HydroShear's projected surface point `o_hat` in patch frame."""

    def __init__(self, normal_axis: int, cfg: ProjectedSurfacePointTrackerCfg | None = None):
        self.normal_axis = int(normal_axis)
        if self.normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2")
        self.cfg = cfg or ProjectedSurfacePointTrackerCfg()
        self.state: ProjectedSurfacePointState | None = None

    def reset(self):
        self.state = None

    def update(
        self,
        *,
        surface_points_p: torch.Tensor,
        displacement_e: torch.Tensor | None = None,
        force_e: torch.Tensor | None = None,
        contact_mask: torch.Tensor,
        patch_quat_e: torch.Tensor | None = None,
        shear_force_e: torch.Tensor | None = None,
    ) -> ProjectedSurfacePointOutput:
        if displacement_e is None:
            displacement_e = force_e
        if displacement_e is None:
            if shear_force_e is None:
                raise ValueError("displacement_e must be provided")
            displacement_e = shear_force_e
        displacement_e = torch.as_tensor(displacement_e, dtype=surface_points_p.dtype, device=surface_points_p.device)
        contact_mask = torch.as_tensor(contact_mask, dtype=torch.bool, device=surface_points_p.device)
        if displacement_e.shape != surface_points_p.shape:
            raise ValueError("displacement_e must have the same shape as surface_points_p")
        if contact_mask.shape != surface_points_p.shape[:1]:
            raise ValueError("contact_mask must have shape (num_points,)")
        displacement_p = SurfacePointForceProjector._shear_force_to_patch(displacement_e, patch_quat_e)
        displacement = float(self.cfg.lambda_d) * displacement_p
        if self.state is not None and self.state.displacement_p.shape == displacement.shape:
            displacement = displacement + float(self.cfg.decay) * self.state.displacement_p.to(
                device=displacement.device,
                dtype=displacement.dtype,
            )

        displacement = displacement.clone()
        if not bool(self.cfg.include_normal_displacement):
            displacement[..., self.normal_axis] = 0.0

        if self.cfg.max_displacement is not None and self.cfg.max_displacement > 0.0:
            norm = displacement.norm(dim=-1, keepdim=True)
            scale = torch.minimum(
                torch.ones_like(norm),
                torch.as_tensor(float(self.cfg.max_displacement), dtype=norm.dtype, device=norm.device)
                / norm.clamp_min(1.0e-12),
            )
            displacement = displacement * scale

        displacement = torch.where(contact_mask.unsqueeze(-1), displacement, torch.zeros_like(displacement))
        self.state = ProjectedSurfacePointState(displacement_p=displacement.detach().clone())
        return ProjectedSurfacePointOutput(
            projected_points_p=surface_points_p + displacement,
            displacement_p=displacement,
            state=self.state,
        )
