from __future__ import annotations

from dataclasses import dataclass

import torch

from .contact import SurfacePointContactState
from .surface import ObjectSurfaceSamples


@dataclass
class SurfacePointHydroShearCfg:
    normal_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    friction_coefficient: float = 0.5
    normal_axis: int = 2
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
        d_n_scalar = d_contact[..., normal_axis]
        d_tangent = d_contact.clone()
        d_tangent[..., normal_axis] = 0.0

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
        normal_displacement_e = torch.zeros_like(contact.points_e)
        normal_displacement_e[..., normal_axis] = normal_displacement
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
    """Fraction of the frame-to-frame segment that lies inside `phi < 0`.

    This is the robust piecewise version of the contact-internal displacement
    fraction: outside->inside and inside->outside use linear interpolation,
    inside->inside is 1, outside->outside is 0.
    """

    prev_inside = prev_sdf < 0.0
    curr_inside = curr_sdf < 0.0

    alpha = torch.zeros_like(curr_sdf)
    alpha = torch.where(prev_inside & curr_inside, torch.ones_like(alpha), alpha)

    entering = (~prev_inside) & curr_inside
    entering_alpha = (-curr_sdf) / (prev_sdf - curr_sdf + eps)
    alpha = torch.where(entering, entering_alpha, alpha)

    exiting = prev_inside & (~curr_inside)
    exiting_alpha = (-prev_sdf) / (curr_sdf - prev_sdf + eps)
    alpha = torch.where(exiting, exiting_alpha, alpha)

    return alpha.clamp(0.0, 1.0)
