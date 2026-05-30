from __future__ import annotations

from dataclasses import dataclass

import torch

from .elastomer import quat_apply_wxyz, quat_conjugate_wxyz


@dataclass
class TaxelShearTrackerCfg:
    """Per-taxel shear tracker driven by target motion in the elastomer frame."""

    shear_stiffness: float = 500.0
    friction_coefficient: float = 0.5
    shear_decay: float = 0.0
    reset_on_contact_loss: bool = True
    force_sign: float = 1.0
    eps: float = 1.0e-8


@dataclass
class TaxelShearState:
    prev_object_pos_e: torch.Tensor
    prev_object_quat_e: torch.Tensor
    has_prev_pose: torch.Tensor
    prev_contact_mask: torch.Tensor
    shear_force_uv: torch.Tensor


@dataclass
class TaxelShearOutput:
    shear_force_uv: torch.Tensor
    slip_ratio: torch.Tensor
    contact_mask: torch.Tensor
    state: TaxelShearState


class TaxelShearTracker:
    """HydroShear-style Coulomb shear state attached to fixed taxels."""

    def __init__(self, cfg: TaxelShearTrackerCfg | None = None):
        self.cfg = cfg or TaxelShearTrackerCfg()
        self.state: TaxelShearState | None = None

    def reset(self):
        self.state = None

    def update(
        self,
        *,
        taxel_positions_e: torch.Tensor,
        object_pos_e: torch.Tensor,
        object_quat_e: torch.Tensor,
        normal_force: torch.Tensor,
        contact_mask: torch.Tensor,
        patch_quat_e: torch.Tensor | None = None,
        normal_axis: int = 0,
    ) -> TaxelShearOutput:
        object_pos_e = _as_batched_pose(object_pos_e, width=3)
        object_quat_e = _as_batched_pose(object_quat_e, width=4)
        normal_force = _as_batched_taxel_scalar(normal_force, object_pos_e.shape[0])
        contact_mask = _as_batched_taxel_scalar(contact_mask, object_pos_e.shape[0]).bool()

        if self.state is None or self.state.shear_force_uv.shape[:-1] != normal_force.shape:
            self.state = self._new_state(object_pos_e, object_quat_e, normal_force, contact_mask)

        prev = self.state
        delta_tangent = compute_per_taxel_delta_tangent(
            taxel_positions_e=taxel_positions_e,
            object_pos_e_curr=object_pos_e,
            object_quat_e_curr=object_quat_e,
            object_pos_e_prev=prev.prev_object_pos_e,
            object_quat_e_prev=prev.prev_object_quat_e,
            patch_quat_e=patch_quat_e,
            normal_axis=normal_axis,
            has_prev_pose=prev.has_prev_pose,
        )

        new_contact = contact_mask & ~prev.prev_contact_mask
        delta_tangent = torch.where(new_contact.unsqueeze(-1), torch.zeros_like(delta_tangent), delta_tangent)
        shear_prev = torch.where(new_contact.unsqueeze(-1), torch.zeros_like(prev.shear_force_uv), prev.shear_force_uv)
        shear_force_uv, slip_ratio = update_taxel_shear_force(
            shear_prev,
            delta_tangent,
            normal_force,
            contact_mask,
            shear_stiffness=float(self.cfg.shear_stiffness),
            friction_coefficient=float(self.cfg.friction_coefficient),
            shear_decay=float(self.cfg.shear_decay),
            reset_on_contact_loss=bool(self.cfg.reset_on_contact_loss),
            eps=float(self.cfg.eps),
            force_sign=float(self.cfg.force_sign),
        )

        self.state = TaxelShearState(
            prev_object_pos_e=object_pos_e.detach().clone(),
            prev_object_quat_e=object_quat_e.detach().clone(),
            has_prev_pose=torch.ones_like(prev.has_prev_pose, dtype=torch.bool),
            prev_contact_mask=contact_mask.detach().clone(),
            shear_force_uv=shear_force_uv.detach().clone(),
        )
        return TaxelShearOutput(
            shear_force_uv=shear_force_uv,
            slip_ratio=slip_ratio,
            contact_mask=contact_mask,
            state=self.state,
        )

    @staticmethod
    def _new_state(
        object_pos_e: torch.Tensor,
        object_quat_e: torch.Tensor,
        normal_force: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> TaxelShearState:
        return TaxelShearState(
            prev_object_pos_e=object_pos_e.detach().clone(),
            prev_object_quat_e=object_quat_e.detach().clone(),
            has_prev_pose=torch.zeros(object_pos_e.shape[0], device=object_pos_e.device, dtype=torch.bool),
            prev_contact_mask=torch.zeros_like(contact_mask, dtype=torch.bool),
            shear_force_uv=torch.zeros(normal_force.shape + (2,), device=normal_force.device, dtype=normal_force.dtype),
        )


def compute_per_taxel_delta_tangent(
    *,
    taxel_positions_e: torch.Tensor,
    object_pos_e_curr: torch.Tensor,
    object_quat_e_curr: torch.Tensor,
    object_pos_e_prev: torch.Tensor,
    object_quat_e_prev: torch.Tensor,
    patch_quat_e: torch.Tensor | None = None,
    normal_axis: int = 0,
    has_prev_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate target-induced taxel motion and return tangent components in patch frame."""

    if normal_axis not in (0, 1, 2):
        raise ValueError(f"normal_axis must be 0, 1, or 2, got {normal_axis}")
    if taxel_positions_e.shape[-1] != 3:
        raise ValueError(f"taxel_positions_e must end in 3 coordinates, got {taxel_positions_e.shape}")

    object_pos_e_curr = _as_batched_pose(object_pos_e_curr, width=3)
    object_quat_e_curr = _as_batched_pose(object_quat_e_curr, width=4)
    object_pos_e_prev = _as_batched_pose(object_pos_e_prev, width=3)
    object_quat_e_prev = _as_batched_pose(object_quat_e_prev, width=4)
    num_envs = int(object_pos_e_curr.shape[0])

    if taxel_positions_e.ndim == 2:
        points_e = taxel_positions_e.unsqueeze(0).expand(num_envs, -1, -1)
    elif taxel_positions_e.ndim == 3:
        if taxel_positions_e.shape[0] != num_envs:
            raise ValueError(f"taxel env dimension mismatch: {taxel_positions_e.shape[0]} vs {num_envs}")
        points_e = taxel_positions_e
    else:
        raise ValueError(f"taxel_positions_e must have shape (P,3) or (E,P,3), got {taxel_positions_e.shape}")

    if has_prev_pose is not None:
        has_prev_pose = has_prev_pose.to(device=points_e.device, dtype=torch.bool).reshape(num_envs)
        object_pos_e_prev = torch.where(has_prev_pose.unsqueeze(-1), object_pos_e_prev, object_pos_e_curr)
        object_quat_e_prev = torch.where(has_prev_pose.unsqueeze(-1), object_quat_e_prev, object_quat_e_curr)

    object_pos_prev = object_pos_e_prev.unsqueeze(1)
    object_quat_prev = object_quat_e_prev.unsqueeze(1).expand(-1, points_e.shape[1], -1)
    object_pos_curr = object_pos_e_curr.unsqueeze(1)
    object_quat_curr = object_quat_e_curr.unsqueeze(1).expand(-1, points_e.shape[1], -1)

    points_object_prev = quat_apply_wxyz(quat_conjugate_wxyz(object_quat_prev), points_e - object_pos_prev)
    points_moved_e = object_pos_curr + quat_apply_wxyz(object_quat_curr, points_object_prev)
    delta_e = points_moved_e - points_e

    if patch_quat_e is not None:
        patch_quat = torch.as_tensor(patch_quat_e, device=points_e.device, dtype=points_e.dtype)
        if patch_quat.ndim == 1:
            patch_quat = patch_quat.view(1, 1, 4).expand(num_envs, points_e.shape[1], -1)
        elif patch_quat.ndim == 2:
            patch_quat = patch_quat.unsqueeze(1).expand(-1, points_e.shape[1], -1)
        else:
            raise ValueError(f"patch_quat_e must have shape (4,) or (E,4), got {patch_quat.shape}")
        delta_e = quat_apply_wxyz(quat_conjugate_wxyz(patch_quat), delta_e)

    tangent_axes = [0, 1, 2]
    tangent_axes.remove(int(normal_axis))
    delta_tangent = delta_e[..., tangent_axes]
    if has_prev_pose is not None:
        delta_tangent = torch.where(has_prev_pose.view(num_envs, 1, 1), delta_tangent, torch.zeros_like(delta_tangent))
    return delta_tangent


def update_taxel_shear_force(
    shear_prev: torch.Tensor,
    delta_tangent: torch.Tensor,
    normal_force: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    shear_stiffness: float,
    friction_coefficient: float,
    shear_decay: float = 0.0,
    reset_on_contact_loss: bool = True,
    eps: float = 1.0e-8,
    force_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shear_prev.shape != delta_tangent.shape:
        raise ValueError(f"shear_prev and delta_tangent shape mismatch: {shear_prev.shape} vs {delta_tangent.shape}")
    if shear_prev.shape[-1] != 2:
        raise ValueError(f"Expected 2D tangential shear vectors, got shape {shear_prev.shape}")
    if shear_prev.shape[:-1] != normal_force.shape:
        raise ValueError(f"shear and normal force shape mismatch: {shear_prev.shape[:-1]} vs {normal_force.shape}")

    candidate = shear_prev + float(force_sign) * float(shear_stiffness) * delta_tangent
    limit = (float(friction_coefficient) * normal_force.clamp_min(0.0)).unsqueeze(-1)
    norm = torch.linalg.norm(candidate, dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), limit / (norm + float(eps)))
    shear = candidate * scale

    if reset_on_contact_loss:
        no_contact_shear = torch.zeros_like(shear)
    else:
        no_contact_shear = float(shear_decay) * shear_prev
    shear = torch.where(contact_mask.unsqueeze(-1), shear, no_contact_shear)

    shear_norm = torch.linalg.norm(shear, dim=-1)
    slip_ratio = shear_norm / (float(friction_coefficient) * normal_force.clamp_min(0.0) + float(eps))
    slip_ratio = torch.where(contact_mask, slip_ratio, torch.zeros_like(slip_ratio))
    return shear, slip_ratio


def _as_batched_pose(x: torch.Tensor, *, width: int) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else None)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[-1] != width:
        raise ValueError(f"expected shape ({width},) or (E,{width}), got {tuple(x.shape)}")
    return x


def _as_batched_taxel_scalar(x: torch.Tensor, num_envs: int) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else None)
    if x.ndim == 1:
        x = x.unsqueeze(0).expand(num_envs, -1)
    if x.ndim != 2 or x.shape[0] != num_envs:
        raise ValueError(f"expected shape (P,) or ({num_envs},P), got {tuple(x.shape)}")
    return x
