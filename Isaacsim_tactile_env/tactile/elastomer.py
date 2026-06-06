from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ElastomerSdfResult:
    points_p: torch.Tensor
    sdf: torch.Tensor
    closest_points_p: torch.Tensor | None = None
    normals_p: torch.Tensor | None = None


@dataclass
class FlatPatchElastomerSdfCfg:
    normal_axis: int = 2
    surface_offset: float = 0.0
    half_extent_u: float | None = None
    half_extent_v: float | None = None
    eps: float = 1.0e-12


class FlatPatchElastomerSdf:
    """Flat elastomer SDF in patch frame.

    `phi < 0` means the object surface point has penetrated the elastomer.
    The flat surface is `points_p[..., normal_axis] == surface_offset`.
    """

    def __init__(self, cfg: FlatPatchElastomerSdfCfg | None = None):
        self.cfg = cfg or FlatPatchElastomerSdfCfg()
        if self.cfg.normal_axis not in (0, 1, 2):
            raise ValueError("normal_axis must be 0, 1, or 2")

    def evaluate(
        self,
        points_e: torch.Tensor,
        *,
        patch_pos_e: torch.Tensor | None = None,
        patch_quat_e: torch.Tensor | None = None,
    ) -> ElastomerSdfResult:
        points_p = points_e
        if patch_pos_e is not None or patch_quat_e is not None:
            if patch_pos_e is None or patch_quat_e is None:
                raise ValueError("patch_pos_e and patch_quat_e must be provided together")
            points_p = inverse_transform_points(points_e, patch_pos_e, patch_quat_e)

        sdf = points_p[..., self.cfg.normal_axis] - float(self.cfg.surface_offset)
        if self.cfg.half_extent_u is not None and self.cfg.half_extent_v is not None:
            axis_u, axis_v = _tangential_axes(int(self.cfg.normal_axis))
            outside_u = points_p[..., axis_u].abs() - float(self.cfg.half_extent_u)
            outside_v = points_p[..., axis_v].abs() - float(self.cfg.half_extent_v)
            outside = torch.maximum(outside_u, outside_v)
            sdf = torch.where(outside > 0.0, outside.clamp_min(float(self.cfg.eps)), sdf)
        normals_p = torch.zeros_like(points_p)
        normals_p[..., self.cfg.normal_axis] = 1.0
        return ElastomerSdfResult(points_p=points_p, sdf=sdf, normals_p=normals_p)


class MeshPatchElastomerSdf:
    """Elastomer mesh SDF in patch frame."""

    def __init__(
        self,
        *,
        vertices_p: torch.Tensor,
        faces: torch.Tensor,
        chunk_size: int | None = 2048,
    ):
        self.vertices_p = torch.as_tensor(vertices_p, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, dtype=torch.long)
        self.chunk_size = chunk_size

    def evaluate(
        self,
        points_e: torch.Tensor,
        *,
        patch_pos_e: torch.Tensor | None = None,
        patch_quat_e: torch.Tensor | None = None,
    ) -> ElastomerSdfResult:
        points_p = points_e
        if patch_pos_e is not None or patch_quat_e is not None:
            if patch_pos_e is None or patch_quat_e is None:
                raise ValueError("patch_pos_e and patch_quat_e must be provided together")
            points_p = inverse_transform_points(points_e, patch_pos_e, patch_quat_e)

        from .surface import signed_distance_to_mesh

        vertices_p = self.vertices_p.to(device=points_p.device, dtype=points_p.dtype)
        faces = self.faces.to(device=points_p.device)
        out = signed_distance_to_mesh(points_p, vertices_p, faces, chunk_size=self.chunk_size)
        return ElastomerSdfResult(
            points_p=points_p,
            sdf=out.sdf,
            closest_points_p=out.closest_points_o,
            normals_p=out.closest_normals_o,
        )


def normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    q_xyz = quat[..., 1:]
    q_w = quat[..., :1]
    uv = torch.cross(q_xyz, vec, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def rotate_vectors(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    quat = _expand_quat_for_points(quat, vec)
    return quat_apply_wxyz(quat, vec)


def transform_points(points_b: torch.Tensor, pos_a: torch.Tensor, quat_ab: torch.Tensor) -> torch.Tensor:
    quat_ab = _expand_quat_for_points(quat_ab, points_b)
    pos_a = _expand_pos_for_points(pos_a, points_b)
    return quat_apply_wxyz(quat_ab, points_b) + pos_a


def inverse_transform_points(points_a: torch.Tensor, pos_a: torch.Tensor, quat_ab: torch.Tensor) -> torch.Tensor:
    quat_ab = _expand_quat_for_points(quat_ab, points_a)
    pos_a = _expand_pos_for_points(pos_a, points_a)
    return quat_apply_wxyz(quat_conjugate_wxyz(quat_ab), points_a - pos_a)


def _expand_quat_for_points(quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    quat = torch.as_tensor(quat, dtype=points.dtype, device=points.device)
    while quat.ndim < points.ndim:
        quat = quat.unsqueeze(-2)
    return quat.expand(points.shape[:-1] + (4,))


def _expand_pos_for_points(pos: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    pos = torch.as_tensor(pos, dtype=points.dtype, device=points.device)
    while pos.ndim < points.ndim:
        pos = pos.unsqueeze(-2)
    return pos.expand(points.shape)


def _tangential_axes(normal_axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    return axes[0], axes[1]
