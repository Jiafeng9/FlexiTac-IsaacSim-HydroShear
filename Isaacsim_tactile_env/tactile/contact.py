from __future__ import annotations

from dataclasses import dataclass

import torch

from .elastomer import FlatPatchElastomerSdf, MeshPatchElastomerSdf, rotate_vectors, transform_points
from .surface import ObjectSurfaceSamples


@dataclass
class SurfacePointContactState:
    points_e: torch.Tensor
    normals_e: torch.Tensor
    points_p: torch.Tensor
    sdf: torch.Tensor
    contact_mask: torch.Tensor
    penetration: torch.Tensor
    elastomer_normals_p: torch.Tensor | None = None
    elastomer_normals_e: torch.Tensor | None = None


class SurfacePointContactQuery:
    """Query object surface samples against an elastomer SDF.

    Inputs are object-local samples and the current object pose expressed in
    elastomer frame E. Optional patch pose is expressed as patch frame P in E.
    """

    def __init__(self, elastomer_sdf: FlatPatchElastomerSdf | MeshPatchElastomerSdf | None = None):
        self.elastomer_sdf = elastomer_sdf or FlatPatchElastomerSdf()

    def compute(
        self,
        samples: ObjectSurfaceSamples,
        *,
        object_pos_e,
        object_quat_e,
        patch_pos_e=None,
        patch_quat_e=None,
    ) -> SurfacePointContactState:
        points_o = samples.points_o
        normals_o = samples.normals_o

        object_pos_e = torch.as_tensor(object_pos_e, dtype=points_o.dtype, device=points_o.device)
        object_quat_e = torch.as_tensor(object_quat_e, dtype=points_o.dtype, device=points_o.device)
        points_e = transform_points(points_o, object_pos_e, object_quat_e)
        normals_e = rotate_vectors(normals_o, object_quat_e)
        normals_e = normals_e / normals_e.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)

        if patch_pos_e is not None:
            patch_pos_e = torch.as_tensor(patch_pos_e, dtype=points_o.dtype, device=points_o.device)
        if patch_quat_e is not None:
            patch_quat_e = torch.as_tensor(patch_quat_e, dtype=points_o.dtype, device=points_o.device)

        sdf_result = self.elastomer_sdf.evaluate(points_e, patch_pos_e=patch_pos_e, patch_quat_e=patch_quat_e)
        penetration = (-sdf_result.sdf).clamp_min(0.0)
        contact_mask = sdf_result.sdf < 0.0
        elastomer_normals_p = sdf_result.normals_p
        elastomer_normals_e = None
        if elastomer_normals_p is not None:
            if patch_quat_e is not None:
                elastomer_normals_e = rotate_vectors(elastomer_normals_p, patch_quat_e)
            else:
                elastomer_normals_e = elastomer_normals_p
            elastomer_normals_e = elastomer_normals_e / elastomer_normals_e.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-12)

        return SurfacePointContactState(
            points_e=points_e,
            normals_e=normals_e,
            points_p=sdf_result.points_p,
            sdf=sdf_result.sdf,
            contact_mask=contact_mask,
            penetration=penetration,
            elastomer_normals_p=elastomer_normals_p,
            elastomer_normals_e=elastomer_normals_e,
        )
