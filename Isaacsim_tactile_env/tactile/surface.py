from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ObjectSurfaceSamples:
    """Object-local surface samples for HydroShear-style contact tracking."""

    points_o: torch.Tensor
    """Sample positions in object frame, shape (M, 3)."""

    normals_o: torch.Tensor
    """Unit normals in object frame, shape (M, 3)."""

    area: torch.Tensor
    """Area weight for each sample, shape (M,)."""

    face_index: torch.Tensor | None = None
    """Source triangle index for debugging, shape (M,)."""

    vertices_o: torch.Tensor | None = None
    """Object-local mesh vertices used for SDF queries, shape (V, 3)."""

    faces: torch.Tensor | None = None
    """Triangle vertex indices used for SDF queries, shape (F, 3)."""

    @property
    def num_points(self) -> int:
        return int(self.points_o.shape[0])

    @property
    def total_area(self) -> torch.Tensor:
        return self.area.sum()

    def query_sdf(self, points_o: torch.Tensor, *, chunk_size: int | None = None) -> "ObjectMeshSdfResult":
        if self.vertices_o is None or self.faces is None:
            raise RuntimeError("ObjectSurfaceSamples does not include mesh vertices/faces for SDF queries")
        return signed_distance_to_mesh(
            points_o,
            self.vertices_o,
            self.faces,
            chunk_size=chunk_size,
        )


@dataclass
class ObjectMeshSdfResult:
    sdf: torch.Tensor
    unsigned_distance: torch.Tensor
    closest_points_o: torch.Tensor


@dataclass
class ObjectSurfaceSamplerCfg:
    num_points: int = 2048
    seed: int | None = 0
    smooth_normals: bool = True
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class ObjectSurfaceSampler:
    """Area-weighted sampler for object mesh surface points.

    The returned samples are in object-local mesh coordinates. With random area
    sampling, each sample represents an equal Monte Carlo area weight:
    `A_j = total_mesh_area / num_points`.
    """

    def __init__(self, cfg: ObjectSurfaceSamplerCfg | None = None):
        self.cfg = cfg or ObjectSurfaceSamplerCfg()

    def sample_mesh_file(
        self,
        mesh_path: str | os.PathLike,
        *,
        scale: tuple[float, float, float] | None = None,
    ) -> ObjectSurfaceSamples:
        mesh = load_trimesh(mesh_path, scale=scale)
        return self.sample_trimesh(mesh)

    def sample_urdf_visual_mesh(self, urdf_path: str | os.PathLike) -> ObjectSurfaceSamples:
        mesh_path, scale = first_visual_mesh_from_urdf(urdf_path)
        return self.sample_mesh_file(mesh_path, scale=scale)

    def sample_trimesh(self, mesh) -> ObjectSurfaceSamples:
        vertices, faces = _triangulated_vertices_faces(mesh)
        return self.sample_arrays(vertices, faces)

    def sample_arrays(self, vertices, faces) -> ObjectSurfaceSamples:
        vertices_np = np.asarray(vertices, dtype=np.float64)
        faces_np = np.asarray(faces, dtype=np.int64)

        if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3), got {vertices_np.shape}")
        if faces_np.ndim != 2 or faces_np.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), got {faces_np.shape}")
        if self.cfg.num_points <= 0:
            raise ValueError("num_points must be positive")

        tri = vertices_np[faces_np]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        cross = np.cross(e1, e2)
        double_area = np.linalg.norm(cross, axis=1)
        face_area = 0.5 * double_area
        valid = face_area > 1.0e-16
        if not np.any(valid):
            raise ValueError("mesh has no non-degenerate triangles")

        tri = tri[valid]
        faces_valid = faces_np[valid]
        face_area = face_area[valid]
        face_normals = cross[valid] / np.clip(double_area[valid, None], 1.0e-16, None)

        total_area = float(face_area.sum())
        probabilities = face_area / total_area
        rng = np.random.default_rng(self.cfg.seed)

        face_ids = rng.choice(face_area.shape[0], size=self.cfg.num_points, replace=True, p=probabilities)
        bary = _uniform_barycentric(rng, self.cfg.num_points)
        sampled_tri = tri[face_ids]
        points = (
            bary[:, 0:1] * sampled_tri[:, 0]
            + bary[:, 1:2] * sampled_tri[:, 1]
            + bary[:, 2:3] * sampled_tri[:, 2]
        )

        if self.cfg.smooth_normals:
            normals = self._sample_smooth_normals(vertices_np, faces_valid, face_ids, bary, face_normals)
        else:
            normals = face_normals[face_ids]

        normals = _normalize(normals)
        area = np.full((self.cfg.num_points,), total_area / float(self.cfg.num_points), dtype=np.float64)

        return ObjectSurfaceSamples(
            points_o=torch.as_tensor(points, dtype=self.cfg.dtype, device=self.cfg.device),
            normals_o=torch.as_tensor(normals, dtype=self.cfg.dtype, device=self.cfg.device),
            area=torch.as_tensor(area, dtype=self.cfg.dtype, device=self.cfg.device),
            face_index=torch.as_tensor(face_ids, dtype=torch.long, device=self.cfg.device),
            vertices_o=torch.as_tensor(vertices_np, dtype=self.cfg.dtype, device=self.cfg.device),
            faces=torch.as_tensor(faces_valid, dtype=torch.long, device=self.cfg.device),
        )

    @staticmethod
    def _sample_smooth_normals(vertices_np, faces_valid, face_ids, bary, fallback_face_normals):
        vertex_normals = np.zeros_like(vertices_np, dtype=np.float64)
        tri = vertices_np[faces_valid]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        np.add.at(vertex_normals, faces_valid[:, 0], cross)
        np.add.at(vertex_normals, faces_valid[:, 1], cross)
        np.add.at(vertex_normals, faces_valid[:, 2], cross)
        vertex_normals = _normalize(vertex_normals)

        sampled_faces = faces_valid[face_ids]
        n_tri = vertex_normals[sampled_faces]
        normals = (
            bary[:, 0:1] * n_tri[:, 0]
            + bary[:, 1:2] * n_tri[:, 1]
            + bary[:, 2:3] * n_tri[:, 2]
        )

        bad = np.linalg.norm(normals, axis=1) < 1.0e-12
        if np.any(bad):
            normals[bad] = fallback_face_normals[face_ids[bad]]
        return normals


def load_trimesh(mesh_path: str | os.PathLike, *, scale: tuple[float, float, float] | None = None):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {mesh_path}")
    if scale is not None:
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    return mesh


def signed_distance_to_mesh(
    points_o: torch.Tensor,
    vertices_o: torch.Tensor,
    faces: torch.Tensor,
    *,
    chunk_size: int | None = None,
    eps: float = 1.0e-12,
) -> ObjectMeshSdfResult:
    """Return signed distance from points to a closed triangle mesh.

    The sign uses a solid-angle winding test. This mirrors the official
    HydroShear path at the interface level: marker points are queried against
    the indenter/object SDF, and negative values mean the marker lies inside
    the object.
    """

    points_o = torch.as_tensor(points_o)
    vertices_o = torch.as_tensor(vertices_o, dtype=points_o.dtype, device=points_o.device)
    faces = torch.as_tensor(faces, dtype=torch.long, device=points_o.device)
    if points_o.ndim != 2 or points_o.shape[-1] != 3:
        raise ValueError("points_o must have shape (N, 3)")
    if vertices_o.ndim != 2 or vertices_o.shape[-1] != 3:
        raise ValueError("vertices_o must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape (F, 3)")

    triangles = vertices_o[faces]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= points_o.shape[0]:
        return _signed_distance_to_mesh_chunk(points_o, triangles, eps=eps)

    sdf_parts = []
    dist_parts = []
    closest_parts = []
    for start in range(0, points_o.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), points_o.shape[0])
        part = _signed_distance_to_mesh_chunk(points_o[start:end], triangles, eps=eps)
        sdf_parts.append(part.sdf)
        dist_parts.append(part.unsigned_distance)
        closest_parts.append(part.closest_points_o)
    return ObjectMeshSdfResult(
        sdf=torch.cat(sdf_parts, dim=0),
        unsigned_distance=torch.cat(dist_parts, dim=0),
        closest_points_o=torch.cat(closest_parts, dim=0),
    )


def _signed_distance_to_mesh_chunk(points_o: torch.Tensor, triangles: torch.Tensor, *, eps: float) -> ObjectMeshSdfResult:
    closest, sq_dist = _closest_points_on_triangles(points_o, triangles, eps=eps)
    min_sq_dist, min_face = sq_dist.min(dim=1)
    closest_points = closest[torch.arange(points_o.shape[0], device=points_o.device), min_face]
    unsigned = min_sq_dist.clamp_min(0.0).sqrt()
    inside = _points_inside_closed_mesh(points_o, triangles, eps=eps)
    sdf = torch.where(inside, -unsigned, unsigned)
    return ObjectMeshSdfResult(sdf=sdf, unsigned_distance=unsigned, closest_points_o=closest_points)


def _closest_points_on_triangles(points: torch.Tensor, triangles: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)

    bp = p - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)

    cp = p - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    denom = (va + vb + vc).clamp_min(float(eps))
    v = vb / denom
    w = vc / denom
    closest = a + ab * v.unsqueeze(-1) + ac * w.unsqueeze(-1)

    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    mask_c = (d6 >= 0.0) & (d5 <= d6)

    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v_ab = d1 / (d1 - d3).clamp_min(float(eps))
    closest_ab = a + v_ab.unsqueeze(-1) * ab

    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w_ac = d2 / (d2 - d6).clamp_min(float(eps))
    closest_ac = a + w_ac.unsqueeze(-1) * ac

    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(float(eps))
    closest_bc = b + w_bc.unsqueeze(-1) * (c - b)

    closest = torch.where(mask_ab.unsqueeze(-1), closest_ab, closest)
    closest = torch.where(mask_ac.unsqueeze(-1), closest_ac, closest)
    closest = torch.where(mask_bc.unsqueeze(-1), closest_bc, closest)
    closest = torch.where(mask_a.unsqueeze(-1), a.expand_as(closest), closest)
    closest = torch.where(mask_b.unsqueeze(-1), b.expand_as(closest), closest)
    closest = torch.where(mask_c.unsqueeze(-1), c.expand_as(closest), closest)

    diff = p - closest
    return closest, (diff * diff).sum(dim=-1)


def _points_inside_closed_mesh(points: torch.Tensor, triangles: torch.Tensor, *, eps: float) -> torch.Tensor:
    a = triangles[None, :, 0, :] - points[:, None, :]
    b = triangles[None, :, 1, :] - points[:, None, :]
    c = triangles[None, :, 2, :] - points[:, None, :]

    la = a.norm(dim=-1)
    lb = b.norm(dim=-1)
    lc = c.norm(dim=-1)
    numerator = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    denominator = (
        la * lb * lc
        + (a * b).sum(dim=-1) * lc
        + (b * c).sum(dim=-1) * la
        + (c * a).sum(dim=-1) * lb
    )
    den_sign = torch.where(denominator < 0.0, -torch.ones_like(denominator), torch.ones_like(denominator))
    denominator = torch.where(denominator.abs() < float(eps), den_sign * float(eps), denominator)
    omega = 2.0 * torch.atan2(numerator, denominator)
    winding = omega.sum(dim=-1).abs()
    return winding > (2.0 * torch.pi)


def first_visual_mesh_from_urdf(urdf_path: str | os.PathLike) -> tuple[Path, tuple[float, float, float]]:
    urdf_path = Path(os.path.expanduser(str(urdf_path))).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)

    root = ET.parse(urdf_path).getroot()
    for visual in root.iter("visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_el = geom.find("mesh")
        if mesh_el is None:
            continue

        filename = mesh_el.get("filename", "")
        scale = _parse_scale(mesh_el.get("scale", "1 1 1"))
        mesh_path = (urdf_path.parent / filename).resolve()
        if mesh_path.is_file():
            return mesh_path, scale

    raise RuntimeError(f"No visual mesh found in URDF: {urdf_path}")


def _triangulated_vertices_faces(mesh) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(mesh, "geometry") and not hasattr(mesh, "faces"):
        import trimesh

        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def _uniform_barycentric(rng: np.random.Generator, count: int) -> np.ndarray:
    u = rng.random(count)
    v = rng.random(count)
    sqrt_u = np.sqrt(u)
    return np.stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-12, None)


def _parse_scale(scale_str: str | None) -> tuple[float, float, float]:
    if scale_str is None:
        return (1.0, 1.0, 1.0)
    vals = tuple(float(v) for v in scale_str.split())
    if len(vals) != 3:
        raise ValueError(f"URDF mesh scale must have 3 values, got: {scale_str!r}")
    return vals
