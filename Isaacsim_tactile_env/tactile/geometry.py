from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------

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

    sdf_backend: object | None = None
    """Optional pv.MeshSDF-style query backend for the source mesh."""

    @property
    def num_points(self) -> int:
        return int(self.points_o.shape[0])

    @property
    def total_area(self) -> torch.Tensor:
        return self.area.sum()

    def query_sdf(self, points_o: torch.Tensor, *, chunk_size: int | None = None) -> "ObjectMeshSdfResult":
        if self.sdf_backend is None:
            raise RuntimeError("ObjectSurfaceSamples requires a pytorch_volumetric SDF backend for SDF queries")
        return self.sdf_backend.query(points_o, chunk_size=chunk_size)


@dataclass
class ObjectMeshSdfResult:
    sdf: torch.Tensor
    unsigned_distance: torch.Tensor
    closest_points_o: torch.Tensor
    closest_normals_o: torch.Tensor | None = None
    closest_face_index: torch.Tensor | None = None


@dataclass
class ObjectSurfaceSamplerCfg:
    num_points: int = 2048
    seed: int | None = 0
    poisson_radius: float = 0.00075
    poisson_initial_num_points: int = 50_000
    sdf_resolution: float = 0.001
    sdf_cache_path: str | os.PathLike | None = None
    sdf_object_name: str = "mesh"
    sdf_clean_cache: bool = False
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class ObjectSurfaceSampler:
    """Sampler for object mesh surface points.

    The returned samples are in object-local mesh coordinates. This mirrors the
    original HydroFOTS path: draw dense samples with
    `pytorch_volumetric.sample_mesh_points`, then Poisson disk downsample them.
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
        if self.cfg.poisson_initial_num_points <= 0:
            raise ValueError("poisson_initial_num_points must be positive")

        tri = vertices_np[faces_np]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        cross = np.cross(e1, e2)
        double_area = np.linalg.norm(cross, axis=1)
        face_area = 0.5 * double_area
        valid = face_area > 1.0e-16
        if not np.any(valid):
            raise ValueError("mesh has no non-degenerate triangles")

        faces_valid = faces_np[valid]
        face_area = face_area[valid]
        total_area = float(face_area.sum())

        sdf_backend = self._make_sdf_backend(vertices_np, faces_valid)
        points_t, normals_t = sdf_backend.sample_poisson_surface_points(
            radius=float(self.cfg.poisson_radius),
            initial_num_points=max(int(self.cfg.poisson_initial_num_points), int(self.cfg.num_points)),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        if points_t.numel() == 0:
            raise RuntimeError("Poisson disk sampling returned no surface points")

        normals_t = normals_t / normals_t.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        area = np.full((points_t.shape[0],), total_area / float(points_t.shape[0]), dtype=np.float64)

        return ObjectSurfaceSamples(
            points_o=points_t,
            normals_o=normals_t,
            area=torch.as_tensor(area, dtype=self.cfg.dtype, device=self.cfg.device),
            face_index=None,
            vertices_o=torch.as_tensor(vertices_np, dtype=self.cfg.dtype, device=self.cfg.device),
            faces=torch.as_tensor(faces_valid, dtype=torch.long, device=self.cfg.device),
            sdf_backend=sdf_backend,
        )

    def _make_sdf_backend(self, vertices_np, faces_np):
        return PytorchVolumetricMeshSdf(
            vertices_np,
            faces_np,
            object_name=self.cfg.sdf_object_name,
            resolution=float(self.cfg.sdf_resolution),
            device=self.cfg.device,
            cache_path=self.cfg.sdf_cache_path,
            clean_cache=bool(self.cfg.sdf_clean_cache),
        )


def load_trimesh(mesh_path: str | os.PathLike, *, scale: tuple[float, float, float] | None = None):
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {mesh_path}")
    if scale is not None:
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    return mesh


class PytorchVolumetricMeshSdf:
    """pv.MeshSDF/CachedSDF wrapper matching the original HydroFOTS SDF path."""

    def __init__(
        self,
        vertices,
        faces,
        *,
        object_name: str = "mesh",
        resolution: float = 0.001,
        device: str | torch.device = "cpu",
        cache_path: str | os.PathLike | None = None,
        clean_cache: bool = False,
    ):
        import pytorch_volumetric as pv

        self.device = device
        self.object_name = _safe_sdf_name(object_name)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="hydroshear_sdf_")
        self.mesh_path = self._write_obj(vertices, faces)
        if cache_path is None:
            cache_path = Path(self._tmpdir.name) / f"{self.object_name}_cached_sdf.pkl"
        else:
            cache_path = Path(os.path.expanduser(str(cache_path)))
            if cache_path.parent:
                cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.mesh_pv = pv.MeshObjectFactory(str(self.mesh_path))
        if hasattr(self.mesh_pv, "_mesh") and hasattr(self.mesh_pv._mesh, "compute_vertex_normals"):
            self.mesh_pv._mesh.compute_vertex_normals()
        self.mesh_pv.precompute_sdf()
        self.mesh_sdf_gt = pv.MeshSDF(self.mesh_pv)
        self.mesh_sdf = pv.CachedSDF(
            object_name=self.object_name,
            resolution=float(resolution),
            range_per_dim=self.mesh_pv.bounding_box(padding=0.1),
            gt_sdf=self.mesh_sdf_gt,
            device=device,
            cache_path=str(cache_path),
            clean_cache=bool(clean_cache),
        )

    def query(self, points_o: torch.Tensor, *, chunk_size: int | None = None) -> ObjectMeshSdfResult:
        points_o = torch.as_tensor(points_o)
        if points_o.shape[-1] != 3:
            raise ValueError("points_o must end with 3 coordinates")

        flat = points_o.reshape(-1, 3)
        if chunk_size is None or chunk_size <= 0 or chunk_size >= flat.shape[0]:
            sdf, normals = self.mesh_sdf(flat)
        else:
            sdf_parts = []
            normal_parts = []
            for start in range(0, flat.shape[0], int(chunk_size)):
                end = min(start + int(chunk_size), flat.shape[0])
                sdf_part, normal_part = self.mesh_sdf(flat[start:end])
                sdf_parts.append(sdf_part)
                normal_parts.append(normal_part)
            sdf = torch.cat(sdf_parts, dim=0)
            normals = torch.cat(normal_parts, dim=0)

        sdf = sdf.to(device=points_o.device, dtype=points_o.dtype).reshape(points_o.shape[:-1])
        normals = normals.to(device=points_o.device, dtype=points_o.dtype).reshape(points_o.shape)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        closest = points_o - sdf.unsqueeze(-1) * normals
        return ObjectMeshSdfResult(
            sdf=sdf,
            unsigned_distance=sdf.abs(),
            closest_points_o=closest,
            closest_normals_o=normals,
            closest_face_index=None,
        )

    def sample_poisson_surface_points(
        self,
        *,
        radius: float,
        initial_num_points: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import point_cloud_utils as pcu
        import pytorch_volumetric as pv

        dbpath = Path(self._tmpdir.name) / f"{self.object_name}_model_points_cache.pkl"
        points, normals, _ = pv.sample_mesh_points(
            self.mesh_pv,
            num_points=int(initial_num_points),
            dbpath=str(dbpath),
            device=device,
            dtype=torch.float32,
        )
        dbpath.unlink(missing_ok=True)

        idx_np = np.asarray(
            pcu.downsample_point_cloud_poisson_disk(
                points.detach().cpu().numpy(),
                radius=float(radius),
                target_num_samples=-1,
            ),
            dtype=np.int64,
        )
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=points.device)
        return points[idx].to(device=device, dtype=dtype), normals[idx].to(device=device, dtype=dtype)

    def _write_obj(self, vertices, faces) -> Path:
        vertices_np = np.asarray(vertices, dtype=np.float64)
        faces_np = np.asarray(faces, dtype=np.int64)
        path = Path(self._tmpdir.name) / f"{self.object_name}.obj"
        with path.open("w", encoding="utf-8") as f:
            for v in vertices_np:
                f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
            for face in faces_np:
                f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")
        return path

    def __del__(self):
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()


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


def _safe_sdf_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return safe or "mesh"


def _parse_scale(scale_str: str | None) -> tuple[float, float, float]:
    if scale_str is None:
        return (1.0, 1.0, 1.0)
    vals = tuple(float(v) for v in scale_str.split())
    if len(vals) != 3:
        raise ValueError(f"URDF mesh scale must have 3 values, got: {scale_str!r}")
    return vals

# ---------------------------------------------------------------------------
# Elastomer
# ---------------------------------------------------------------------------

@dataclass
class ElastomerSdfResult:
    points_p: torch.Tensor
    sdf: torch.Tensor
    closest_points_p: torch.Tensor | None = None
    normals_p: torch.Tensor | None = None


class MeshPatchElastomerSdf:
    """Elastomer mesh SDF in patch frame."""

    def __init__(
        self,
        *,
        vertices_p: torch.Tensor,
        faces: torch.Tensor,
        chunk_size: int | None = 2048,
        sdf_resolution: float = 0.001,
        sdf_cache_path: str | os.PathLike | None = None,
        sdf_object_name: str = "elastomer",
        sdf_clean_cache: bool = False,
    ):
        self.vertices_p = torch.as_tensor(vertices_p, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, dtype=torch.long)
        self.chunk_size = chunk_size
        self._pv_sdf = PytorchVolumetricMeshSdf(
            self.vertices_p.detach().cpu().numpy(),
            self.faces.detach().cpu().numpy(),
            object_name=sdf_object_name,
            resolution=float(sdf_resolution),
            device=self.vertices_p.device,
            cache_path=sdf_cache_path,
            clean_cache=bool(sdf_clean_cache),
        )

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

        out = self._pv_sdf.query(points_p, chunk_size=self.chunk_size)
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


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------

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

    def __init__(self, elastomer_sdf: MeshPatchElastomerSdf):
        if elastomer_sdf is None:
            raise ValueError("SurfacePointContactQuery requires a mesh elastomer SDF")
        self.elastomer_sdf = elastomer_sdf

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
