from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* or chardet .* doesn't match a supported version!",
    category=Warning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TACTILE_ROOT = REPO_ROOT / "Isaacsim_tactile_env"
if str(TACTILE_ROOT) not in sys.path:
    sys.path.insert(0, str(TACTILE_ROOT))

from tactile.geometry import ObjectSurfaceSampler, ObjectSurfaceSamplerCfg  # noqa: E402
from tactile.hydroshear import (  # noqa: E402
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
    SurfacePointHydroShearCfg,
)
from tactile.readout import (  # noqa: E402
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
    tangential_axes,
)


OBJECT_ASSETS = {
    "sphere": "sphere.obj",
    "cross": "cross2_09x.stl",
    "cow": "small_cow.stl",
    "torus": "torus_7mm.stl",
}
PROCEDURAL_OBJECTS = ("cylinder", "cube")
GALLERY_OBJECTS = tuple(OBJECT_ASSETS)
OBJECT_CHOICES = tuple(OBJECT_ASSETS) + PROCEDURAL_OBJECTS

OBJECT_SCALES = {
    "sphere": 1.0,
    "cross": 1.0,
    "cow": 0.55,
    "torus": 1.0,
    "cylinder": 1.0,
    "cube": 1.0,
}

OBJECT_THICKNESS_SCALES = {
    "sphere": 1.0,
    "cross": 2.5,
    "cow": 1.0,
    "torus": 1.0,
    "cylinder": 1.0,
    "cube": 1.0,
}


@dataclass
class DemoFrame:
    object_pose_p: np.ndarray
    marker_vectors_p: np.ndarray


@dataclass
class DemoScene:
    object_name: str
    object_path: Path
    elastomer_path: Path
    elastomer_vertices: np.ndarray
    elastomer_faces: np.ndarray
    object_vertices: np.ndarray
    object_faces: np.ndarray
    taxel_points_p: np.ndarray
    frames: list[DemoFrame]
    elastomer_pose_w: np.ndarray


def parse_args(default_object: str = "sphere"):
    parser = argparse.ArgumentParser(description="Original-style Vedo HydroShear demo using the local implementation.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/home/jiafeng/hydroshear/demo_assets"),
        help="Directory containing the original HydroShear demo_assets meshes.",
    )
    parser.add_argument("--object", choices=OBJECT_CHOICES, default=default_object)
    parser.add_argument("--object-scale", type=float, default=None, help="Override the per-object demo mesh scale.")
    parser.add_argument(
        "--object-thickness-scale",
        type=float,
        default=None,
        help="Scale the object along the tactile normal axis. Cross defaults thicker for SDF robustness.",
    )
    parser.add_argument("--gallery", action="store_true", help="Show sphere/cross/cow/torus in a 2x2 layout.")
    parser.add_argument("--cylinder-radius", type=float, default=0.0035)
    parser.add_argument("--cylinder-length", type=float, default=0.018)
    parser.add_argument("--cylinder-segments", type=int, default=64)
    parser.add_argument("--cube-side", type=float, default=0.008)
    parser.add_argument("--num-points", type=int, default=120, help="Target object surface sample count.")
    parser.add_argument("--initial-samples", type=int, default=6000, help="pv.sample_mesh_points count before Poisson downsample.")
    parser.add_argument("--poisson-radius", type=float, default=0.00075)
    parser.add_argument("--sdf-resolution", type=float, default=0.001)
    parser.add_argument("--num-rows", type=int, default=7)
    parser.add_argument("--num-cols", type=int, default=9)
    parser.add_argument("--taxel-margin", type=float, default=0.002)
    parser.add_argument("--local-z-dir", type=float, default=-1.0, choices=(-1.0, 1.0))
    parser.add_argument(
        "--normal-mode",
        choices=("fixed_axis", "sdf_normal"),
        default="fixed_axis",
        help="Use original fixed-axis normal split or per-point elastomer SDF normals.",
    )
    parser.add_argument("--penetration", type=float, default=0.0015)
    parser.add_argument("--clearance", type=float, default=0.0015)
    parser.add_argument("--slide-x", type=float, default=-5.0e-4)
    parser.add_argument("--slide-y", type=float, default=-3.0e-4)
    parser.add_argument(
        "--motion-script",
        choices=("press_slide", "press_left_right_spin", "press_four_way_spin"),
        default="press_slide",
        help="Object motion sequence used by --animate/--gif.",
    )
    parser.add_argument("--object-spin-deg", type=float, default=0.0, help="Rotate the object over the animation.")
    parser.add_argument(
        "--object-spin-axis",
        choices=("normal", "long", "x", "y", "z"),
        default="normal",
        help="Patch-frame axis used by --object-spin-deg.",
    )
    parser.add_argument("--mu", type=float, default=3.0)
    parser.add_argument("--lambda-d", type=float, default=50_000.0)
    parser.add_argument("--lambda-s", type=float, default=100_000.0)
    parser.add_argument("--dilation-scale", type=float, default=38461.5385 * 5.0)
    parser.add_argument("--shear-scale", type=float, default=790000.0 / 40.0 * 10.0)
    parser.add_argument("--arrow-scale", type=float, default=0.001, help="Display scale for green arrows, matching the original demo.")
    parser.add_argument(
        "--arrow-source",
        choices=("marker", "marker_shear", "force"),
        default="marker",
        help="Use marker-field arrows or direct force shear arrows.",
    )
    parser.add_argument(
        "--normalize-arrows",
        action="store_true",
        help="Normalize each panel's arrows to --arrow-length instead of using the original fixed scale.",
    )
    parser.add_argument("--arrow-length", type=float, default=0.0045, help="Maximum displayed green arrow length when --normalize-arrows is used.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Optional output screenshot path.")
    parser.add_argument(
        "--static-frame",
        choices=("press", "slide"),
        default="press",
        help="Frame used by static screenshots/windows. The original README-style image is closer to press.",
    )
    parser.add_argument("--animate", action="store_true", help="Play the contact/shear sequence in the Vedo window.")
    parser.add_argument("--gif", type=Path, default=None, help="Save the contact/shear sequence as an animated GIF.")
    parser.add_argument("--frames", type=int, default=48, help="Frame count for --animate/--gif.")
    parser.add_argument("--fps", type=float, default=12.0, help="Playback/export frame rate for --animate/--gif.")
    parser.add_argument("--camera-orbit-deg", type=float, default=0.0, help="Camera orbit angle over the animation.")
    return parser.parse_args()


def _require_vedo():
    try:
        import vedo
    except ImportError as exc:
        raise SystemExit("Missing dependency: install vedo in the active environment.") from exc

    vedo.settings.enable_default_keyboard_callbacks = False
    vedo.settings.enable_pipeline = False
    return vedo


def _load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if hasattr(mesh, "geometry") and not hasattr(mesh, "faces"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {path}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _raycast_tactile_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_divs: tuple[int, int],
    margin: float,
    local_z_dir: float,
) -> tuple[np.ndarray, int]:
    import open3d as o3d

    vertices32 = np.asarray(vertices, dtype=np.float32)
    faces32 = np.asarray(faces, dtype=np.uint32)
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(vertices32, dtype=o3d.core.Dtype.Float32)
    mesh.triangle.indices = o3d.core.Tensor(faces32, dtype=o3d.core.Dtype.UInt32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    bbox_min = vertices32.min(axis=0)
    bbox_max = vertices32.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    dims = bbox_max - bbox_min
    normal_axis = int(np.argmin(dims))
    axis_idxs = [0, 1, 2]
    axis_idxs.remove(normal_axis)

    div_sz = (dims[axis_idxs] - float(margin) * 2.0) / (np.asarray(num_divs, dtype=np.float32) + 1.0)
    tactile_dx = float(np.min(div_sz))
    u = np.linspace(
        center[axis_idxs[0]] - tactile_dx * (num_divs[0] + 1.0) / 2.0,
        center[axis_idxs[0]] + tactile_dx * (num_divs[0] + 1.0) / 2.0,
        num_divs[0] + 2,
    )[1:-1]
    v = np.linspace(
        center[axis_idxs[1]] - tactile_dx * (num_divs[1] + 1.0) / 2.0,
        center[axis_idxs[1]] + tactile_dx * (num_divs[1] + 1.0) / 2.0,
        num_divs[1] + 2,
    )[1:-1]
    uu, vv = np.meshgrid(u, v)

    local = [None, None, None]
    local[axis_idxs[0]] = uu
    local[axis_idxs[1]] = vv
    local[normal_axis] = np.zeros_like(uu) + center[normal_axis]
    origins = np.stack(local, axis=-1).reshape(-1, 3).astype(np.float32)

    ray_dir = np.zeros(3, dtype=np.float32)
    ray_dir[normal_axis] = float(local_z_dir)
    rays = np.concatenate((origins, np.repeat(ray_dir[None, :], origins.shape[0], axis=0)), axis=-1)
    hits = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    hit_distance = hits["t_hit"].numpy()
    if not np.all(np.isfinite(hit_distance)):
        missing = int((~np.isfinite(hit_distance)).sum())
        raise RuntimeError(f"failed to raycast {missing} tactile points on elastomer mesh")

    origins[:, normal_axis] = origins[:, normal_axis] + hit_distance * float(local_z_dir)
    return origins.astype(np.float64), normal_axis


def _make_pose(translation: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    if rotation is not None:
        pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return pose


def _rot_x(degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    rad = np.deg2rad(float(degrees))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) * c + (1.0 - c) * np.outer(axis, axis) + s * k


def _axis_angle_quat_wxyz(axis: np.ndarray, degrees: float) -> torch.Tensor:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    half = np.deg2rad(float(degrees)) * 0.5
    quat = np.concatenate(([np.cos(half)], axis * np.sin(half)))
    return torch.as_tensor(quat, dtype=torch.float32)


def _spin_axis_vector(axis_name: str, normal_axis: int) -> np.ndarray:
    if axis_name == "normal":
        axis = np.zeros(3, dtype=np.float64)
        axis[int(normal_axis)] = 1.0
        return axis
    if axis_name == "long":
        axis_u, _axis_v = tangential_axes(int(normal_axis))
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_u] = 1.0
        return axis
    axes = {"x": 0, "y": 1, "z": 2}
    axis = np.zeros(3, dtype=np.float64)
    axis[axes[axis_name]] = 1.0
    return axis


def _make_laid_cylinder_arrays(
    *,
    normal_axis: int,
    radius: float,
    length: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis_u, axis_v = tangential_axes(int(normal_axis))
    segments = max(8, int(segments))
    radius = float(radius)
    half_length = float(length) * 0.5

    vertices: list[np.ndarray] = []
    for u in (-half_length, half_length):
        for i in range(segments):
            theta = 2.0 * np.pi * float(i) / float(segments)
            p = np.zeros(3, dtype=np.float64)
            p[axis_u] = u
            p[axis_v] = radius * float(np.sin(theta))
            p[int(normal_axis)] = radius * float(np.cos(theta))
            vertices.append(p)

    left_center = len(vertices)
    p = np.zeros(3, dtype=np.float64)
    p[axis_u] = -half_length
    vertices.append(p.copy())
    right_center = len(vertices)
    p[axis_u] = half_length
    vertices.append(p.copy())

    faces: list[list[int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        left_i = i
        left_j = j
        right_i = segments + i
        right_j = segments + j
        faces.append([left_i, right_i, right_j])
        faces.append([left_i, right_j, left_j])
        faces.append([left_center, left_j, left_i])
        faces.append([right_center, right_i, right_j])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _make_cube_arrays(*, side: float) -> tuple[np.ndarray, np.ndarray]:
    h = float(side) * 0.5
    vertices = np.asarray(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _object_pose_sequence(
    *,
    object_vertices: np.ndarray,
    taxel_points: np.ndarray,
    normal_axis: int,
    local_z_dir: float,
    clearance: float,
    penetration: float,
    slide_x: float,
    slide_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = object_vertices.min(axis=0)
    maxs = object_vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    taxel_center = taxel_points.mean(axis=0)
    axis_u, axis_v = tangential_axes(normal_axis)

    base = np.zeros(3, dtype=np.float64)
    base[axis_u] = taxel_center[axis_u] - center[axis_u]
    base[axis_v] = taxel_center[axis_v] - center[axis_v]

    surface_n = float(np.median(taxel_points[:, normal_axis]))
    if local_z_dir < 0.0:
        prev_n = surface_n - maxs[normal_axis] - float(clearance)
        press_n = surface_n - maxs[normal_axis] + float(penetration)
    else:
        prev_n = surface_n - mins[normal_axis] + float(clearance)
        press_n = surface_n - mins[normal_axis] - float(penetration)

    prev = base.copy()
    prev[normal_axis] = prev_n
    press = base.copy()
    press[normal_axis] = press_n
    slide = press.copy()
    slide[axis_u] += float(slide_x)
    slide[axis_v] += float(slide_y)
    return prev, press, slide


def _motion_positions(prev: np.ndarray, press: np.ndarray, slide: np.ndarray, frame_count: int) -> list[np.ndarray]:
    if frame_count <= 1:
        return [slide.copy()]

    key_times = np.array([0.0, 0.28, 0.62, 0.82, 1.0], dtype=np.float64)
    key_positions = [prev, press, slide, press, prev]
    positions = []
    for i in range(frame_count):
        t = i / float(frame_count - 1)
        seg = min(int(np.searchsorted(key_times, t, side="right")) - 1, len(key_times) - 2)
        seg = max(seg, 0)
        local_t = (t - key_times[seg]) / max(key_times[seg + 1] - key_times[seg], 1.0e-12)
        smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)
        positions.append((1.0 - smooth_t) * key_positions[seg] + smooth_t * key_positions[seg + 1])
    return positions


def _interpolate_keyframes(
    *,
    key_times: np.ndarray,
    key_positions: list[np.ndarray],
    key_spins: list[float],
    frame_count: int,
) -> list[tuple[np.ndarray, float]]:
    if frame_count <= 1:
        return [(key_positions[-1].copy(), float(key_spins[-1]))]

    states = []
    for i in range(frame_count):
        t = i / float(frame_count - 1)
        seg = min(int(np.searchsorted(key_times, t, side="right")) - 1, len(key_times) - 2)
        seg = max(seg, 0)
        local_t = (t - key_times[seg]) / max(key_times[seg + 1] - key_times[seg], 1.0e-12)
        smooth_t = local_t * local_t * (3.0 - 2.0 * local_t)
        pos = (1.0 - smooth_t) * key_positions[seg] + smooth_t * key_positions[seg + 1]
        spin = (1.0 - smooth_t) * float(key_spins[seg]) + smooth_t * float(key_spins[seg + 1])
        states.append((pos, spin))
    return states


def _motion_states(
    prev: np.ndarray,
    press: np.ndarray,
    slide: np.ndarray,
    *,
    frame_count: int,
    script: str,
    spin_degrees: float,
) -> list[tuple[np.ndarray, float]]:
    if script == "press_slide":
        positions = _motion_positions(prev, press, slide, frame_count)
        denom = max(len(positions) - 1, 1)
        return [(pos, float(spin_degrees) * idx / float(denom)) for idx, pos in enumerate(positions)]

    if script not in ("press_left_right_spin", "press_four_way_spin"):
        raise ValueError(f"unknown motion script: {script}")

    delta = slide - press
    if np.linalg.norm(delta) <= 1.0e-12:
        delta = np.array([8.0e-4, 0.0, 0.0], dtype=np.float64)

    if script == "press_four_way_spin":
        delta_u = np.zeros_like(press)
        delta_v = np.zeros_like(press)
        nonzero = np.flatnonzero(np.abs(delta) > 1.0e-12)
        if nonzero.size >= 1:
            delta_u[nonzero[0]] = delta[nonzero[0]]
        else:
            delta_u[0] = 8.0e-4
        if nonzero.size >= 2:
            delta_v[nonzero[1]] = delta[nonzero[1]]
        else:
            fallback_axis = 1 if abs(delta_u[0]) > 0.0 else 0
            delta_v[fallback_axis] = 8.0e-4

        key_times = np.array([0.0, 0.12, 0.25, 0.38, 0.50, 0.63, 0.76, 0.88, 1.0], dtype=np.float64)
        key_positions = [
            prev,
            press,
            press - delta_u,
            press + delta_u,
            press,
            press - delta_v,
            press + delta_v,
            press,
            prev,
        ]
        key_spins = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(spin_degrees),
            float(spin_degrees),
        ]
        return _interpolate_keyframes(
            key_times=key_times,
            key_positions=key_positions,
            key_spins=key_spins,
            frame_count=frame_count,
        )

    left = press - delta
    right = press + delta
    key_times = np.array([0.0, 0.16, 0.36, 0.56, 0.70, 0.90, 1.0], dtype=np.float64)
    key_positions = [prev, press, left, right, press, press, prev]
    key_spins = [0.0, 0.0, 0.0, 0.0, 0.0, float(spin_degrees), float(spin_degrees)]
    return _interpolate_keyframes(
        key_times=key_times,
        key_positions=key_positions,
        key_spins=key_spins,
        frame_count=frame_count,
    )


def _channels_to_vectors(marker_field: torch.Tensor, normal_axis: int) -> np.ndarray:
    flat = marker_field.detach().cpu().numpy().reshape(-1, 3)
    axis_u, axis_v = tangential_axes(normal_axis)
    vectors = np.zeros((flat.shape[0], 3), dtype=np.float64)
    vectors[:, normal_axis] = flat[:, 0]
    vectors[:, axis_u] = flat[:, 1]
    vectors[:, axis_v] = flat[:, 2]
    return vectors


def _shear_uv_to_vectors(shear_uv: torch.Tensor, normal_axis: int) -> np.ndarray:
    flat = shear_uv.detach().cpu().numpy().reshape(-1, 2)
    axis_u, axis_v = tangential_axes(normal_axis)
    vectors = np.zeros((flat.shape[0], 3), dtype=np.float64)
    vectors[:, axis_u] = flat[:, 0]
    vectors[:, axis_v] = flat[:, 1]
    return vectors


def _scale_along_axis(vertices: np.ndarray, axis: int, scale: float) -> np.ndarray:
    if scale == 1.0:
        return vertices
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    out = vertices.copy()
    out[:, axis] = center[axis] + (out[:, axis] - center[axis]) * float(scale)
    return out


def _build_scene(args, object_name: str) -> DemoScene:
    asset_root = args.asset_root.expanduser().resolve()
    elastomer_path = asset_root / "gsmini_elastomer.obj"
    if not elastomer_path.is_file():
        raise FileNotFoundError(elastomer_path)

    print(f"[HydroShear demo] building {object_name} with pytorch_volumetric SDF...", flush=True)
    elastomer_vertices, elastomer_faces = _load_mesh_arrays(elastomer_path)
    taxel_points, normal_axis = _raycast_tactile_points(
        elastomer_vertices,
        elastomer_faces,
        num_divs=(int(args.num_rows), int(args.num_cols)),
        margin=float(args.taxel_margin),
        local_z_dir=float(args.local_z_dir),
    )

    if object_name == "cylinder":
        object_path = Path("<procedural-cylinder>")
        object_vertices, object_faces = _make_laid_cylinder_arrays(
            normal_axis=normal_axis,
            radius=float(args.cylinder_radius),
            length=float(args.cylinder_length),
            segments=int(args.cylinder_segments),
        )
    elif object_name == "cube":
        object_path = Path("<procedural-cube>")
        object_vertices, object_faces = _make_cube_arrays(side=float(args.cube_side))
    else:
        object_path = asset_root / OBJECT_ASSETS[object_name]
        if not object_path.is_file():
            raise FileNotFoundError(object_path)
        object_vertices, object_faces = _load_mesh_arrays(object_path)

    object_scale = OBJECT_SCALES[object_name] if args.object_scale is None else float(args.object_scale)
    object_vertices = object_vertices * float(object_scale)
    thickness_scale = (
        OBJECT_THICKNESS_SCALES[object_name]
        if args.object_thickness_scale is None
        else float(args.object_thickness_scale)
    )
    object_vertices = _scale_along_axis(object_vertices, normal_axis, thickness_scale)

    sampler = ObjectSurfaceSampler(
        ObjectSurfaceSamplerCfg(
            num_points=int(args.num_points),
            poisson_radius=float(args.poisson_radius),
            poisson_initial_num_points=max(int(args.initial_samples), int(args.num_points)),
            sdf_resolution=float(args.sdf_resolution),
            sdf_object_name=f"demo_{object_name}_object",
            dtype=torch.float32,
            device="cpu",
        )
    )
    samples = sampler.sample_arrays(object_vertices, object_faces)

    grid_cfg = TaxelGridCfg(
        num_rows=int(args.num_rows),
        num_cols=int(args.num_cols),
        point_distance=0.001,
        normal_axis=normal_axis,
        dtype=torch.float32,
        device="cpu",
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=grid_cfg,
            taxel_positions_p=torch.as_tensor(taxel_points, dtype=torch.float32),
            elastomer_vertices_p=torch.as_tensor(elastomer_vertices, dtype=torch.float32),
            elastomer_faces=torch.as_tensor(elastomer_faces, dtype=torch.long),
            elastomer_sdf_resolution=float(args.sdf_resolution),
            elastomer_sdf_object_name="demo_gsmini_elastomer",
            hydroshear=SurfacePointHydroShearCfg(
                friction_coefficient=float(args.mu),
                normal_mode=str(args.normal_mode),
                normal_axis=normal_axis,
                normal_direction=float(args.local_z_dir),
                area_mode="unit",
            ),
            projected_surface=ProjectedSurfacePointTrackerCfg(lambda_d=float(args.lambda_d)),
            projection=SurfacePointForceProjectorCfg(lambda_s=float(args.lambda_s)),
            marker_projection=HydroShearMarkerReadoutCfg(
                lambda_s=float(args.lambda_s),
                lambda_d=float(args.lambda_d),
                shear_scale=float(args.shear_scale),
                dilation_scale=float(args.dilation_scale),
                sdf_query_chunk_size=64,
            ),
            output_mode="marker_field",
            output_key="tactile",
        )
    )

    prev_pos, press_pos, slide_pos = _object_pose_sequence(
        object_vertices=object_vertices,
        taxel_points=taxel_points,
        normal_axis=normal_axis,
        local_z_dir=float(args.local_z_dir),
        clearance=float(args.clearance),
        penetration=float(args.penetration),
        slide_x=float(args.slide_x),
        slide_y=float(args.slide_y),
    )
    use_motion = bool(args.animate or args.gif)
    if use_motion:
        motion_states = _motion_states(
            prev_pos,
            press_pos,
            slide_pos,
            frame_count=max(1, int(args.frames)),
            script=str(args.motion_script),
            spin_degrees=float(args.object_spin_deg),
        )
    else:
        motion_states = [(prev_pos, 0.0), (press_pos, 0.0), (slide_pos, 0.0)]

    frames = []
    spin_axis = _spin_axis_vector(str(args.object_spin_axis), normal_axis)
    for pos, spin_angle in motion_states:
        object_rot = _axis_angle_matrix(spin_axis, spin_angle)
        object_quat = _axis_angle_quat_wxyz(spin_axis, spin_angle)
        output = backend.update(
            samples,
            object_pos_e=torch.as_tensor(pos, dtype=torch.float32),
            object_quat_e=object_quat,
        )
        if str(args.arrow_source) == "force":
            marker_vectors = _shear_uv_to_vectors(output.readout.tactile_shear, normal_axis)
        elif str(args.arrow_source) == "marker_shear":
            marker_vectors = _channels_to_vectors(output.marker_readout.shear_field, normal_axis)
            marker_vectors[:, normal_axis] = 0.0
        else:
            marker_vectors = _channels_to_vectors(output.marker_readout.marker_field, normal_axis)
            marker_vectors[:, normal_axis] = 0.0
        frames.append(DemoFrame(object_pose_p=_make_pose(pos, object_rot), marker_vectors_p=marker_vectors))

    elastomer_pose_w = _make_pose(np.array([0.0, -0.06, -0.01]), _rot_x(180.0))
    return DemoScene(
        object_name=object_name,
        object_path=object_path,
        elastomer_path=elastomer_path,
        elastomer_vertices=elastomer_vertices,
        elastomer_faces=elastomer_faces,
        object_vertices=object_vertices,
        object_faces=object_faces,
        taxel_points_p=taxel_points,
        frames=frames,
        elastomer_pose_w=elastomer_pose_w,
    )


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _make_mesh_actor(vedo, path: Path, pose: np.ndarray, *, alpha: float, color: str):
    mesh = vedo.Mesh(str(path)).c(color).alpha(float(alpha))
    mesh.apply_transform(vedo.LinearTransform(pose))
    return mesh


def _make_array_mesh_actor(vedo, vertices: np.ndarray, faces: np.ndarray, pose: np.ndarray, *, alpha: float, color: str):
    mesh = vedo.Mesh([vertices, faces]).c(color).alpha(float(alpha))
    mesh.apply_transform(vedo.LinearTransform(pose))
    return mesh


def _frame_actors(vedo, pose: np.ndarray, *, size: float = 0.003):
    origin = pose[:3, 3]
    colors = ("red", "green", "blue")
    actors = []
    for axis_idx, color in enumerate(colors):
        end = origin + pose[:3, axis_idx] * float(size)
        actors.append(vedo.Arrow(start_pt=origin, end_pt=end, c=color, s=0.000025))
    return actors


def _shear_arrow_actors(vedo, scene: DemoScene, frame: DemoFrame, args):
    starts_w = _transform_points(scene.taxel_points_p, scene.elastomer_pose_w)
    vectors_w = frame.marker_vectors_p @ scene.elastomer_pose_w[:3, :3].T
    norms = np.linalg.norm(vectors_w, axis=1)
    if bool(args.normalize_arrows):
        scale = 0.0 if norms.max(initial=0.0) <= 1.0e-12 else float(args.arrow_length) / float(norms.max())
    else:
        scale = float(args.arrow_scale)

    min_norm = 1.0e-12 if bool(args.normalize_arrows) else 1.0e-6
    arrows = []
    for start, vec, norm in zip(starts_w, vectors_w, norms, strict=True):
        if norm <= min_norm:
            continue
        end = start + vec * scale
        arrows.append(vedo.Arrow(start_pt=start, end_pt=end, c="green", s=0.000015))
    return arrows


def _actors_for_scene(vedo, scene: DemoScene, args, frame: DemoFrame | None = None) -> list:
    if frame is None:
        frame = scene.frames[1] if args.static_frame == "press" and len(scene.frames) > 1 else scene.frames[-1]
    object_pose_w = scene.elastomer_pose_w @ frame.object_pose_p
    actors = [
        _make_mesh_actor(vedo, scene.elastomer_path, scene.elastomer_pose_w, alpha=0.30, color="lightgray"),
        _make_array_mesh_actor(
            vedo,
            scene.object_vertices,
            scene.object_faces,
            object_pose_w,
            alpha=0.30,
            color="slategray",
        ),
    ]
    actors.extend(_frame_actors(vedo, object_pose_w))
    actors.extend(_shear_arrow_actors(vedo, scene, frame, args))
    actors.append(vedo.Text2D(scene.object_name.capitalize(), pos="top-left", s=0.9, c="black"))
    return actors


def _rotate_about_z(vec: np.ndarray, degrees: float) -> np.ndarray:
    rad = np.deg2rad(float(degrees))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot @ vec


def _set_camera(plotter, *, orbit_degrees: float = 0.0):
    base_pos = np.array([0.023415281951362058, 0.033288135031922586, 0.06029433054766119], dtype=np.float64)
    focal = np.array([-0.0004999996162950941, -0.0002499999850988366, 0.004617669895691867], dtype=np.float64)
    base_up = np.array([-0.36158070842044415, -0.721818140926492, 0.5901169059835455], dtype=np.float64)

    pos = focal + _rotate_about_z(base_pos - focal, orbit_degrees)
    up = _rotate_about_z(base_up, orbit_degrees)
    plotter.camera.SetPosition(*pos)
    plotter.camera.SetFocalPoint(*focal)
    plotter.camera.SetViewUp(*up)


def _show_static(vedo, scenes: list[DemoScene], args) -> None:
    shape = (2, 2) if args.gallery else (1, 1)
    plotter = vedo.Plotter(shape=shape, size=(1100, 900) if args.gallery else (900, 760), bg="white", axes=0)
    for idx, scene in enumerate(scenes):
        plotter.at(idx)
        for actor in _actors_for_scene(vedo, scene, args):
            plotter += actor
        _set_camera(plotter)

    print("[HydroShear demo] opening original-style Vedo window...", flush=True)
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(interactive=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"[HydroShear demo] wrote screenshot: {args.screenshot}", flush=True)
    else:
        plotter.show(title="HydroShear Vedo Demo", interactive=True)


def _plot_shape(scene_count: int) -> tuple[int, int]:
    return (2, 2) if scene_count > 1 else (1, 1)


def _plot_size(scene_count: int) -> tuple[int, int]:
    return (1100, 900) if scene_count > 1 else (900, 760)


def _draw_scene_frame(
    vedo,
    plotter,
    scene: DemoScene,
    args,
    frame: DemoFrame,
    *,
    at: int = 0,
    orbit_degrees: float = 0.0,
) -> None:
    plotter.at(at)
    plotter.clear(at=at, deep=True)
    for actor in _actors_for_scene(vedo, scene, args, frame=frame):
        plotter += actor
    _set_camera(plotter, orbit_degrees=orbit_degrees)


def _draw_scene_set_frame(vedo, plotter, scenes: list[DemoScene], args, frame_idx: int) -> None:
    frame_count = max(len(scene.frames) for scene in scenes)
    orbit = 0.0
    if frame_count > 1:
        orbit = float(args.camera_orbit_deg) * frame_idx / float(frame_count - 1)
    for scene_idx, scene in enumerate(scenes):
        frame = scene.frames[frame_idx % len(scene.frames)]
        _draw_scene_frame(vedo, plotter, scene, args, frame, at=scene_idx, orbit_degrees=orbit)


def _show_animation(vedo, scenes: list[DemoScene], args) -> None:
    plotter = vedo.Plotter(shape=_plot_shape(len(scenes)), size=_plot_size(len(scenes)), bg="white", axes=0)
    state = {"index": 0}
    frame_count = max(len(scene.frames) for scene in scenes)

    def on_timer(_event):
        _draw_scene_set_frame(vedo, plotter, scenes, args, state["index"])
        plotter.render()
        state["index"] = (state["index"] + 1) % frame_count

    _draw_scene_set_frame(vedo, plotter, scenes, args, 0)
    plotter.add_callback("timer", on_timer, enable_picking=False)
    plotter.timer_callback("start", dt=max(1, int(1000.0 / max(float(args.fps), 1.0))))
    print("[HydroShear demo] opening animated Vedo window...", flush=True)
    plotter.show(title="HydroShear Animated Vedo Demo", interactive=True)
    plotter.close()


def _save_gif(vedo, scenes: list[DemoScene], args) -> None:
    import imageio.v2 as imageio

    gif_path = args.gif.expanduser().resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[HydroShear demo] writing animated GIF: {gif_path}", flush=True)

    with tempfile.TemporaryDirectory(prefix="hydroshear_demo_gif_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        frame_paths = []
        frame_count = max(len(scene.frames) for scene in scenes)
        plotter = vedo.Plotter(
            shape=_plot_shape(len(scenes)),
            size=_plot_size(len(scenes)),
            bg="white",
            axes=0,
            offscreen=True,
        )
        for frame_idx in range(frame_count):
            _draw_scene_set_frame(vedo, plotter, scenes, args, frame_idx)
            plotter.show(interactive=False, resetcam=False)
            frame_path = tmpdir_path / f"frame_{frame_idx:04d}.png"
            plotter.screenshot(str(frame_path))
            frame_paths.append(frame_path)
        plotter.close()

        images = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(str(gif_path), images, duration=1.0 / max(float(args.fps), 1.0))

    print(f"[HydroShear demo] wrote animated GIF: {gif_path}", flush=True)


def show_demo(args) -> None:
    if int(args.frames) <= 0:
        raise SystemExit("--frames must be positive")
    if float(args.fps) <= 0.0:
        raise SystemExit("--fps must be positive")

    vedo = _require_vedo()
    names = GALLERY_OBJECTS if args.gallery else (args.object,)
    scenes = [_build_scene(args, name) for name in names]

    if args.gif is not None:
        _save_gif(vedo, scenes, args)
        if not args.animate and args.screenshot is None:
            return

    if args.animate:
        _show_animation(vedo, scenes, args)
    else:
        _show_static(vedo, scenes, args)


def main(default_object: str = "sphere"):
    args = parse_args(default_object=default_object)
    show_demo(args)


if __name__ == "__main__":
    main()
