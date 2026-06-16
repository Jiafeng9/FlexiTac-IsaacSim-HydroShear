from __future__ import annotations

import argparse
import csv
import os
import site
import sys
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# --- SimulationApp MUST be created before any omni/isaaclab imports ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ALOHA bimanual drag control and visualization with cube objects")
parser.add_argument("--port", type=int, default=8080, help="Viser server port")
parser.add_argument("--save_dir", type=str, default="", help="Directory for saved trajectory .npz files")
parser.add_argument("--slow_interval", type=int, default=8, help="UI refresh interval for expensive updates")
parser.add_argument(
    "--compare_hydro_normal",
    action="store_true",
    help="Show original WarpSDF normal, HydroShear normal, and HydroShear shear x/y.",
)
parser.add_argument(
    "--tactile_backend",
    choices=("normal", "taxel_shear", "surface_hydro"),
    default="normal",
    help="Main tactile backend: original normal, taxel-level shear, or surface-point HydroShear.",
)
parser.add_argument(
    "--shear_vis_vmax",
    type=float,
    default=0.0,
    help="Fixed vmax for shear heatmaps. Use 0 for per-frame auto scale.",
)
parser.add_argument(
    "--shear_vis_deadband",
    type=float,
    default=0.0,
    help="Display-only deadband for shear heatmaps; values with abs below this are shown as zero.",
)
parser.add_argument(
    "--shear_csv_dir",
    type=str,
    default="",
    help="Directory for Start Shear CSV recordings. Default: output/tactile_shear_csv.",
)
parser.add_argument(
    "--shear_csv_interval",
    type=int,
    default=1,
    help="Record tactile_shear CSV every N simulation steps.",
)
parser.add_argument(
    "--hydro_normal_scale",
    type=float,
    default=1.0,
    help="Readout scale applied to HydroShear normal values in --compare_hydro_normal mode.",
)
parser.add_argument(
    "--hydro_shear_scale",
    type=float,
    default=1.0,
    help="Readout scale applied to HydroShear shear x/y values in --compare_hydro_normal mode.",
)
parser.add_argument(
    "--hydro_shear_stiffness",
    type=float,
    default=None,
    help="Optional HydroShear tangential stiffness override in --compare_hydro_normal mode.",
)
parser.add_argument(
    "--use_bump_pad",
    action="store_true",
    help="Use the generated bump-pad ALOHA URDF instead of the flat-pad tactile URDF.",
)
parser.add_argument(
    "--robot_urdf",
    type=str,
    default="",
    help="Optional robot URDF override. Takes precedence over --use_bump_pad.",
)
parser.add_argument(
    "--force_flat_elastomer_sdf",
    action="store_true",
    help="Force HydroShear to use the flat patch SDF instead of the elastomer USD mesh SDF.",
)
parser.add_argument(
    "--hydro_debug_print",
    action="store_true",
    help="Print HydroShear sensor diagnostics, including whether each pad uses mesh or flat elastomer SDF.",
)
parser.add_argument(
    "--auto_object_slide_probe",
    choices=("none", "x", "y", "both"),
    default="none",
    help="Run object-slide CSV probes automatically from headless mode and exit after they finish.",
)
parser.add_argument("--auto_object_slide_pad", type=int, default=2, help="Pad index for --auto_object_slide_probe.")
parser.add_argument(
    "--auto_object_slide_distance_mm",
    type=float,
    default=2.0,
    help="Total object-slide distance for --auto_object_slide_probe.",
)
parser.add_argument(
    "--auto_object_slide_step_mm",
    type=float,
    default=0.5,
    help="Per-frame object-slide step for --auto_object_slide_probe.",
)
parser.add_argument(
    "--auto_object_slide_settle_steps",
    type=int,
    default=30,
    help="Settle steps before --auto_object_slide_probe recording starts.",
)
parser.add_argument(
    "--auto_object_slide_hold_steps",
    type=int,
    default=1,
    help="Hold steps between object-slide probe moves.",
)
parser.add_argument(
    "--disable_working_probe_demo",
    action="store_true",
    help="Disable the default visual working demo that runs pad2 shear_x then shear_y object slides.",
)
AppLauncher.add_app_launcher_args(parser)
_AUTO_OBJECT_SLIDE_PROBE_EXPLICIT = any(
    arg == "--auto_object_slide_probe" or arg.startswith("--auto_object_slide_probe=")
    for arg in sys.argv[1:]
)
args = parser.parse_args()
WORKING_PROBE_DEMO = (
    args.auto_object_slide_probe == "none"
    and not _AUTO_OBJECT_SLIDE_PROBE_EXPLICIT
    and not bool(args.disable_working_probe_demo)
)


def _preload_viser_websockets() -> None:
    """Keep Isaac/Kit packages from shadowing the conda websockets package used by Viser."""

    for site_dir in reversed(site.getsitepackages()):
        if site_dir in sys.path:
            sys.path.remove(site_dir)
        sys.path.insert(0, site_dir)

    try:
        import websockets.asyncio.server  # noqa: F401
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")).startswith("websockets"):
            raise ModuleNotFoundError(
                "Viser requires websockets.asyncio.server. Install a compatible version with: "
                'pip install -U "websockets>=13"'
            ) from e
        raise


_preload_viser_websockets()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Now safe to import everything else ---
sys.path.insert(0, str(Path(__file__).parent))
from aloha_ik_controller import AlohaArmIKController
from aloha.camera import AlohaCameraCfg
from aloha.cfg import AlohaTactileEnvCfg, DATASET_JOINT_ORDER
from aloha.env import AlohaTactileEnv
from aloha.scene import AlohaSimCfg
from aloha.tactile import HydroShearTactileBackendCfg, TaxelShearTactileBackendCfg


# ---------------------------------------------------------------------------
# Tactile heatmap rendering
# ---------------------------------------------------------------------------

def _jet_colormap(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _grid_vis_scale(grid: np.ndarray, scale: int) -> int:
    grid = np.asarray(grid)
    if grid.ndim < 2:
        return int(scale)
    min_cells = max(1, int(min(grid.shape[0], grid.shape[1])))
    return max(int(scale), int(np.ceil(120.0 / float(min_cells))))


def tactile_to_rgb(grid: np.ndarray, scale: int = 8, vmax: float | None = None) -> np.ndarray:
    from PIL import Image

    grid = tactile_normal_channel(grid)
    scale = _grid_vis_scale(grid, scale)
    vmax = float(grid.max()) if vmax is None else float(vmax)
    normed = grid / (vmax + 1e-8) if vmax > 0 else np.zeros_like(grid)
    rgb = _jet_colormap(normed)
    img = Image.fromarray(rgb)
    img = img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    return np.array(img)


def positive_tactile_to_rgb(grid: np.ndarray, scale: int = 8, vmax: float | None = None) -> np.ndarray:
    from PIL import Image

    grid = np.asarray(grid)
    scale = _grid_vis_scale(grid, scale)
    vmax = float(grid.max()) if vmax is None else float(vmax)
    normed = np.clip(grid / vmax, 0.0, 1.0) if vmax > 0 else np.zeros_like(grid)
    rgb = _jet_colormap(normed)
    img = Image.fromarray(rgb)
    img = img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    return np.array(img)


def tactile_normal_channel(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim >= 3 and grid.shape[-1] == 3:
        return grid[..., 0]
    return grid


def tactile_shear_uv(grid: np.ndarray | None) -> np.ndarray | None:
    if grid is None:
        return None
    grid = np.asarray(grid)
    if grid.ndim >= 3 and grid.shape[-1] == 3:
        return grid[..., 1:3]
    return grid


def tactile_marker_vector_uv(grid: np.ndarray | None) -> np.ndarray | None:
    if grid is None:
        return None
    grid = np.asarray(grid)
    if grid.ndim >= 3 and grid.shape[-1] == 3:
        return grid[..., 1:3]
    return None


def signed_tactile_to_rgb(grid: np.ndarray, scale: int = 8, vmax: float | None = None) -> np.ndarray:
    from PIL import Image

    grid = np.asarray(grid)
    scale = _grid_vis_scale(grid, scale)
    vmax = float(np.max(np.abs(grid))) if vmax is None else float(vmax)
    if vmax <= 0.0:
        rgb = np.full(grid.shape + (3,), 255, dtype=np.uint8)
    else:
        signed = np.clip(grid / vmax, -1.0, 1.0)
        mag = np.abs(signed)
        base = (255.0 * (1.0 - mag)).astype(np.uint8)
        rgb = np.empty(grid.shape + (3,), dtype=np.uint8)
        positive = signed >= 0.0
        rgb[..., 0] = np.where(positive, 255, base)
        rgb[..., 1] = base
        rgb[..., 2] = np.where(positive, base, 255)
    img = Image.fromarray(rgb)
    img = img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    return np.array(img)


def _label_rgb(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    lines = str(text).splitlines()
    label_h = max(22, 18 * len(lines) + 4)
    labeled = Image.new("RGB", (pil.width, pil.height + label_h), (0, 0, 0))
    labeled.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(labeled)
    for i, line in enumerate(lines):
        draw.text((5, 3 + 18 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(labeled)


def _fmt_scalar(value: float) -> str:
    value = float(value)
    if 0.0 < abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _shear_vis_deadband() -> float:
    return max(0.0, float(getattr(args, "shear_vis_deadband", 0.0)))


def _apply_shear_vis_deadband(grid: np.ndarray) -> np.ndarray:
    deadband = _shear_vis_deadband()
    grid = np.asarray(grid)
    if deadband <= 0.0:
        return grid
    return np.where(np.abs(grid) < deadband, 0.0, grid)


def _shear_vis_vmax(*grids: np.ndarray) -> float:
    fixed_vmax = max(0.0, float(getattr(args, "shear_vis_vmax", 0.0)))
    if fixed_vmax > 0.0:
        return fixed_vmax
    if not grids:
        return 0.0
    return max(float(np.max(np.abs(grid))) for grid in grids)


def _tangential_axes(normal_axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(int(normal_axis))
    return axes[0], axes[1]


def _quat_apply_np(q_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    w = q[0]
    xyz = q[1:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _read_pad_axes_world(env, cfg) -> dict[int, dict[str, np.ndarray]]:
    setup = getattr(env, "_tactile_setup", None)
    backend = getattr(setup, "backend", None)
    states = getattr(setup, "sensors", None)
    if states is None:
        states = getattr(backend, "_states", None)
    slot_order = getattr(setup, "sensor_slot_order", None)
    if not states or not hasattr(backend, "_patch_world_pose"):
        return {}

    axis_u, axis_v = _tangential_axes(int(cfg.tactile.normal_axis))
    basis_n = np.zeros(3, dtype=np.float64)
    basis_u = np.zeros(3, dtype=np.float64)
    basis_v = np.zeros(3, dtype=np.float64)
    basis_n[int(cfg.tactile.normal_axis)] = 1.0
    basis_u[axis_u] = 1.0
    basis_v[axis_v] = 1.0

    axes_by_slot: dict[int, dict[str, np.ndarray]] = {}
    for i, state in enumerate(states):
        try:
            slot_value = getattr(state, "slot", None)
            if slot_value is None:
                if slot_order is None or i >= len(slot_order):
                    continue
                slot_value = slot_order[i]
            slot = int(slot_value)
            patch_pos_w, patch_quat_w = backend._patch_world_pose(state)
            pos = patch_pos_w[0].detach().cpu().numpy() if hasattr(patch_pos_w, "detach") else np.asarray(patch_pos_w)[0]
            quat = patch_quat_w[0].detach().cpu().numpy() if hasattr(patch_quat_w, "detach") else np.asarray(patch_quat_w)[0]
            signs = backend._shear_axis_signs_for_slot(slot) if hasattr(backend, "_shear_axis_signs_for_slot") else (1.0, 1.0)
            normal_w = _quat_apply_np(quat, basis_n)
            axis_x_w = _quat_apply_np(quat, basis_u) * float(signs[0])
            axis_y_w = _quat_apply_np(quat, basis_v) * float(signs[1])
            axes_by_slot[slot] = {
                "position_w": pos,
                "normal_w": normal_w,
                "axis_x_w": axis_x_w,
                "axis_y_w": axis_y_w,
            }
        except Exception:
            continue
    return axes_by_slot


def _read_hydro_target_poses_world(env) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    setup = getattr(env, "_tactile_setup", None)
    if setup is None:
        return {}
    tracker = getattr(setup, "target_tracker", None)
    states = getattr(setup, "sensors", None)
    if tracker is None or not states:
        return {}

    poses_by_slot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, state in enumerate(states):
        try:
            slot = int(getattr(state, "slot", i))
            pos, quat = tracker.target_pose_for_sensor(i)
            poses_by_slot[slot] = (
                np.asarray(pos, dtype=np.float64).reshape(3).copy(),
                _normalize_quat_wxyz_np(np.asarray(quat, dtype=np.float64).reshape(4)),
            )
        except Exception:
            continue
    return poses_by_slot


def _read_hydro_relative_poses(env) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    setup = getattr(env, "_tactile_setup", None)
    if setup is None:
        return {}
    backend = getattr(setup, "backend", None)
    tracker = getattr(setup, "target_tracker", None)
    states = getattr(setup, "sensors", None)
    if backend is None or tracker is None or not states or not hasattr(backend, "_patch_world_pose"):
        return {}

    poses_by_slot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, state in enumerate(states):
        try:
            slot = int(getattr(state, "slot", i))
            patch_pos_w, patch_quat_w = backend._patch_world_pose(state)
            patch_pos = patch_pos_w[0].detach().cpu().numpy() if hasattr(patch_pos_w, "detach") else np.asarray(patch_pos_w)[0]
            patch_quat = patch_quat_w[0].detach().cpu().numpy() if hasattr(patch_quat_w, "detach") else np.asarray(patch_quat_w)[0]
            target_pos, target_quat = tracker.target_pose_for_sensor(i)
            poses_by_slot[slot] = _relative_pose_np(
                parent_pos_w=np.asarray(patch_pos, dtype=np.float64).reshape(3),
                parent_quat_w=np.asarray(patch_quat, dtype=np.float64).reshape(4),
                child_pos_w=np.asarray(target_pos, dtype=np.float64).reshape(3),
                child_quat_w=np.asarray(target_quat, dtype=np.float64).reshape(4),
            )
        except Exception:
            continue
    return poses_by_slot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_mesh_path_from_urdf(urdf_path: str):
    urdf_path = os.path.expanduser(urdf_path)
    if not os.path.isfile(urdf_path):
        return None
    tree = ET.parse(urdf_path)
    for visual in tree.getroot().iter("visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_el = geom.find("mesh")
        if mesh_el is None:
            continue
        filename = mesh_el.get("filename", "")
        scale_str = mesh_el.get("scale", "1 1 1")
        scale = tuple(float(v) for v in scale_str.split())
        if len(scale) != 3:
            scale = (1.0, 1.0, 1.0)
        mesh_abs = os.path.normpath(os.path.join(os.path.dirname(urdf_path), filename))
        if os.path.isfile(mesh_abs):
            return mesh_abs, scale
    return None


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_conjugate_wxyz_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4).copy()
    q[1:] *= -1.0
    return q


def _normalize_quat_wxyz_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / (np.linalg.norm(q) + 1.0e-12)


def _quat_angle_between_wxyz(q0: np.ndarray | None, q1: np.ndarray | None) -> float:
    if q0 is None or q1 is None:
        return 0.0
    a = _normalize_quat_wxyz_np(q0)
    b = _normalize_quat_wxyz_np(q1)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _relative_pose_np(
    *,
    parent_pos_w: np.ndarray,
    parent_quat_w: np.ndarray,
    child_pos_w: np.ndarray,
    child_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_quat_inv = _quat_conjugate_wxyz_np(_normalize_quat_wxyz_np(parent_quat_w))
    child_pos_parent = _quat_apply_np(parent_quat_inv, np.asarray(child_pos_w, dtype=np.float64).reshape(3) - np.asarray(parent_pos_w, dtype=np.float64).reshape(3))
    child_quat_parent = _quat_mul_wxyz(parent_quat_inv, _normalize_quat_wxyz_np(child_quat_w))
    return child_pos_parent, _normalize_quat_wxyz_np(child_quat_parent)


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._targets = {
            "left": {
                "pos": [0.0, -0.167, 0.074],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "gripper": 1.0,
                "dragging": False,
            },
            "right": {
                "pos": [0.0, 0.167, 0.074],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "gripper": 1.0,
                "dragging": False,
            },
        }
        self._reset_requested = False
        self._recording = False
        self._shear_csv_recording = False
        self._object_slide_trial_requests: list[dict] = []

    def set_target(self, arm: str, x, y, z):
        with self._lock:
            self._targets[arm]["pos"] = [float(x), float(y), float(z)]

    def set_target_quat(self, arm: str, w, x, y, z):
        with self._lock:
            self._targets[arm]["quat"] = [float(w), float(x), float(y), float(z)]

    def get_target(self, arm: str):
        with self._lock:
            return list(self._targets[arm]["pos"])

    def get_target_quat(self, arm: str):
        with self._lock:
            return list(self._targets[arm]["quat"])

    def set_gripper(self, arm: str, g):
        with self._lock:
            self._targets[arm]["gripper"] = float(g)

    def get_gripper(self, arm: str):
        with self._lock:
            return float(self._targets[arm]["gripper"])

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset(self):
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False

    def set_dragging(self, arm: str, v):
        with self._lock:
            self._targets[arm]["dragging"] = bool(v)

    def is_dragging(self, arm: str):
        with self._lock:
            return bool(self._targets[arm]["dragging"])

    def set_recording(self, v):
        with self._lock:
            self._recording = bool(v)

    def is_recording(self):
        with self._lock:
            return self._recording

    def set_shear_csv_recording(self, v):
        with self._lock:
            self._shear_csv_recording = bool(v)

    def is_shear_csv_recording(self):
        with self._lock:
            return self._shear_csv_recording

    def request_object_slide_trial(
        self,
        pad: int,
        axis: str,
        direction: float,
        distance_m: float,
        step_m: float,
        settle_steps: int,
        hold_steps: int,
        auto_place: bool,
        preload_m: float,
    ):
        with self._lock:
            self._object_slide_trial_requests.append(
                {
                    "pad": int(pad),
                    "axis": str(axis),
                    "direction": float(direction),
                    "distance_m": max(0.0, float(distance_m)),
                    "step_m": max(1.0e-6, float(step_m)),
                    "settle_steps": max(0, int(settle_steps)),
                    "hold_steps": max(1, int(hold_steps)),
                    "auto_place": bool(auto_place),
                    "preload_m": max(0.0, float(preload_m)),
                }
            )

    def consume_object_slide_trial_request(self):
        with self._lock:
            if not self._object_slide_trial_requests:
                return None
            return self._object_slide_trial_requests.pop(0)


shared = SharedState()


def build_joint_name_map(urdf_joint_names):
    mapping = []
    dataset_lower = [n.lower() for n in DATASET_JOINT_ORDER]
    for urdf_name in urdf_joint_names:
        name = urdf_name.lower()
        found = None
        if name in dataset_lower:
            found = dataset_lower.index(name)
        else:
            name_underscore = name.replace("/", "_")
            for j, dt in enumerate(dataset_lower):
                if name_underscore == dt.replace("/", "_"):
                    found = j
                    break
        mapping.append(found)
    return mapping


# ---------------------------------------------------------------------------
# Trajectory recorder for current env outputs
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Buffers per-step data and saves to .npz on flush."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._traj_count = len([f for f in os.listdir(self._save_dir) if f.endswith(".npz")])
        self._buf: dict[str, list[np.ndarray]] = {}
        self._steps = 0

    def _append(self, key: str, arr: np.ndarray):
        self._buf.setdefault(key, []).append(np.asarray(arr).copy())

    def record(
        self,
        obs: dict,
        left_target_pos: np.ndarray,
        left_target_quat: np.ndarray,
        left_gripper: float,
        right_target_pos: np.ndarray,
        right_target_quat: np.ndarray,
        right_gripper: float,
        left_ee_pos: np.ndarray | None = None,
        left_ee_quat: np.ndarray | None = None,
        right_ee_pos: np.ndarray | None = None,
        right_ee_quat: np.ndarray | None = None,
        joint_commands: np.ndarray | None = None,
    ):
        left_action = np.concatenate([
            np.asarray(left_target_pos, dtype=np.float32).reshape(3),
            np.asarray(left_target_quat, dtype=np.float32).reshape(4),
            np.array([left_gripper], dtype=np.float32),
        ])
        right_action = np.concatenate([
            np.asarray(right_target_pos, dtype=np.float32).reshape(3),
            np.asarray(right_target_quat, dtype=np.float32).reshape(4),
            np.array([right_gripper], dtype=np.float32),
        ])
        self._append("left_actions", left_action)
        self._append("right_actions", right_action)
        self._append("actions", np.concatenate([left_action, right_action], axis=0))
        self._append("tactile", obs["tactile"])
        if "tactile_force" in obs:
            self._append("tactile_force", obs["tactile_force"])
        if "tactile_shear" in obs:
            self._append("tactile_shear", obs["tactile_shear"])
        if "tactile_slip_ratio" in obs:
            self._append("tactile_slip_ratio", obs["tactile_slip_ratio"])
        if "tactile_shear_vector_w" in obs:
            self._append("tactile_shear_vector_w", obs["tactile_shear_vector_w"])
        if "tactile_hydro" in obs:
            self._append("tactile_hydro", obs["tactile_hydro"])
        if "tactile_hydro_force" in obs:
            self._append("tactile_hydro_force", obs["tactile_hydro_force"])
        if "tactile_hydro_shear" in obs:
            self._append("tactile_hydro_shear", obs["tactile_hydro_shear"])
        if "tactile_marker" in obs:
            self._append("tactile_marker", obs["tactile_marker"])
        if "tactile_marker_shear" in obs:
            self._append("tactile_marker_shear", obs["tactile_marker_shear"])
        if "tactile_hydro_marker" in obs:
            self._append("tactile_hydro_marker", obs["tactile_hydro_marker"])
        if "tactile_hydro_marker_shear" in obs:
            self._append("tactile_hydro_marker_shear", obs["tactile_hydro_marker_shear"])
        self._append("joint_pos", obs["joint_pos"])
        self._append("joint_vel", obs["joint_vel"])
        self._append("plug_pose", obs["plug_pose"])
        self._append("socket_pose", obs["socket_pose"])

        if "rgb" in obs:
            self._append("rgb", obs["rgb"])

        if left_ee_pos is not None and left_ee_quat is not None:
            self._append("left_eef_pos_quat", np.concatenate([
                np.asarray(left_ee_pos, dtype=np.float32).reshape(3),
                np.asarray(left_ee_quat, dtype=np.float32).reshape(4),
            ]))

        if right_ee_pos is not None and right_ee_quat is not None:
            self._append("right_eef_pos_quat", np.concatenate([
                np.asarray(right_ee_pos, dtype=np.float32).reshape(3),
                np.asarray(right_ee_quat, dtype=np.float32).reshape(4),
            ]))

        if joint_commands is not None:
            self._append("joint_commands", np.asarray(joint_commands, dtype=np.float32))

        self._steps += 1

    @property
    def num_steps(self) -> int:
        return self._steps

    def flush(self) -> str | None:
        if self._steps == 0:
            return None

        data = {k: np.stack(v, axis=0) for k, v in self._buf.items()}
        data["traj_lengths"] = np.array([self._steps], dtype=np.int32)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"traj_{self._traj_count:04d}_{ts}.npz"
        path = os.path.join(self._save_dir, fname)
        np.savez_compressed(path, **data)

        n = self._steps
        self._buf.clear()
        self._steps = 0
        self._traj_count += 1
        print(f"[SAVE] Trajectory saved: {path} ({n} steps, keys={sorted(data.keys())})", flush=True)
        return path


class TactileShearCsvRecorder:
    """Streams all tactile_shear taxels to CSV while the Viser button is active."""

    def __init__(self, save_dir: str, pad_labels: list[str], interval: int = 1, normal_axis: int = 2):
        self._save_dir = save_dir
        self._pad_labels = list(pad_labels)
        self._interval = max(1, int(interval))
        self._normal_axis = int(normal_axis)
        os.makedirs(self._save_dir, exist_ok=True)
        self._file = None
        self._writer = None
        self._path: str | None = None
        self._frames = 0
        self._rows = 0
        self._file_count = len([f for f in os.listdir(self._save_dir) if f.endswith(".csv")])
        self._prev_left_target_pos: np.ndarray | None = None
        self._prev_right_target_pos: np.ndarray | None = None
        self._prev_left_ee_pos: np.ndarray | None = None
        self._prev_right_ee_pos: np.ndarray | None = None
        self._prev_plug_pos: np.ndarray | None = None
        self._prev_socket_pos: np.ndarray | None = None
        self._prev_plug_quat: np.ndarray | None = None
        self._prev_socket_quat: np.ndarray | None = None
        self._prev_hydro_target_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._prev_hydro_relative_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def active(self) -> bool:
        return self._file is not None

    @property
    def num_frames(self) -> int:
        return self._frames

    @property
    def num_rows(self) -> int:
        return self._rows

    @property
    def path(self) -> str | None:
        return self._path

    def seed_reference_poses(
        self,
        *,
        left_target_pos: np.ndarray | None = None,
        right_target_pos: np.ndarray | None = None,
        left_ee_pos: np.ndarray | None = None,
        right_ee_pos: np.ndarray | None = None,
        plug_pose: np.ndarray | None = None,
        socket_pose: np.ndarray | None = None,
        hydro_target_poses_w: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        hydro_relative_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        def vec3(value) -> np.ndarray | None:
            if value is None:
                return None
            return np.asarray(value, dtype=np.float64).reshape(-1)[:3].copy()

        def quat4(value) -> np.ndarray | None:
            if value is None:
                return None
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size < 7:
                return None
            return _normalize_quat_wxyz_np(arr[3:7])

        self._prev_left_target_pos = vec3(left_target_pos)
        self._prev_right_target_pos = vec3(right_target_pos)
        self._prev_left_ee_pos = vec3(left_ee_pos)
        self._prev_right_ee_pos = vec3(right_ee_pos)
        self._prev_plug_pos = vec3(plug_pose)
        self._prev_socket_pos = vec3(socket_pose)
        self._prev_plug_quat = quat4(plug_pose)
        self._prev_socket_quat = quat4(socket_pose)
        self._prev_hydro_target_poses = {
            int(pad): (pos.copy(), quat.copy())
            for pad, (pos, quat) in (hydro_target_poses_w or {}).items()
        }
        self._prev_hydro_relative_poses = {
            int(pad): (pos.copy(), quat.copy())
            for pad, (pos, quat) in (hydro_relative_poses or {}).items()
        }

    def start(self) -> str:
        if self.active:
            return str(self._path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"tactile_shear_{self._file_count:04d}_{ts}.csv"
        self._path = os.path.join(self._save_dir, fname)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "frame",
                "sim_step",
                "pad",
                "pad_label",
                "row",
                "col",
                "shear_x",
                "shear_y",
                "normal_force",
                "arm",
                "target_dx_w",
                "target_dy_w",
                "target_dz_w",
                "ee_dx_w",
                "ee_dy_w",
                "ee_dz_w",
                "pad_axis_x_wx",
                "pad_axis_x_wy",
                "pad_axis_x_wz",
                "pad_axis_y_wx",
                "pad_axis_y_wy",
                "pad_axis_y_wz",
                "pad_axis_n_wx",
                "pad_axis_n_wy",
                "pad_axis_n_wz",
                "target_delta_shear_x",
                "target_delta_shear_y",
                "ee_delta_shear_x",
                "ee_delta_shear_y",
                "target_delta_normal",
                "ee_delta_normal",
                "commanded_pad",
                "commanded_axis",
                "trial_phase",
                "command_dx_w",
                "command_dy_w",
                "command_dz_w",
                "command_delta_shear_x",
                "command_delta_shear_y",
                "command_delta_normal",
                "object_name",
                "object_dx_w",
                "object_dy_w",
                "object_dz_w",
                "object_delta_shear_x",
                "object_delta_shear_y",
                "object_delta_normal",
                "object_rot_angle_rad",
                "object_command_dx_w",
                "object_command_dy_w",
                "object_command_dz_w",
                "object_command_delta_shear_x",
                "object_command_delta_shear_y",
                "object_command_delta_normal",
                "hydro_target_dx_w",
                "hydro_target_dy_w",
                "hydro_target_dz_w",
                "hydro_target_delta_shear_x",
                "hydro_target_delta_shear_y",
                "hydro_target_delta_normal",
                "hydro_target_rot_angle_rad",
                "hydro_rel_delta_shear_x",
                "hydro_rel_delta_shear_y",
                "hydro_rel_delta_normal",
                "hydro_rel_rot_angle_rad",
            ]
        )
        self._frames = 0
        self._rows = 0
        self._prev_left_target_pos = None
        self._prev_right_target_pos = None
        self._prev_left_ee_pos = None
        self._prev_right_ee_pos = None
        self._prev_plug_pos = None
        self._prev_socket_pos = None
        self._prev_plug_quat = None
        self._prev_socket_quat = None
        self._prev_hydro_target_poses = {}
        self._prev_hydro_relative_poses = {}
        self._file_count += 1
        print(f"[SAVE] Tactile shear CSV started: {self._path}", flush=True)
        return self._path

    def record(
        self,
        obs: dict,
        sim_step: int,
        *,
        left_target_pos: np.ndarray | None = None,
        right_target_pos: np.ndarray | None = None,
        left_ee_pos: np.ndarray | None = None,
        right_ee_pos: np.ndarray | None = None,
        pad_axes_w: dict[int, dict[str, np.ndarray]] | None = None,
        commanded_left_delta_w: np.ndarray | None = None,
        commanded_right_delta_w: np.ndarray | None = None,
        commanded_pad: int | None = None,
        commanded_axis: str = "",
        trial_phase: str = "",
        plug_pose: np.ndarray | None = None,
        socket_pose: np.ndarray | None = None,
        commanded_object_name: str = "",
        commanded_object_delta_w: np.ndarray | None = None,
        hydro_target_poses_w: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        hydro_relative_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        if not self.active or self._writer is None:
            return
        sim_step = int(sim_step)
        if sim_step % self._interval != 0:
            return
        shear = tactile_shear_uv(obs.get("tactile_shear"))
        if shear is None:
            return
        shear = np.asarray(shear)
        if shear.ndim != 4 or shear.shape[-1] != 2:
            return
        tactile_force = np.asarray(obs.get("tactile_force")) if obs.get("tactile_force") is not None else None
        normal_force = None
        if tactile_force is not None and tactile_force.shape[:3] == shear.shape[:3] and tactile_force.shape[-1] >= 1:
            normal_force = tactile_force[..., 0]

        def vec3(value) -> np.ndarray | None:
            if value is None:
                return None
            return np.asarray(value, dtype=np.float64).reshape(3)

        left_target = vec3(left_target_pos)
        right_target = vec3(right_target_pos)
        left_ee = vec3(left_ee_pos)
        right_ee = vec3(right_ee_pos)
        commanded_left_delta = (
            np.zeros(3, dtype=np.float64)
            if commanded_left_delta_w is None
            else np.asarray(commanded_left_delta_w, dtype=np.float64).reshape(3)
        )
        commanded_right_delta = (
            np.zeros(3, dtype=np.float64)
            if commanded_right_delta_w is None
            else np.asarray(commanded_right_delta_w, dtype=np.float64).reshape(3)
        )
        plug_pos = None if plug_pose is None else np.asarray(plug_pose, dtype=np.float64).reshape(-1)[:3]
        socket_pos = None if socket_pose is None else np.asarray(socket_pose, dtype=np.float64).reshape(-1)[:3]
        plug_quat = None
        socket_quat = None
        if plug_pose is not None and np.asarray(plug_pose).size >= 7:
            plug_quat = _normalize_quat_wxyz_np(np.asarray(plug_pose, dtype=np.float64).reshape(-1)[3:7])
        if socket_pose is not None and np.asarray(socket_pose).size >= 7:
            socket_quat = _normalize_quat_wxyz_np(np.asarray(socket_pose, dtype=np.float64).reshape(-1)[3:7])
        commanded_object_delta = (
            np.zeros(3, dtype=np.float64)
            if commanded_object_delta_w is None
            else np.asarray(commanded_object_delta_w, dtype=np.float64).reshape(3)
        )

        left_target_delta = np.zeros(3, dtype=np.float64) if self._prev_left_target_pos is None or left_target is None else left_target - self._prev_left_target_pos
        right_target_delta = np.zeros(3, dtype=np.float64) if self._prev_right_target_pos is None or right_target is None else right_target - self._prev_right_target_pos
        left_ee_delta = np.zeros(3, dtype=np.float64) if self._prev_left_ee_pos is None or left_ee is None else left_ee - self._prev_left_ee_pos
        right_ee_delta = np.zeros(3, dtype=np.float64) if self._prev_right_ee_pos is None or right_ee is None else right_ee - self._prev_right_ee_pos
        plug_delta = np.zeros(3, dtype=np.float64) if self._prev_plug_pos is None or plug_pos is None else plug_pos - self._prev_plug_pos
        socket_delta = np.zeros(3, dtype=np.float64) if self._prev_socket_pos is None or socket_pos is None else socket_pos - self._prev_socket_pos
        plug_rot_angle = _quat_angle_between_wxyz(self._prev_plug_quat, plug_quat)
        socket_rot_angle = _quat_angle_between_wxyz(self._prev_socket_quat, socket_quat)

        hydro_target_curr = {
            int(pad): (np.asarray(pos, dtype=np.float64).reshape(3), _normalize_quat_wxyz_np(quat))
            for pad, (pos, quat) in (hydro_target_poses_w or {}).items()
        }
        hydro_target_delta_by_pad: dict[int, np.ndarray] = {}
        hydro_target_rot_by_pad: dict[int, float] = {}
        for pad, (pos, quat) in hydro_target_curr.items():
            prev_pose = self._prev_hydro_target_poses.get(pad)
            if prev_pose is None:
                hydro_target_delta_by_pad[pad] = np.zeros(3, dtype=np.float64)
                hydro_target_rot_by_pad[pad] = 0.0
            else:
                prev_pos, prev_quat = prev_pose
                hydro_target_delta_by_pad[pad] = pos - prev_pos
                hydro_target_rot_by_pad[pad] = _quat_angle_between_wxyz(prev_quat, quat)

        hydro_rel_curr = {
            int(pad): (np.asarray(pos, dtype=np.float64).reshape(3), _normalize_quat_wxyz_np(quat))
            for pad, (pos, quat) in (hydro_relative_poses or {}).items()
        }
        hydro_rel_delta_by_pad: dict[int, np.ndarray] = {}
        hydro_rel_rot_by_pad: dict[int, float] = {}
        for pad, (pos, quat) in hydro_rel_curr.items():
            prev_pose = self._prev_hydro_relative_poses.get(pad)
            if prev_pose is None:
                hydro_rel_delta_by_pad[pad] = np.zeros(3, dtype=np.float64)
                hydro_rel_rot_by_pad[pad] = 0.0
            else:
                prev_pos, prev_quat = prev_pose
                hydro_rel_delta_by_pad[pad] = pos - prev_pos
                hydro_rel_rot_by_pad[pad] = _quat_angle_between_wxyz(prev_quat, quat)

        self._prev_left_target_pos = None if left_target is None else left_target.copy()
        self._prev_right_target_pos = None if right_target is None else right_target.copy()
        self._prev_left_ee_pos = None if left_ee is None else left_ee.copy()
        self._prev_right_ee_pos = None if right_ee is None else right_ee.copy()
        self._prev_plug_pos = None if plug_pos is None else plug_pos.copy()
        self._prev_socket_pos = None if socket_pos is None else socket_pos.copy()
        self._prev_plug_quat = None if plug_quat is None else plug_quat.copy()
        self._prev_socket_quat = None if socket_quat is None else socket_quat.copy()
        self._prev_hydro_target_poses = {
            pad: (pos.copy(), quat.copy()) for pad, (pos, quat) in hydro_target_curr.items()
        }
        self._prev_hydro_relative_poses = {
            pad: (pos.copy(), quat.copy()) for pad, (pos, quat) in hydro_rel_curr.items()
        }

        frame = self._frames
        axis_u, axis_v = _tangential_axes(self._normal_axis)
        rows_out = []
        for pad in range(shear.shape[0]):
            pad_label = self._pad_labels[pad] if pad < len(self._pad_labels) else str(pad)
            arm = "left" if pad < 2 else "right"
            target_delta = left_target_delta if arm == "left" else right_target_delta
            ee_delta = left_ee_delta if arm == "left" else right_ee_delta
            command_delta = commanded_left_delta if arm == "left" else commanded_right_delta
            object_name = "socket" if arm == "left" else "plug"
            object_delta = socket_delta if arm == "left" else plug_delta
            object_command_delta = (
                commanded_object_delta
                if str(commanded_object_name) == object_name
                else np.zeros(3, dtype=np.float64)
            )
            pad_axes = (pad_axes_w or {}).get(pad, {})
            axis_x_w = np.asarray(pad_axes.get("axis_x_w", np.zeros(3)), dtype=np.float64).reshape(3)
            axis_y_w = np.asarray(pad_axes.get("axis_y_w", np.zeros(3)), dtype=np.float64).reshape(3)
            axis_n_w = np.asarray(pad_axes.get("normal_w", np.zeros(3)), dtype=np.float64).reshape(3)
            target_delta_shear_x = float(np.dot(target_delta, axis_x_w))
            target_delta_shear_y = float(np.dot(target_delta, axis_y_w))
            ee_delta_shear_x = float(np.dot(ee_delta, axis_x_w))
            ee_delta_shear_y = float(np.dot(ee_delta, axis_y_w))
            target_delta_normal = float(np.dot(target_delta, axis_n_w))
            ee_delta_normal = float(np.dot(ee_delta, axis_n_w))
            command_delta_shear_x = float(np.dot(command_delta, axis_x_w))
            command_delta_shear_y = float(np.dot(command_delta, axis_y_w))
            command_delta_normal = float(np.dot(command_delta, axis_n_w))
            object_delta_shear_x = float(np.dot(object_delta, axis_x_w))
            object_delta_shear_y = float(np.dot(object_delta, axis_y_w))
            object_delta_normal = float(np.dot(object_delta, axis_n_w))
            object_command_delta_shear_x = float(np.dot(object_command_delta, axis_x_w))
            object_command_delta_shear_y = float(np.dot(object_command_delta, axis_y_w))
            object_command_delta_normal = float(np.dot(object_command_delta, axis_n_w))
            object_rot_angle = socket_rot_angle if object_name == "socket" else plug_rot_angle
            hydro_target_delta = hydro_target_delta_by_pad.get(pad, np.zeros(3, dtype=np.float64))
            hydro_target_delta_shear_x = float(np.dot(hydro_target_delta, axis_x_w))
            hydro_target_delta_shear_y = float(np.dot(hydro_target_delta, axis_y_w))
            hydro_target_delta_normal = float(np.dot(hydro_target_delta, axis_n_w))
            hydro_target_rot_angle = float(hydro_target_rot_by_pad.get(pad, 0.0))
            hydro_rel_delta = hydro_rel_delta_by_pad.get(pad, np.zeros(3, dtype=np.float64))
            hydro_rel_delta_shear_x = float(hydro_rel_delta[axis_u])
            hydro_rel_delta_shear_y = float(hydro_rel_delta[axis_v])
            hydro_rel_delta_normal = float(hydro_rel_delta[int(self._normal_axis)])
            hydro_rel_rot_angle = float(hydro_rel_rot_by_pad.get(pad, 0.0))
            for row in range(shear.shape[1]):
                for col in range(shear.shape[2]):
                    rows_out.append(
                        [
                            frame,
                            sim_step,
                            pad,
                            pad_label,
                            row,
                            col,
                            f"{float(shear[pad, row, col, 0]):.17g}",
                            f"{float(shear[pad, row, col, 1]):.17g}",
                            f"{float(normal_force[pad, row, col] if normal_force is not None else 0.0):.17g}",
                            arm,
                            f"{float(target_delta[0]):.17g}",
                            f"{float(target_delta[1]):.17g}",
                            f"{float(target_delta[2]):.17g}",
                            f"{float(ee_delta[0]):.17g}",
                            f"{float(ee_delta[1]):.17g}",
                            f"{float(ee_delta[2]):.17g}",
                            f"{float(axis_x_w[0]):.17g}",
                            f"{float(axis_x_w[1]):.17g}",
                            f"{float(axis_x_w[2]):.17g}",
                            f"{float(axis_y_w[0]):.17g}",
                            f"{float(axis_y_w[1]):.17g}",
                            f"{float(axis_y_w[2]):.17g}",
                            f"{float(axis_n_w[0]):.17g}",
                            f"{float(axis_n_w[1]):.17g}",
                            f"{float(axis_n_w[2]):.17g}",
                            f"{target_delta_shear_x:.17g}",
                            f"{target_delta_shear_y:.17g}",
                            f"{ee_delta_shear_x:.17g}",
                            f"{ee_delta_shear_y:.17g}",
                            f"{target_delta_normal:.17g}",
                            f"{ee_delta_normal:.17g}",
                            "" if commanded_pad is None else int(commanded_pad),
                            str(commanded_axis),
                            str(trial_phase),
                            f"{float(command_delta[0]):.17g}",
                            f"{float(command_delta[1]):.17g}",
                            f"{float(command_delta[2]):.17g}",
                            f"{command_delta_shear_x:.17g}",
                            f"{command_delta_shear_y:.17g}",
                            f"{command_delta_normal:.17g}",
                            object_name,
                            f"{float(object_delta[0]):.17g}",
                            f"{float(object_delta[1]):.17g}",
                            f"{float(object_delta[2]):.17g}",
                            f"{object_delta_shear_x:.17g}",
                            f"{object_delta_shear_y:.17g}",
                            f"{object_delta_normal:.17g}",
                            f"{object_rot_angle:.17g}",
                            f"{float(object_command_delta[0]):.17g}",
                            f"{float(object_command_delta[1]):.17g}",
                            f"{float(object_command_delta[2]):.17g}",
                            f"{object_command_delta_shear_x:.17g}",
                            f"{object_command_delta_shear_y:.17g}",
                            f"{object_command_delta_normal:.17g}",
                            f"{float(hydro_target_delta[0]):.17g}",
                            f"{float(hydro_target_delta[1]):.17g}",
                            f"{float(hydro_target_delta[2]):.17g}",
                            f"{hydro_target_delta_shear_x:.17g}",
                            f"{hydro_target_delta_shear_y:.17g}",
                            f"{hydro_target_delta_normal:.17g}",
                            f"{hydro_target_rot_angle:.17g}",
                            f"{hydro_rel_delta_shear_x:.17g}",
                            f"{hydro_rel_delta_shear_y:.17g}",
                            f"{hydro_rel_delta_normal:.17g}",
                            f"{hydro_rel_rot_angle:.17g}",
                        ]
                    )
        self._writer.writerows(rows_out)
        self._frames += 1
        self._rows += len(rows_out)
        if self._frames % 30 == 0 and self._file is not None:
            self._file.flush()

    def stop(self) -> str | None:
        if not self.active:
            return self._path
        assert self._file is not None
        self._file.flush()
        self._file.close()
        self._file = None
        self._writer = None
        path = self._path
        print(f"[SAVE] Tactile shear CSV saved: {path} ({self._frames} frames, {self._rows} rows)", flush=True)
        return path


# ---------------------------------------------------------------------------
# Viser setup
# ---------------------------------------------------------------------------

def create_viser_server(port: int, cfg=None):
    try:
        import viser
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")).startswith("websockets"):
            raise ModuleNotFoundError(
                "Viser requires websockets.asyncio.server. Install a compatible version with: "
                'pip install -U "websockets>=13"'
            ) from e
        raise

    print(f"[viser] Creating ViserServer on port {port}...", flush=True)
    server = viser.ViserServer(port=port)
    print(f"[viser] *** Open http://localhost:{port} in your browser ***", flush=True)

    if cfg is not None and hasattr(server, "initial_camera"):
        try:
            eye = np.asarray(cfg.camera.eye, dtype=np.float64).reshape(3)
            target = np.asarray(cfg.camera.target, dtype=np.float64).reshape(3)
            server.initial_camera.position = eye
            server.initial_camera.look_at = target
            server.initial_camera.up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            print(
                f"[viser] Initial browser camera set: eye={eye.tolist()} target={target.tolist()}",
                flush=True,
            )
        except Exception as e:
            print(f"[viser] WARNING: Failed to set initial browser camera: {e}", flush=True)

    with server.gui.add_folder("Left Gripper"):
        left_btns = server.gui.add_button_group("Left gripper", options=["Open", "Close"])
        left_slider = server.gui.add_slider("Left fine", min=0.0, max=1.0, step=0.01, initial_value=1.0)

    with server.gui.add_folder("Right Gripper"):
        right_btns = server.gui.add_button_group("Right gripper", options=["Open", "Close"])
        right_slider = server.gui.add_slider("Right fine", min=0.0, max=1.0, step=0.01, initial_value=1.0)

    @left_btns.on_click
    def _(event):
        cur = shared.get_gripper("left")
        v = min(1.0, cur + 0.1) if event.target.value == "Open" else max(0.0, cur - 0.1)
        shared.set_gripper("left", v)
        left_slider.value = v

    @right_btns.on_click
    def _(event):
        cur = shared.get_gripper("right")
        v = min(1.0, cur + 0.1) if event.target.value == "Open" else max(0.0, cur - 0.1)
        shared.set_gripper("right", v)
        right_slider.value = v

    @left_slider.on_update
    def _(_):
        shared.set_gripper("left", left_slider.value)

    @right_slider.on_update
    def _(_):
        shared.set_gripper("right", right_slider.value)

    reset_btn = server.gui.add_button("Reset", color="red")

    @reset_btn.on_click
    def _(_):
        shared.request_reset()

    record_btn = server.gui.add_button("Start Recording", color="green")

    @record_btn.on_click
    def _(_):
        is_recording = shared.is_recording()
        shared.set_recording(not is_recording)
        record_btn.name = "Start Recording" if is_recording else "Stop Recording"
        record_btn.color = "green" if is_recording else "orange"

    shear_csv_btn = server.gui.add_button("Start Shear CSV", color="blue")

    @shear_csv_btn.on_click
    def _(_):
        is_recording = shared.is_shear_csv_recording()
        shared.set_shear_csv_recording(not is_recording)
        shear_csv_btn.name = "Start Shear CSV" if is_recording else "Stop Shear CSV"
        shear_csv_btn.color = "blue" if is_recording else "orange"

    with server.gui.add_folder("Object Slide CSV", expand_by_default=False):
        object_total_move_mm = server.gui.add_slider(
            "Total move (mm)",
            min=0.05,
            max=200.0,
            step=0.05,
            initial_value=2.0,
        )
        object_step_mm = server.gui.add_slider("Step (mm)", min=0.05, max=20.0, step=0.05, initial_value=0.5)
        object_settle_steps = server.gui.add_slider(
            "Settle steps",
            min=0,
            max=240,
            step=1,
            initial_value=30,
        )
        object_hold_steps = server.gui.add_slider(
            "Hold steps",
            min=1,
            max=240,
            step=1,
            initial_value=40,
        )
        object_auto_place = server.gui.add_checkbox("Auto-place", initial_value=True)
        object_preload_mm = server.gui.add_slider(
            "Preload (mm)",
            min=0.0,
            max=5.0,
            step=0.05,
            initial_value=1.0,
        )
        object_pad2_x = server.gui.add_button("Pad2 +shear_x CSV")
        object_pad2_y = server.gui.add_button("Pad2 +shear_y CSV")
        object_pad3_x = server.gui.add_button("Pad3 +shear_x CSV")
        object_pad3_y = server.gui.add_button("Pad3 +shear_y CSV")

        for button, pad, axis in (
            (object_pad2_x, 2, "x"),
            (object_pad2_y, 2, "y"),
            (object_pad3_x, 3, "x"),
            (object_pad3_y, 3, "y"),
        ):
            @button.on_click
            def _(_, pad=pad, axis=axis):
                shared.request_object_slide_trial(
                    pad,
                    axis,
                    1.0,
                    float(object_total_move_mm.value) * 1.0e-3,
                    float(object_step_mm.value) * 1.0e-3,
                    int(object_settle_steps.value),
                    int(object_hold_steps.value),
                    bool(object_auto_place.value),
                    float(object_preload_mm.value) * 1.0e-3,
                )

    hydro_enabled = getattr(cfg.tactile, "enable_hydro_normal_observation", False)
    main_shear_enabled = bool(getattr(cfg.tactile.backend, "include_force_observations", False))
    dummy_img = np.zeros((96, 256, 3), dtype=np.uint8)
    pad_labels = [
        "Left arm / left finger",
        "Left arm / right finger",
        "Right arm / left finger",
        "Right arm / right finger",
    ]
    with server.gui.add_folder("Tactile Normal"):
        tac_handles = [
            server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
            for label in pad_labels
        ]
    hydro_handles = []
    main_force_normal_handles = []
    main_shear_x_handles = []
    main_shear_y_handles = []
    main_shear_mag_handles = []
    hydro_shear_x_handles = []
    hydro_shear_y_handles = []
    hydro_shear_mag_handles = []
    marker_shear_x_handles = []
    marker_shear_y_handles = []
    marker_shear_mag_handles = []
    marker_combined_x_handles = []
    marker_combined_y_handles = []
    marker_combined_mag_handles = []
    if main_shear_enabled:
        with server.gui.add_folder("Force Normal"):
            main_force_normal_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Force Shear X"):
            main_shear_x_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Force Shear Y"):
            main_shear_y_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Force Shear Magnitude"):
            main_shear_mag_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Shear X"):
            marker_shear_x_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Shear Y"):
            marker_shear_y_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Shear Magnitude"):
            marker_shear_mag_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Combined X"):
            marker_combined_x_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Combined Y"):
            marker_combined_y_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Marker Combined Magnitude"):
            marker_combined_mag_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
    if hydro_enabled:
        with server.gui.add_folder("Hydro Normal"):
            hydro_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Hydro Shear X"):
            hydro_shear_x_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Hydro Shear Y"):
            hydro_shear_y_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]
        with server.gui.add_folder("Hydro Shear Magnitude"):
            hydro_shear_mag_handles = [
                server.gui.add_image(dummy_img, label=label, format="jpeg", jpeg_quality=80)
                for label in pad_labels
            ]

    with server.gui.add_folder("State", expand_by_default=False):
        state_md = server.gui.add_markdown("**Loading...**")
    with server.gui.add_folder("HydroShear Debug", expand_by_default=True):
        hydro_debug_md = server.gui.add_markdown(
            "**Vectors:** blue=command, orange=HydroShear relative motion, purple=force shear."
        )

    with server.gui.add_folder("Sim Camera", expand_by_default=False):
        camera_img = server.gui.add_image(
            np.zeros((480, 640, 3), dtype=np.uint8),
            label="Camera Feed", format="jpeg", jpeg_quality=70,
        )

    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, -0.05))
    hydro_debug_arrows = server.scene.add_arrows(
        "/hydroshear_debug_vectors",
        points=np.zeros((3, 2, 3), dtype=np.float32),
        colors=np.array(
            [
                [0, 120, 255],    # commanded motion
                [255, 120, 0],    # HydroShear relative motion
                [180, 80, 255],   # final force shear readout
            ],
            dtype=np.uint8,
        ),
        shaft_radius=0.0022,
        head_radius=0.0055,
        head_length=0.012,
        visible=False,
    )

    handles = {
        "server": server,
        "viser_urdf": None,
        "plug_mesh": None,
        "socket_mesh": None,
        "left_ee_gizmo": None,
        "right_ee_gizmo": None,
        "left_gripper_slider": left_slider,
        "right_gripper_slider": right_slider,
        "record_btn": record_btn,
        "shear_csv_btn": shear_csv_btn,
        "hydro_debug_arrows": hydro_debug_arrows,
        "tac_handles": tac_handles,
        "hydro_handles": hydro_handles,
        "main_force_normal_handles": main_force_normal_handles,
        "main_shear_x_handles": main_shear_x_handles,
        "main_shear_y_handles": main_shear_y_handles,
        "main_shear_mag_handles": main_shear_mag_handles,
        "hydro_shear_x_handles": hydro_shear_x_handles,
        "hydro_shear_y_handles": hydro_shear_y_handles,
        "hydro_shear_mag_handles": hydro_shear_mag_handles,
        "marker_shear_x_handles": marker_shear_x_handles,
        "marker_shear_y_handles": marker_shear_y_handles,
        "marker_shear_mag_handles": marker_shear_mag_handles,
        "marker_combined_x_handles": marker_combined_x_handles,
        "marker_combined_y_handles": marker_combined_y_handles,
        "marker_combined_mag_handles": marker_combined_mag_handles,
        "state_md": state_md,
        "hydro_debug_md": hydro_debug_md,
        "camera_img": camera_img,
    }
    return server, handles


def load_scene_objects(server, handles, cfg):
    from viser.extras import ViserUrdf

    robot_cfg = cfg.robot
    objects_cfg = cfg.objects

    urdf_path = Path(os.path.expanduser(robot_cfg.urdf_path))
    print(f"[viser] Loading URDF from {urdf_path}...", flush=True)
    try:
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )
        handles["viser_urdf"] = viser_urdf
        print("[viser] URDF loaded.", flush=True)
    except Exception as e:
        print(f"[viser] WARNING: URDF load failed: {e}", flush=True)

    if bool(getattr(objects_cfg, "use_cube_objects", False)):
        import trimesh

        def pose_wxyz(pose):
            return (float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5]))

        def add_cube(name: str, pose, color):
            mesh = trimesh.creation.box(extents=tuple(float(v) for v in objects_cfg.cube_size))
            mesh.visual.face_colors = color
            return server.scene.add_mesh_trimesh(
                name,
                mesh,
                position=tuple(float(v) for v in pose[:3]),
                wxyz=pose_wxyz(pose),
            )

        if objects_cfg.enable_plug:
            handles["plug_mesh"] = add_cube("/plug_cube", objects_cfg.plug_default_pose, (48, 96, 230, 230))
            print("[viser] Plug cube loaded.", flush=True)
        if objects_cfg.enable_socket:
            handles["socket_mesh"] = add_cube("/socket_cube", objects_cfg.socket_default_pose, (230, 128, 48, 230))
            print("[viser] Socket cube loaded.", flush=True)
        return

    asset_root = os.path.expanduser(objects_cfg.asset_root)
    automate_dir = os.path.join(asset_root, "automate_scaled", "urdf")

    if objects_cfg.enable_plug:
        import trimesh

        plug_urdf = os.path.join(automate_dir, f"{objects_cfg.automate_asset_id}_plug.urdf")
        try:
            result = _extract_mesh_path_from_urdf(plug_urdf)
            if result:
                mesh_path, scale = result
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh.apply_scale(scale)
                handles["plug_mesh"] = server.scene.add_mesh_trimesh(
                    "/plug",
                    mesh,
                    position=(0.0, 0.05, 0.0175),
                    wxyz=(0.5, -0.5, -0.5, -0.5),
                )
                print("[viser] Plug mesh loaded.", flush=True)
        except Exception as e:
            print(f"[viser] WARNING: Plug mesh load failed: {e}", flush=True)

    if objects_cfg.enable_socket:
        import trimesh

        socket_urdf = os.path.join(automate_dir, f"{objects_cfg.automate_asset_id}_socket.urdf")
        try:
            result = _extract_mesh_path_from_urdf(socket_urdf)
            if result:
                mesh_path, scale = result
                mesh = trimesh.load(mesh_path, force="mesh")
                mesh.apply_scale(scale)
                handles["socket_mesh"] = server.scene.add_mesh_trimesh(
                    "/socket",
                    mesh,
                    position=(0.0, -0.05, 0.123),
                    wxyz=(0.0, 0.0, 0.0, 1.0),
                )
                print("[viser] Socket mesh loaded.", flush=True)
        except Exception as e:
            print(f"[viser] WARNING: Socket mesh load failed: {e}", flush=True)


def create_ee_gizmo(server, handles, arm: str, ee_pos, ee_quat):
    label = f"/{arm}_ee_target"
    ee_gizmo = server.scene.add_transform_controls(
        label,
        scale=0.15,
        disable_axes=False,
        disable_sliders=False,
        disable_rotations=False,
        position=tuple(float(v) for v in ee_pos),
        wxyz=tuple(float(v) for v in ee_quat),
    )

    @ee_gizmo.on_update
    def _(event):
        pos = event.target.position
        quat = event.target.wxyz
        shared.set_target(arm, float(pos[0]), float(pos[1]), float(pos[2]))
        shared.set_target_quat(arm, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    @ee_gizmo.on_drag_start
    def _(_):
        shared.set_dragging(arm, True)

    @ee_gizmo.on_drag_end
    def _(_):
        shared.set_dragging(arm, False)

    handles[f"{arm}_ee_gizmo"] = ee_gizmo
    shared.set_target(arm, *[float(v) for v in ee_pos])
    shared.set_target_quat(arm, *[float(v) for v in ee_quat])


def _unit_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return vec / norm


def _ratio_text(primary: float, secondary: float) -> str:
    primary = abs(float(primary))
    secondary = abs(float(secondary))
    if primary <= 1.0e-30 and secondary <= 1.0e-30:
        return "0"
    return f"{primary / (secondary + 1.0e-30):.1f}:1"


def _pad_force_shear_vector_w(obs, pad: int, axes: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[float, float]]:
    shear_uv = tactile_shear_uv(obs.get("tactile_shear"))
    if shear_uv is None or pad < 0 or pad >= int(shear_uv.shape[0]):
        return np.zeros(3, dtype=np.float64), (0.0, 0.0)

    grid = np.asarray(shear_uv[pad], dtype=np.float64)
    if grid.ndim != 3 or grid.shape[-1] < 2:
        return np.zeros(3, dtype=np.float64), (0.0, 0.0)

    mask = None
    tactile_force = obs.get("tactile_force")
    if tactile_force is not None:
        force = np.asarray(tactile_force)
        if force.ndim >= 4 and pad < int(force.shape[0]) and force.shape[-1] >= 1:
            normal = np.asarray(force[pad, ..., 0], dtype=np.float64)
            if normal.shape == grid.shape[:2]:
                mask = np.abs(normal) > 1.0e-12

    if mask is not None and np.any(mask):
        sx = float(np.mean(grid[..., 0][mask]))
        sy = float(np.mean(grid[..., 1][mask]))
    else:
        sx = float(np.mean(grid[..., 0]))
        sy = float(np.mean(grid[..., 1]))

    axis_x = _unit_vec(axes.get("axis_x_w", np.zeros(3)))
    axis_y = _unit_vec(axes.get("axis_y_w", np.zeros(3)))
    return axis_x * sx + axis_y * sy, (sx, sy)


def update_hydroshear_debug_vectors(
    handles,
    obs,
    pad_axes_w: dict[int, dict[str, np.ndarray]],
    *,
    pad: int | None,
    command_delta_w: np.ndarray,
    hydro_rel_delta_p: np.ndarray,
    visible: bool = True,
) -> str:
    handle = handles.get("hydro_debug_arrows")
    if handle is None or pad is None:
        if handle is not None:
            try:
                handle.visible = False
            except Exception:
                pass
        return "disabled"

    axes = pad_axes_w.get(int(pad))
    if axes is None:
        try:
            handle.visible = False
        except Exception:
            pass
        return f"pad{pad}: no pad axes"

    origin = np.asarray(axes.get("position_w", np.zeros(3)), dtype=np.float64).reshape(3)
    axis_x = _unit_vec(axes.get("axis_x_w", np.zeros(3)))
    axis_y = _unit_vec(axes.get("axis_y_w", np.zeros(3)))
    normal = _unit_vec(axes.get("normal_w", np.zeros(3)))
    origin = origin + normal * 0.018

    command_delta_w = np.asarray(command_delta_w, dtype=np.float64).reshape(3)
    hydro_rel_delta_p = np.asarray(hydro_rel_delta_p, dtype=np.float64).reshape(3)
    hydro_delta_w = (
        axis_x * float(hydro_rel_delta_p[0])
        + axis_y * float(hydro_rel_delta_p[1])
        + normal * float(hydro_rel_delta_p[2])
    )
    force_shear_w, force_shear_xy = _pad_force_shear_vector_w(obs, int(pad), axes)

    command_xy = (float(np.dot(command_delta_w, axis_x)), float(np.dot(command_delta_w, axis_y)))
    hydro_xy = (float(hydro_rel_delta_p[0]), float(hydro_rel_delta_p[1]))

    vectors = [command_delta_w, hydro_delta_w, force_shear_w]
    scales = [80.0, 80.0, 2500.0]
    offsets = [0.0, 0.012, 0.024]
    points = np.zeros((3, 2, 3), dtype=np.float32)
    for i, (vec, scale, offset) in enumerate(zip(vectors, scales, offsets, strict=True)):
        start = origin + normal * offset
        end = start + np.asarray(vec, dtype=np.float64).reshape(3) * scale
        points[i, 0] = start.astype(np.float32)
        points[i, 1] = end.astype(np.float32)

    try:
        handle.points = points
        handle.visible = bool(visible)
    except Exception:
        pass

    command_ratio = _ratio_text(command_xy[0], command_xy[1])
    hydro_ratio = _ratio_text(hydro_xy[0], hydro_xy[1])
    force_ratio = _ratio_text(force_shear_xy[0], force_shear_xy[1])
    return (
        f"pad{pad} vectors: blue=command ({command_xy[0]:.2e},{command_xy[1]:.2e}, x/y {command_ratio}); "
        f"orange=hydro_rel ({hydro_xy[0]:.2e},{hydro_xy[1]:.2e}, x/y {hydro_ratio}); "
        f"purple=force_shear ({force_shear_xy[0]:.2e},{force_shear_xy[1]:.2e}, x/y {force_ratio})"
    )


def reset_tactile_state(env) -> None:
    setup = getattr(env, "_tactile_setup", None)
    if setup is None:
        return
    try:
        setup.reset()
    except Exception as e:
        print(f"[WARN] Failed to reset tactile state: {e}", flush=True)


def zero_hydroshear_shear_preserve_normal(env) -> int:
    setup = getattr(env, "_tactile_setup", None)
    if setup is None:
        return 0
    sensors = getattr(setup, "sensors", None) or []
    count = 0
    for state in sensors:
        cores = getattr(state, "cores", None)
        outputs = getattr(state, "last_outputs", None)
        if outputs is None:
            last_output = getattr(state, "last_output", None)
            outputs = [last_output] if last_output is not None else []
        if cores is None:
            core = getattr(state, "core", None)
            cores = [core] if core is not None else []
        for core, out in zip(cores, outputs, strict=False):
            if core is None or out is None or getattr(core, "tracker", None) is None:
                continue
            surface_state = getattr(out.surface, "state", None)
            if surface_state is None:
                continue
            core.tracker.state = type(surface_state)(
                prev_points_e=out.contact.points_e.detach().clone(),
                prev_sdf=out.contact.sdf.detach().clone(),
                normal_displacement=out.surface.normal_displacement.detach().clone(),
                shear_displacement_e=torch.zeros_like(out.surface.shear_displacement_e),
                initialized=torch.ones_like(out.contact.sdf, dtype=torch.bool),
            )
            count += 1
    return count


def hydroshear_elastomer_source_summary(env) -> str:
    setup = getattr(env, "_tactile_setup", None)
    backend = getattr(setup, "backend", None)
    sensors = getattr(setup, "sensors", None)
    if not sensors or backend is None or not hasattr(backend, "_patch_world_pose"):
        return "disabled"

    parts = []
    for i, state in enumerate(sensors):
        slot = getattr(state, "slot", i)
        vertices = getattr(state, "elastomer_vertices_p", None)
        faces = getattr(state, "elastomer_faces", None)
        if vertices is not None and faces is not None:
            try:
                source = f"mesh(v={int(vertices.shape[0])},f={int(faces.shape[0])})"
            except Exception:
                source = "mesh"
        else:
            source = "flat"
        parts.append(f"pad{int(slot)}={source}")
    return ", ".join(parts) if parts else "disabled"


def update_scene_fast(handles, obs, left_ee_pos, left_ee_quat, right_ee_pos, right_ee_quat, joint_name_map, urdf_joint_names):
    viser_urdf = handles["viser_urdf"]
    joint_pos = obs["joint_pos"]
    if viser_urdf is not None and joint_name_map is not None:
        cfg_array = np.zeros(len(urdf_joint_names), dtype=np.float32)
        for urdf_idx, dataset_idx in enumerate(joint_name_map):
            if dataset_idx is not None and dataset_idx < len(joint_pos):
                cfg_array[urdf_idx] = joint_pos[dataset_idx]
        viser_urdf.update_cfg(cfg_array)

    if handles["plug_mesh"] is not None:
        plug_pose = obs["plug_pose"]
        handles["plug_mesh"].position = tuple(float(v) for v in plug_pose[:3])
        handles["plug_mesh"].wxyz = tuple(float(v) for v in plug_pose[3:7])

    if handles["socket_mesh"] is not None:
        socket_pose = obs["socket_pose"]
        if np.any(np.abs(socket_pose) > 1e-8):
            handles["socket_mesh"].position = tuple(float(v) for v in socket_pose[:3])
            handles["socket_mesh"].wxyz = tuple(float(v) for v in socket_pose[3:7])

    if handles["left_ee_gizmo"] is not None and not shared.is_dragging("left"):
        handles["left_ee_gizmo"].position = tuple(float(v) for v in left_ee_pos)
        handles["left_ee_gizmo"].wxyz = tuple(float(v) for v in left_ee_quat)

    if handles["right_ee_gizmo"] is not None and not shared.is_dragging("right"):
        handles["right_ee_gizmo"].position = tuple(float(v) for v in right_ee_pos)
        handles["right_ee_gizmo"].wxyz = tuple(float(v) for v in right_ee_quat)


def update_scene_slow(
    handles,
    obs,
    left_ee,
    right_ee,
    left_gripper,
    right_gripper,
    step,
    recording,
    rec_steps,
    shear_csv_recording=False,
    shear_csv_frames=0,
    shear_csv_rows=0,
    hydroshear_source_summary="disabled",
):
    tactile = obs["tactile"]
    tactile_force = obs.get("tactile_force")
    tactile_force_normal = None
    if tactile_force is not None:
        tactile_force = np.asarray(tactile_force)
        if tactile_force.ndim >= 4 and tactile_force.shape[-1] >= 1:
            tactile_force_normal = tactile_force[..., 0]
    tactile_shear = tactile_shear_uv(obs.get("tactile_shear"))
    tactile_marker_shear = tactile_marker_vector_uv(obs.get("tactile_marker_shear"))
    tactile_marker_combined = tactile_marker_vector_uv(obs.get("tactile_marker"))
    tactile_hydro = obs.get("tactile_hydro")
    tactile_hydro_shear = tactile_shear_uv(obs.get("tactile_hydro_marker_shear"))
    if tactile_hydro_shear is None:
        tactile_hydro_shear = tactile_shear_uv(obs.get("tactile_hydro_shear"))
    tac_maxes = []
    force_normal_maxes = []
    main_shear_x_ranges = []
    main_shear_y_ranges = []
    main_shear_mag_maxes = []
    marker_shear_x_ranges = []
    marker_shear_y_ranges = []
    marker_shear_mag_maxes = []
    marker_combined_x_ranges = []
    marker_combined_y_ranges = []
    marker_combined_mag_maxes = []
    hydro_maxes = []
    shear_x_ranges = []
    shear_y_ranges = []
    shear_mag_maxes = []
    for i, handle in enumerate(handles["tac_handles"]):
        grid = tactile[i]
        grid_normal = tactile_normal_channel(grid)
        tac_maxes.append(float(grid_normal.max()))
        main_shear_grid = tactile_shear[i] if tactile_shear is not None else None
        force_normal_grid = tactile_force_normal[i] if tactile_force_normal is not None else None
        if force_normal_grid is not None:
            force_normal_maxes.append(float(np.max(force_normal_grid)))
        if main_shear_grid is not None:
            sx = main_shear_grid[:, :, 0]
            sy = main_shear_grid[:, :, 1]
            mag = np.linalg.norm(main_shear_grid, axis=-1)
            main_shear_x_ranges.append((float(sx.min()), float(sx.max())))
            main_shear_y_ranges.append((float(sy.min()), float(sy.max())))
            main_shear_mag_maxes.append(float(mag.max()))
        marker_shear_grid = tactile_marker_shear[i] if tactile_marker_shear is not None else None
        if marker_shear_grid is not None:
            sx = marker_shear_grid[:, :, 0]
            sy = marker_shear_grid[:, :, 1]
            mag = np.linalg.norm(marker_shear_grid, axis=-1)
            marker_shear_x_ranges.append((float(sx.min()), float(sx.max())))
            marker_shear_y_ranges.append((float(sy.min()), float(sy.max())))
            marker_shear_mag_maxes.append(float(mag.max()))
        marker_combined_grid = tactile_marker_combined[i] if tactile_marker_combined is not None else None
        if marker_combined_grid is not None:
            sx = marker_combined_grid[:, :, 0]
            sy = marker_combined_grid[:, :, 1]
            mag = np.linalg.norm(marker_combined_grid, axis=-1)
            marker_combined_x_ranges.append((float(sx.min()), float(sx.max())))
            marker_combined_y_ranges.append((float(sy.min()), float(sy.max())))
            marker_combined_mag_maxes.append(float(mag.max()))
        hydro_grid = tactile_hydro[i] if tactile_hydro is not None else None
        if hydro_grid is not None:
            hydro_normal = tactile_normal_channel(hydro_grid)
            hydro_maxes.append(float(hydro_normal.max()))
        shear_grid = tactile_hydro_shear[i] if tactile_hydro_shear is not None else None
        if shear_grid is not None:
            sx = shear_grid[:, :, 0]
            sy = shear_grid[:, :, 1]
            mag = np.linalg.norm(shear_grid, axis=-1)
            shear_x_ranges.append((float(sx.min()), float(sx.max())))
            shear_y_ranges.append((float(sy.min()), float(sy.max())))
            shear_mag_maxes.append(float(mag.max()))
        try:
            handle.image = _label_rgb(tactile_to_rgb(grid_normal), f"orig max={_fmt_scalar(float(grid_normal.max()))}")
        except Exception:
            pass
        if main_shear_grid is not None:
            sx = main_shear_grid[:, :, 0]
            sy = main_shear_grid[:, :, 1]
            mag = np.linalg.norm(main_shear_grid, axis=-1)
            sx_vis = _apply_shear_vis_deadband(sx)
            sy_vis = _apply_shear_vis_deadband(sy)
            mag_vis = np.linalg.norm(np.stack((sx_vis, sy_vis), axis=-1), axis=-1)
            signed_vmax = _shear_vis_vmax(sx_vis, sy_vis)
            mag_vmax = _shear_vis_vmax(mag_vis)
            if i < len(handles.get("main_shear_x_handles", [])):
                try:
                    handles["main_shear_x_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sx_vis, vmax=signed_vmax),
                        f"x min {_fmt_scalar(float(sx.min()))}\nx max {_fmt_scalar(float(sx.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("main_shear_y_handles", [])):
                try:
                    handles["main_shear_y_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sy_vis, vmax=signed_vmax),
                        f"y min {_fmt_scalar(float(sy.min()))}\ny max {_fmt_scalar(float(sy.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("main_shear_mag_handles", [])):
                try:
                    handles["main_shear_mag_handles"][i].image = _label_rgb(
                        positive_tactile_to_rgb(mag_vis, vmax=mag_vmax),
                        f"|s| max={_fmt_scalar(float(mag.max()))}",
                    )
                except Exception:
                    pass
        if force_normal_grid is not None and i < len(handles.get("main_force_normal_handles", [])):
            try:
                handles["main_force_normal_handles"][i].image = _label_rgb(
                    tactile_to_rgb(force_normal_grid),
                    f"force max={_fmt_scalar(float(np.max(force_normal_grid)))}",
                )
            except Exception:
                pass
        if marker_shear_grid is not None:
            sx = marker_shear_grid[:, :, 0]
            sy = marker_shear_grid[:, :, 1]
            mag = np.linalg.norm(marker_shear_grid, axis=-1)
            sx_vis = _apply_shear_vis_deadband(sx)
            sy_vis = _apply_shear_vis_deadband(sy)
            mag_vis = np.linalg.norm(np.stack((sx_vis, sy_vis), axis=-1), axis=-1)
            signed_vmax = _shear_vis_vmax(sx_vis, sy_vis)
            mag_vmax = _shear_vis_vmax(mag_vis)
            if i < len(handles.get("marker_shear_x_handles", [])):
                try:
                    handles["marker_shear_x_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sx_vis, vmax=signed_vmax),
                        f"x min {_fmt_scalar(float(sx.min()))}\nx max {_fmt_scalar(float(sx.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("marker_shear_y_handles", [])):
                try:
                    handles["marker_shear_y_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sy_vis, vmax=signed_vmax),
                        f"y min {_fmt_scalar(float(sy.min()))}\ny max {_fmt_scalar(float(sy.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("marker_shear_mag_handles", [])):
                try:
                    handles["marker_shear_mag_handles"][i].image = _label_rgb(
                        positive_tactile_to_rgb(mag_vis, vmax=mag_vmax),
                        f"|s| max={_fmt_scalar(float(mag.max()))}",
                    )
                except Exception:
                    pass
        if marker_combined_grid is not None:
            sx = marker_combined_grid[:, :, 0]
            sy = marker_combined_grid[:, :, 1]
            mag = np.linalg.norm(marker_combined_grid, axis=-1)
            sx_vis = _apply_shear_vis_deadband(sx)
            sy_vis = _apply_shear_vis_deadband(sy)
            mag_vis = np.linalg.norm(np.stack((sx_vis, sy_vis), axis=-1), axis=-1)
            signed_vmax = _shear_vis_vmax(sx_vis, sy_vis)
            mag_vmax = _shear_vis_vmax(mag_vis)
            if i < len(handles.get("marker_combined_x_handles", [])):
                try:
                    handles["marker_combined_x_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sx_vis, vmax=signed_vmax),
                        f"x min {_fmt_scalar(float(sx.min()))}\nx max {_fmt_scalar(float(sx.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("marker_combined_y_handles", [])):
                try:
                    handles["marker_combined_y_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sy_vis, vmax=signed_vmax),
                        f"y min {_fmt_scalar(float(sy.min()))}\ny max {_fmt_scalar(float(sy.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("marker_combined_mag_handles", [])):
                try:
                    handles["marker_combined_mag_handles"][i].image = _label_rgb(
                        positive_tactile_to_rgb(mag_vis, vmax=mag_vmax),
                        f"|m| max={_fmt_scalar(float(mag.max()))}",
                    )
                except Exception:
                    pass
        if hydro_grid is not None and i < len(handles.get("hydro_handles", [])):
            hydro_normal = tactile_normal_channel(hydro_grid)
            try:
                handles["hydro_handles"][i].image = _label_rgb(
                    tactile_to_rgb(hydro_normal),
                    f"hydro max={_fmt_scalar(float(hydro_normal.max()))}",
                )
            except Exception:
                pass
        if shear_grid is not None:
            sx = shear_grid[:, :, 0]
            sy = shear_grid[:, :, 1]
            mag = np.linalg.norm(shear_grid, axis=-1)
            sx_vis = _apply_shear_vis_deadband(sx)
            sy_vis = _apply_shear_vis_deadband(sy)
            mag_vis = np.linalg.norm(np.stack((sx_vis, sy_vis), axis=-1), axis=-1)
            signed_vmax = _shear_vis_vmax(sx_vis, sy_vis)
            mag_vmax = _shear_vis_vmax(mag_vis)
            if i < len(handles.get("hydro_shear_x_handles", [])):
                try:
                    handles["hydro_shear_x_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sx_vis, vmax=signed_vmax),
                        f"x min {_fmt_scalar(float(sx.min()))}\nx max {_fmt_scalar(float(sx.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("hydro_shear_y_handles", [])):
                try:
                    handles["hydro_shear_y_handles"][i].image = _label_rgb(
                        signed_tactile_to_rgb(sy_vis, vmax=signed_vmax),
                        f"y min {_fmt_scalar(float(sy.min()))}\ny max {_fmt_scalar(float(sy.max()))}",
                    )
                except Exception:
                    pass
            if i < len(handles.get("hydro_shear_mag_handles", [])):
                try:
                    handles["hydro_shear_mag_handles"][i].image = _label_rgb(
                        positive_tactile_to_rgb(mag_vis, vmax=mag_vmax),
                        f"|s| max={_fmt_scalar(float(mag.max()))}",
                    )
                except Exception:
                    pass

    if "rgb" in obs:
        try:
            handles["camera_img"].image = obs["rgb"]
        except Exception:
            pass

    rec_status = f"  **REC** ({rec_steps} steps)" if recording else ""
    shear_csv_status = (
        f"REC ({int(shear_csv_frames)} frames, {int(shear_csv_rows)} rows)" if shear_csv_recording else "idle"
    )
    tac_str = ", ".join(_fmt_scalar(m) for m in tac_maxes)
    main_sx_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in main_shear_x_ranges)
        if main_shear_x_ranges
        else "disabled"
    )
    main_sy_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in main_shear_y_ranges)
        if main_shear_y_ranges
        else "disabled"
    )
    main_smag_str = (
        ", ".join(_fmt_scalar(m) for m in main_shear_mag_maxes) if main_shear_mag_maxes else "disabled"
    )
    force_normal_str = (
        ", ".join(_fmt_scalar(m) for m in force_normal_maxes) if force_normal_maxes else "disabled"
    )
    marker_sx_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in marker_shear_x_ranges)
        if marker_shear_x_ranges
        else "disabled"
    )
    marker_sy_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in marker_shear_y_ranges)
        if marker_shear_y_ranges
        else "disabled"
    )
    marker_smag_str = (
        ", ".join(_fmt_scalar(m) for m in marker_shear_mag_maxes) if marker_shear_mag_maxes else "disabled"
    )
    marker_combined_sx_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in marker_combined_x_ranges)
        if marker_combined_x_ranges
        else "disabled"
    )
    marker_combined_sy_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in marker_combined_y_ranges)
        if marker_combined_y_ranges
        else "disabled"
    )
    marker_combined_smag_str = (
        ", ".join(_fmt_scalar(m) for m in marker_combined_mag_maxes)
        if marker_combined_mag_maxes
        else "disabled"
    )
    if hydro_maxes:
        hydro_str = ", ".join(_fmt_scalar(m) for m in hydro_maxes)
        sx_str = (
            ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in shear_x_ranges)
            if shear_x_ranges
            else "disabled"
        )
        sy_str = (
            ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in shear_y_ranges)
            if shear_y_ranges
            else "disabled"
        )
        smag_str = ", ".join(_fmt_scalar(m) for m in shear_mag_maxes) if shear_mag_maxes else "disabled"
        normal_table = (
            "| Pad | Orig normal max | Hydro normal max | Hydro shear-x min/max | Hydro shear-y min/max | Hydro \\|shear\\| max |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: |\n"
            + "\n".join(
                f"| {label} | {_fmt_scalar(orig)} | {_fmt_scalar(hydro)} | "
                f"[{_fmt_scalar(sx[0])},{_fmt_scalar(sx[1])}] | "
                f"[{_fmt_scalar(sy[0])},{_fmt_scalar(sy[1])}] | {_fmt_scalar(smag)} |"
                for label, orig, hydro, sx, sy, smag in zip(
                    ("L-L", "L-R", "R-L", "R-R"),
                    tac_maxes,
                    hydro_maxes,
                    shear_x_ranges or [(0.0, 0.0)] * len(hydro_maxes),
                    shear_y_ranges or [(0.0, 0.0)] * len(hydro_maxes),
                    shear_mag_maxes or [0.0] * len(hydro_maxes),
                    strict=False,
                )
            )
        )
    else:
        hydro_str = "disabled"
        sx_str = "disabled"
        sy_str = "disabled"
        smag_str = "disabled"
        normal_table = ""
    shear_vis_vmax = max(0.0, float(getattr(args, "shear_vis_vmax", 0.0)))
    shear_vis_vmax_str = "auto" if shear_vis_vmax <= 0.0 else _fmt_scalar(shear_vis_vmax)
    shear_vis_deadband_str = _fmt_scalar(_shear_vis_deadband())
    handles["state_md"].content = (
        f"**Left EE:** ({left_ee[0]:.4f}, {left_ee[1]:.4f}, {left_ee[2]:.4f})  \n"
        f"**Right EE:** ({right_ee[0]:.4f}, {right_ee[1]:.4f}, {right_ee[2]:.4f})  \n"
        f"**Left gripper:** {left_gripper:.2f}  \n"
        f"**Right gripper:** {right_gripper:.2f}  \n"
        f"**Shear CSV:** {shear_csv_status}  \n"
        f"**HydroShear elastomer SDF:** {hydroshear_source_summary}  \n"
        f"**Shear vis:** vmax={shear_vis_vmax_str}, deadband={shear_vis_deadband_str} display-only  \n"
        f"**Pad axes:** cyan=shear_x, orange=shear_y, green=normal  \n"
        f"**Tactile max [4 pads]:** [{tac_str}]  \n"
        f"**Force normal max [4 pads]:** [{force_normal_str}]  \n"
        f"**Force shear-x min/max [4 pads]:** [{main_sx_str}]  \n"
        f"**Force shear-y min/max [4 pads]:** [{main_sy_str}]  \n"
        f"**Force |shear| max [4 pads]:** [{main_smag_str}]  \n"
        f"**Marker shear-x min/max [4 pads]:** [{marker_sx_str}]  \n"
        f"**Marker shear-y min/max [4 pads]:** [{marker_sy_str}]  \n"
        f"**Marker |shear| max [4 pads]:** [{marker_smag_str}]  \n"
        f"**Marker combined-x min/max [4 pads]:** [{marker_combined_sx_str}]  \n"
        f"**Marker combined-y min/max [4 pads]:** [{marker_combined_sy_str}]  \n"
        f"**Marker combined |xy| max [4 pads]:** [{marker_combined_smag_str}]  \n"
        f"**Hydro normal max [4 pads]:** [{hydro_str}]  \n"
        f"**Hydro shear-x min/max [4 pads]:** [{sx_str}]  \n"
        f"**Hydro shear-y min/max [4 pads]:** [{sy_str}]  \n"
        f"**Hydro |shear| max [4 pads]:** [{smag_str}]  \n"
        f"{normal_table}\n"
        f"**Step:** {step}{rec_status}"
    )


# ---------------------------------------------------------------------------
# Pose-mode IK
# ---------------------------------------------------------------------------

def upgrade_ik_to_pose_mode(ik_ctrl):
    from isaaclab.controllers.differential_ik import DifferentialIKController
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

    pose_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    ik_ctrl._ik_controller = DifferentialIKController(cfg=pose_cfg, num_envs=1, device=ik_ctrl._device)


def compute_pose(ik_ctrl, target_pos, target_quat, gripper):
    from aloha_ik_controller import GRIPPER_CLOSED, GRIPPER_OPEN

    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
    target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)

    command = np.concatenate([target_pos, target_quat])
    command_t = torch.tensor(command, dtype=torch.float32, device=ik_ctrl._device).unsqueeze(0)

    ee_pos_w = ik_ctrl._robot.data.body_pos_w[:, ik_ctrl._ee_body_idx]
    ee_quat_w = ik_ctrl._robot.data.body_quat_w[:, ik_ctrl._ee_body_idx]
    current_arm_pos = ik_ctrl._robot.data.joint_pos[:, ik_ctrl._arm_joint_ids]
    jacobian = ik_ctrl._robot.root_physx_view.get_jacobians()[:, ik_ctrl._jacobi_body_idx, :, ik_ctrl._jacobi_joint_ids]

    ik_ctrl._ik_controller.set_command(command=command_t)
    joint_pos_des = ik_ctrl._ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, current_arm_pos)

    all_joint_pos = ik_ctrl._robot.data.joint_pos[0, ik_ctrl._dataset_joint_ids].clone()
    action = all_joint_pos.detach().cpu().numpy().astype(np.float32)

    for i, arm_jid in enumerate(ik_ctrl._arm_joint_ids):
        slot = ik_ctrl._artjid_to_dataset_slot[arm_jid]
        action[slot] = float(joint_pos_des[0, i].detach().cpu())

    gripper = float(np.clip(gripper, 0.0, 1.0))
    gripper_pos = GRIPPER_CLOSED + gripper * (GRIPPER_OPEN - GRIPPER_CLOSED)
    left_slot, right_slot = ik_ctrl._gripper_slots
    action[left_slot] = gripper_pos
    action[right_slot] = -gripper_pos

    return action


def get_rigid_object_pose_w(obj) -> tuple[np.ndarray, np.ndarray] | None:
    if obj is None:
        return None
    root_pos = getattr(obj.data, "root_pos_w", None)
    root_quat = getattr(obj.data, "root_quat_w", None)
    if root_pos is None or root_quat is None:
        return None
    pos = root_pos[0].detach().cpu().numpy().astype(np.float64)
    quat = root_quat[0].detach().cpu().numpy().astype(np.float64)
    return pos, quat


def write_rigid_object_pose_w(obj, pos_w: np.ndarray, quat_wxyz: np.ndarray):
    if obj is None:
        return
    device = getattr(obj, "device", "cuda:0")
    pose = np.concatenate(
        [
            np.asarray(pos_w, dtype=np.float32).reshape(3),
            np.asarray(quat_wxyz, dtype=np.float32).reshape(4),
        ]
    )
    pose_t = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
    vel_t = torch.zeros((1, 6), dtype=torch.float32, device=device)
    obj.write_root_pose_to_sim(pose_t)
    obj.write_root_velocity_to_sim(vel_t)


def _quat_from_matrix_wxyz(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + 1.0e-12)


def object_contact_pose_from_pad(pad_axes: dict[str, np.ndarray], cfg, preload_m: float) -> tuple[np.ndarray, np.ndarray] | None:
    normal_axis = int(cfg.tactile.normal_axis)
    axis_u, axis_v = _tangential_axes(normal_axis)
    normal_w = _unit_vec(pad_axes.get("normal_w", np.zeros(3)))
    shear_x_w = _unit_vec(pad_axes.get("axis_x_w", np.zeros(3)))
    shear_y_w = _unit_vec(pad_axes.get("axis_y_w", np.zeros(3)))
    pad_pos_w = np.asarray(pad_axes.get("position_w", np.zeros(3)), dtype=np.float64).reshape(3)
    if not np.any(normal_w) or not np.any(shear_x_w) or not np.any(shear_y_w):
        return None

    rot = np.zeros((3, 3), dtype=np.float64)
    rot[:, normal_axis] = normal_w
    rot[:, axis_u] = shear_x_w
    rot[:, axis_v] = shear_y_w
    if np.linalg.det(rot) < 0.0:
        rot[:, axis_v] *= -1.0

    cube_size = np.asarray(cfg.objects.cube_size, dtype=np.float64).reshape(3)
    half_normal_extent = 0.5 * float(cube_size[normal_axis])
    surface_offset = float(cfg.tactile.normal_offset)
    center_offset = surface_offset + half_normal_extent - float(preload_m)
    pos_w = pad_pos_w + normal_w * center_offset
    quat_wxyz = _quat_from_matrix_wxyz(rot)
    return pos_w, quat_wxyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = int(args.port)

    cfg = AlohaTactileEnvCfg(
        camera=AlohaCameraCfg(enable_camera=True),
        sim=AlohaSimCfg(
            headless=bool(getattr(args, "headless", True)),
            device=str(getattr(args, "device", "cuda:0")),
        ),
    )
    if args.robot_urdf:
        cfg.robot.urdf_path = str(Path(os.path.expanduser(args.robot_urdf)).resolve())
    elif bool(args.use_bump_pad):
        cfg.robot.urdf_path = str((Path(__file__).resolve().parent / "assets" / "aloha_tactile_bump.urdf").resolve())
    cube_side = 0.026
    cube_size = (cube_side * 1.5, cube_side * 3.0, cube_side * 3.0)
    cube_z = cube_size[2] * 0.5 + 0.001
    cfg.objects.use_cube_objects = True
    cfg.objects.enable_plug = True
    cfg.objects.enable_socket = True
    cfg.objects.cube_size = cube_size
    cfg.objects.cube_mass = 0.04
    cfg.objects.plug_fix_base = True
    cfg.objects.socket_fix_base = True
    cfg.objects.plug_scale = 1.0
    cfg.objects.socket_scale = 1.0
    cfg.objects.plug_default_pose = (0.0, 0.05, cube_z, 0.0, 0.0, 0.0, 1.0)
    cfg.objects.socket_default_pose = (0.0, -0.05, cube_z, 0.0, 0.0, 0.0, 1.0)

    if args.tactile_backend == "taxel_shear":
        cfg.tactile.backend = TaxelShearTactileBackendCfg()
    elif args.tactile_backend == "surface_hydro":
        cfg.tactile.backend = HydroShearTactileBackendCfg(
            include_force_observations=True,
            bump_enabled=bool(args.use_bump_pad),
        )
        cfg.tactile.backend.use_elastomer_mesh_sdf = not bool(args.force_flat_elastomer_sdf)
        cfg.tactile.backend.debug_print = True
    cfg.tactile.enable_hydro_normal_observation = bool(args.compare_hydro_normal)
    cfg.tactile.hydro_normal_backend.include_force_observations = bool(args.compare_hydro_normal)
    cfg.tactile.hydro_normal_backend.normal_readout_scale = float(args.hydro_normal_scale)
    cfg.tactile.hydro_normal_backend.shear_readout_scale = float(args.hydro_shear_scale)
    cfg.tactile.hydro_normal_backend.bump_enabled = bool(args.use_bump_pad)
    cfg.tactile.hydro_normal_backend.use_elastomer_mesh_sdf = not bool(args.force_flat_elastomer_sdf)
    cfg.tactile.hydro_normal_backend.debug_print = bool(args.hydro_debug_print)
    if args.hydro_shear_stiffness is not None:
        cfg.tactile.hydro_normal_backend.shear_stiffness = float(args.hydro_shear_stiffness)

    if not cfg.sim.headless:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(list(cfg.camera.eye), list(cfg.camera.target))
        except Exception as e:
            print(f"[WARN] Failed to set viewport camera view: {e}", flush=True)

    server, handles = create_viser_server(port, cfg)

    handles["state_md"].content = "**Loading Isaac Sim environment...**"
    env = AlohaTactileEnv(cfg, simulation_app=simulation_app)
    obs, _ = env.reset()
    print(f"[HydroShear] elastomer sources: {hydroshear_elastomer_source_summary(env)}", flush=True)

    left_ik = AlohaArmIKController(robot=env._robot, device=cfg.sim.device, arm="left")
    right_ik = AlohaArmIKController(robot=env._robot, device=cfg.sim.device, arm="right")
    upgrade_ik_to_pose_mode(left_ik)
    upgrade_ik_to_pose_mode(right_ik)

    aloha_init = np.array([
        0.0, -0.16, 1.15, 0.0, -0.5, 0.0, 0.057, -0.057,
        0.0, -0.16, 1.15, 0.0, -0.5, 0.0, 0.057, -0.057,
    ], dtype=np.float32)
    left_arm_init = aloha_init[:8].copy()
    right_arm_init = aloha_init[8:].copy()
    init_joint_pos = aloha_init.copy()
    left_gripper_init = 1.0
    right_gripper_init = 1.0

    warmup_steps = 50
    handles["state_md"].content = "**Running warmup steps...**"
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(init_joint_pos)

    left_ee_target_pos = np.array([0.0, -0.167, 0.074], dtype=np.float64)
    right_ee_target_pos = np.array([0.0, 0.167, 0.074], dtype=np.float64)
    left_ee_quat_init = left_ik.get_ee_quat()
    right_ee_quat_init = right_ik.get_ee_quat()

    save_dir = args.save_dir or os.path.join(os.path.dirname(__file__), "output", "trajectories")
    recorder = TrajectoryRecorder(save_dir)
    print(f"[INFO] Trajectories will be saved to: {save_dir}", flush=True)
    shear_csv_dir = args.shear_csv_dir or os.path.join(os.path.dirname(__file__), "output", "tactile_shear_csv")
    shear_csv_recorder = TactileShearCsvRecorder(
        shear_csv_dir,
        pad_labels=[
            "Left arm / left finger",
            "Left arm / right finger",
            "Right arm / left finger",
            "Right arm / right finger",
        ],
        interval=int(args.shear_csv_interval),
        normal_axis=int(cfg.tactile.normal_axis),
    )
    print(f"[INFO] Tactile shear CSV will be saved to: {shear_csv_dir}", flush=True)

    handles["state_md"].content = "**Loading 3D models...**"
    load_scene_objects(server, handles, cfg)
    create_ee_gizmo(server, handles, "left", left_ee_target_pos, left_ee_quat_init)
    create_ee_gizmo(server, handles, "right", right_ee_target_pos, right_ee_quat_init)
    shared.set_target("left", *[float(v) for v in left_ee_target_pos])
    shared.set_target_quat("left", *[float(v) for v in left_ee_quat_init])
    shared.set_target("right", *[float(v) for v in right_ee_target_pos])
    shared.set_target_quat("right", *[float(v) for v in right_ee_quat_init])
    shared.set_gripper("left", left_gripper_init)
    shared.set_gripper("right", right_gripper_init)
    handles["left_gripper_slider"].value = left_gripper_init
    handles["right_gripper_slider"].value = right_gripper_init

    urdf_joint_names = ()
    joint_name_map = None
    if handles["viser_urdf"] is not None:
        urdf_joint_names = handles["viser_urdf"].get_actuated_joint_names()
        joint_name_map = build_joint_name_map(urdf_joint_names)

    handles["state_md"].content = "**Ready. Drag either gizmo to control the bimanual arms with cube objects.**"

    print('\n' + '=' * 72, flush=True)
    print(f'[VISER URL] http://localhost:{port}', flush=True)
    print(f'[VISER URL] Open this in your browser: http://localhost:{port}', flush=True)
    print('=' * 72 + '\n', flush=True)

    step = 0
    slow_interval = max(1, int(args.slow_interval))
    was_recording = False
    was_shear_csv_recording = False
    object_slide_trial: dict | None = None
    auto_probe_axes: list[str] = []
    auto_probe_visual_demo = bool(WORKING_PROBE_DEMO)
    if auto_probe_visual_demo:
        auto_probe_axes = ["x", "y"]
        print(
            "[AUTO] Working probe demo enabled: pad2 shear_x then shear_y, "
            "5.00mm total, 0.50mm step, 180 settle steps.",
            flush=True,
        )
    elif args.auto_object_slide_probe == "both":
        auto_probe_axes = ["x", "y"]
    elif args.auto_object_slide_probe in ("x", "y"):
        auto_probe_axes = [str(args.auto_object_slide_probe)]
    auto_probe_pending_axis: str | None = None
    auto_probe_complete = False
    prev_debug_hydro_relative_poses = _read_hydro_relative_poses(env)

    while simulation_app.is_running():
        recording = shared.is_recording()
        shear_csv_recording = shared.is_shear_csv_recording()

        if was_recording and not recording:
            path = recorder.flush()
            if path:
                handles["state_md"].content = f"**Saved!** {os.path.basename(path)}"
        was_recording = recording

        if shear_csv_recording and not was_shear_csv_recording:
            shear_csv_recorder.start()
            shear_csv_recorder.seed_reference_poses(
                left_target_pos=np.array(shared.get_target("left"), dtype=np.float64),
                right_target_pos=np.array(shared.get_target("right"), dtype=np.float64),
                plug_pose=obs.get("plug_pose"),
                socket_pose=obs.get("socket_pose"),
                hydro_target_poses_w=_read_hydro_target_poses_world(env),
                hydro_relative_poses=_read_hydro_relative_poses(env),
            )
        if was_shear_csv_recording and not shear_csv_recording:
            path = shear_csv_recorder.stop()
            if path:
                handles["state_md"].content = f"**Shear CSV saved!** {os.path.basename(path)}"
        was_shear_csv_recording = shear_csv_recording

        if (
            auto_probe_axes
            and auto_probe_pending_axis is None
            and object_slide_trial is None
            and not shear_csv_recorder.active
        ):
            axis = auto_probe_axes.pop(0)
            pad = 2 if auto_probe_visual_demo else int(np.clip(int(args.auto_object_slide_pad), 0, 3))
            distance_mm = 5.0 if auto_probe_visual_demo else max(0.0, float(args.auto_object_slide_distance_mm))
            step_mm = 0.5 if auto_probe_visual_demo else max(0.001, float(args.auto_object_slide_step_mm))
            settle_steps = 180 if auto_probe_visual_demo else max(0, int(args.auto_object_slide_settle_steps))
            hold_steps = 12 if auto_probe_visual_demo else max(1, int(args.auto_object_slide_hold_steps))
            shared.request_object_slide_trial(
                pad=pad,
                axis=axis,
                direction=1.0,
                distance_m=distance_mm * 1.0e-3,
                step_m=step_mm * 1.0e-3,
                settle_steps=settle_steps,
                hold_steps=hold_steps,
                auto_place=True,
                preload_m=0.001,
            )
            auto_probe_pending_axis = axis
            print(
                f"[AUTO] Object slide probe queued: pad{pad} shear_{axis} "
                f"distance={distance_mm:.2f}mm step={step_mm:.2f}mm settle={settle_steps}",
                flush=True,
            )

        object_slide_request = shared.consume_object_slide_trial_request()
        if object_slide_request is not None:
            if shear_csv_recorder.active:
                shear_csv_recorder.stop()
            shared.set_shear_csv_recording(False)
            shear_csv_recording = False
            handles["shear_csv_btn"].name = "Start Shear CSV"
            handles["shear_csv_btn"].color = "blue"
            was_shear_csv_recording = False
            if bool(object_slide_request.get("auto_place", True)):
                reset_tactile_state(env)
            object_slide_trial = {
                **object_slide_request,
                "phase": "settle",
                "settle_remaining": int(object_slide_request["settle_steps"]),
                "hold_remaining": 0,
                "distance_remaining_m": float(object_slide_request["distance_m"]),
                "object_name": "socket" if int(object_slide_request["pad"]) < 2 else "plug",
                "object_pos_w": None,
                "object_quat_w": None,
                "robot_hold_action": None,
                "preload_reference_done": False,
                "preload_pose_applied": False,
            }
            handles["state_md"].content = (
                f"**Object slide armed:** {object_slide_trial['object_name']} pad{object_slide_request['pad']} "
                f"shear_{object_slide_request['axis']} settle={object_slide_request['settle_steps']} "
                f"move={object_slide_request['distance_m'] * 1.0e3:.2f}mm "
                f"step={object_slide_request['step_m'] * 1.0e3:.2f}mm "
                f"hold={object_slide_request['hold_steps']} "
                f"auto_place={object_slide_request['auto_place']} "
                f"preload={object_slide_request['preload_m'] * 1.0e3:.2f}mm"
            )

        if shared.consume_reset():
            if recorder.num_steps > 0:
                recorder.flush()
            if shear_csv_recorder.active:
                shear_csv_recorder.stop()
            shared.set_recording(False)
            shared.set_shear_csv_recording(False)
            handles["record_btn"].name = "Start Recording"
            handles["record_btn"].color = "green"
            handles["shear_csv_btn"].name = "Start Shear CSV"
            handles["shear_csv_btn"].color = "blue"
            recording = False
            shear_csv_recording = False
            was_recording = False
            was_shear_csv_recording = False
            object_slide_trial = None

            obs, _ = env.reset()
            print(f"[HydroShear] elastomer sources: {hydroshear_elastomer_source_summary(env)}", flush=True)
            prev_debug_hydro_relative_poses = _read_hydro_relative_poses(env)
            for _ in range(warmup_steps):
                obs, _, _, _, _ = env.step(init_joint_pos)
                prev_debug_hydro_relative_poses = _read_hydro_relative_poses(env)

            left_pos = left_ee_target_pos.copy()
            right_pos = right_ee_target_pos.copy()
            left_quat = left_ik.get_ee_quat().astype(np.float64)
            right_quat = right_ik.get_ee_quat().astype(np.float64)

            left_delta = np.random.uniform(-np.radians(15), np.radians(15))
            right_delta = np.random.uniform(-np.radians(15), np.radians(15))
            left_dq = np.array([np.cos(left_delta / 2.0), np.sin(left_delta / 2.0), 0.0, 0.0], dtype=np.float64)
            right_dq = np.array([np.cos(right_delta / 2.0), np.sin(right_delta / 2.0), 0.0, 0.0], dtype=np.float64)
            left_quat = _quat_mul_wxyz(left_dq, left_quat)
            right_quat = _quat_mul_wxyz(right_dq, right_quat)
            left_quat /= np.linalg.norm(left_quat)
            right_quat /= np.linalg.norm(right_quat)

            shared.set_target("left", *[float(v) for v in left_pos])
            shared.set_target_quat("left", *[float(v) for v in left_quat])
            shared.set_target("right", *[float(v) for v in right_pos])
            shared.set_target_quat("right", *[float(v) for v in right_quat])
            shared.set_gripper("left", left_gripper_init)
            shared.set_gripper("right", right_gripper_init)
            handles["left_gripper_slider"].value = left_gripper_init
            handles["right_gripper_slider"].value = right_gripper_init
            if handles["left_ee_gizmo"] is not None:
                handles["left_ee_gizmo"].position = tuple(float(v) for v in left_pos)
                handles["left_ee_gizmo"].wxyz = tuple(float(v) for v in left_quat)
            if handles["right_ee_gizmo"] is not None:
                handles["right_ee_gizmo"].position = tuple(float(v) for v in right_pos)
                handles["right_ee_gizmo"].wxyz = tuple(float(v) for v in right_quat)
            step = 0

        left_target_pos = np.array(shared.get_target("left"), dtype=np.float32)
        left_target_quat = np.array(shared.get_target_quat("left"), dtype=np.float32)
        right_target_pos = np.array(shared.get_target("right"), dtype=np.float32)
        right_target_quat = np.array(shared.get_target_quat("right"), dtype=np.float32)
        left_gripper = shared.get_gripper("left")
        right_gripper = shared.get_gripper("right")

        left_ee_fk = left_ik.get_ee_pos()
        left_ee_quat_fk = left_ik.get_ee_quat()
        right_ee_fk = right_ik.get_ee_pos()
        right_ee_quat_fk = right_ik.get_ee_quat()

        if object_slide_trial is not None and object_slide_trial.get("robot_hold_action") is None:
            hold_action = env._robot.data.joint_pos[0, left_ik._dataset_joint_ids].detach().cpu().numpy().astype(np.float32)
            object_slide_trial["robot_hold_action"] = hold_action.copy()
            left_target_pos = left_ee_fk.astype(np.float32, copy=True)
            left_target_quat = left_ee_quat_fk.astype(np.float32, copy=True)
            right_target_pos = right_ee_fk.astype(np.float32, copy=True)
            right_target_quat = right_ee_quat_fk.astype(np.float32, copy=True)
            shared.set_target("left", *[float(v) for v in left_target_pos])
            shared.set_target_quat("left", *[float(v) for v in left_target_quat])
            shared.set_target("right", *[float(v) for v in right_target_pos])
            shared.set_target_quat("right", *[float(v) for v in right_target_quat])
            if handles["left_ee_gizmo"] is not None:
                handles["left_ee_gizmo"].position = tuple(float(v) for v in left_target_pos)
                handles["left_ee_gizmo"].wxyz = tuple(float(v) for v in left_target_quat)
            if handles["right_ee_gizmo"] is not None:
                handles["right_ee_gizmo"].position = tuple(float(v) for v in right_target_pos)
                handles["right_ee_gizmo"].wxyz = tuple(float(v) for v in right_target_quat)

        commanded_left_delta_w = np.zeros(3, dtype=np.float64)
        commanded_right_delta_w = np.zeros(3, dtype=np.float64)
        commanded_object_delta_w = np.zeros(3, dtype=np.float64)
        commanded_object_name = ""
        commanded_pad = None
        commanded_axis = ""
        trial_phase = ""

        if object_slide_trial is not None:
            phase = str(object_slide_trial.get("phase", ""))
            trial_phase = f"object_{phase}"
            if phase == "settle":
                remaining = int(object_slide_trial.get("settle_remaining", 0))
                object_name = str(object_slide_trial.get("object_name", "plug"))
                obj = env._socket_obj if object_name == "socket" else env._plug_obj
                pose = None
                wrote_preload_reference = False
                wrote_first_preload_pose = False
                if bool(object_slide_trial.get("auto_place", True)):
                    pad_axes_w = _read_pad_axes_world(env, cfg)
                    pad_axes = pad_axes_w.get(int(object_slide_trial["pad"]))
                    preload_m = float(object_slide_trial.get("preload_m", 0.0))
                    pose_preload_m = preload_m
                    if preload_m > 0.0 and not bool(object_slide_trial.get("preload_reference_done", False)):
                        pose_preload_m = 0.0
                        object_slide_trial["preload_reference_done"] = True
                        wrote_preload_reference = True
                    elif preload_m > 0.0 and not bool(object_slide_trial.get("preload_pose_applied", False)):
                        object_slide_trial["preload_pose_applied"] = True
                        wrote_first_preload_pose = True
                    contact_pose = (
                        object_contact_pose_from_pad(
                            pad_axes,
                            cfg,
                            pose_preload_m,
                        )
                        if pad_axes is not None
                        else None
                    )
                    if contact_pose is not None:
                        write_rigid_object_pose_w(obj, contact_pose[0], contact_pose[1])
                        pose = contact_pose
                if pose is None:
                    pose = get_rigid_object_pose_w(obj)
                if pose is not None:
                    object_slide_trial["object_pos_w"] = pose[0].copy()
                    object_slide_trial["object_quat_w"] = pose[1].copy()
                if wrote_preload_reference:
                    trial_phase = "object_preload_reference"
                elif wrote_first_preload_pose and remaining <= 0:
                    object_slide_trial["settle_remaining"] = 1
                    trial_phase = "object_preload_apply"
                elif remaining > 0:
                    object_slide_trial["settle_remaining"] = remaining - 1
                else:
                    zeroed = zero_hydroshear_shear_preserve_normal(env)
                    if zeroed > 0:
                        print(f"[HydroShear] Object slide zeroed shear state for {zeroed} core(s).", flush=True)
                    shared.set_shear_csv_recording(True)
                    handles["shear_csv_btn"].name = "Stop Shear CSV"
                    handles["shear_csv_btn"].color = "orange"
                    object_slide_trial["phase"] = "move_wait_csv"
                    trial_phase = "object_move_wait_csv"
            elif phase == "move_wait_csv":
                if shear_csv_recorder.active:
                    object_slide_trial["phase"] = "move"
                    trial_phase = "object_move"
            elif phase == "move":
                distance_remaining = float(object_slide_trial.get("distance_remaining_m", 0.0))
                hold_remaining = int(object_slide_trial.get("hold_remaining", 0))
                if hold_remaining > 0:
                    object_slide_trial["hold_remaining"] = hold_remaining - 1
                    trial_phase = "object_hold"
                elif distance_remaining > 1.0e-9:
                    pad = int(object_slide_trial["pad"])
                    axis = str(object_slide_trial["axis"])
                    object_name = str(object_slide_trial.get("object_name", "plug"))
                    obj = env._socket_obj if object_name == "socket" else env._plug_obj
                    pad_axes_w = _read_pad_axes_world(env, cfg)
                    pad_axes = pad_axes_w.get(pad)
                    pose = get_rigid_object_pose_w(obj)
                    if pad_axes is not None and pose is not None:
                        if object_slide_trial.get("object_pos_w") is None:
                            object_slide_trial["object_pos_w"] = pose[0].copy()
                            object_slide_trial["object_quat_w"] = pose[1].copy()
                        axis_key = "axis_x_w" if axis == "x" else "axis_y_w"
                        axis_vec = np.asarray(pad_axes.get(axis_key, np.zeros(3)), dtype=np.float64).reshape(3)
                        axis_norm = float(np.linalg.norm(axis_vec))
                        if axis_norm > 1.0e-12:
                            axis_vec = axis_vec / axis_norm
                            step_m = min(float(object_slide_trial.get("step_m", 0.0)), distance_remaining)
                            start_pos = np.asarray(object_slide_trial["object_pos_w"], dtype=np.float64).reshape(3)
                            quat_w = np.asarray(object_slide_trial["object_quat_w"], dtype=np.float64).reshape(4)
                            delta = axis_vec * float(object_slide_trial["direction"]) * step_m
                            new_pos = start_pos + delta
                            write_rigid_object_pose_w(obj, new_pos, quat_w)
                            object_slide_trial["object_pos_w"] = new_pos.copy()
                            object_slide_trial["distance_remaining_m"] = max(0.0, distance_remaining - step_m)
                            object_slide_trial["hold_remaining"] = max(0, int(object_slide_trial.get("hold_steps", 1)) - 1)
                            commanded_object_name = object_name
                            commanded_object_delta_w += delta.astype(np.float64)
                            commanded_pad = pad
                            commanded_axis = axis
                    trial_phase = "object_move"
                else:
                    object_slide_trial["phase"] = "stop"
                    trial_phase = "object_stop"
            elif phase == "stop":
                path = shear_csv_recorder.stop()
                shared.set_shear_csv_recording(False)
                shear_csv_recording = False
                was_shear_csv_recording = False
                handles["shear_csv_btn"].name = "Start Shear CSV"
                handles["shear_csv_btn"].color = "blue"
                if path:
                    handles["state_md"].content = f"**Object slide CSV saved!** {os.path.basename(path)}"
                    if auto_probe_pending_axis is not None:
                        print(
                            f"[AUTO] Object slide probe shear_{auto_probe_pending_axis} saved: {path}",
                            flush=True,
                        )
                if auto_probe_pending_axis is not None:
                    auto_probe_pending_axis = None
                    auto_probe_complete = not auto_probe_axes
                object_slide_trial = None

        if object_slide_trial is not None and object_slide_trial.get("robot_hold_action") is not None:
            action = np.asarray(object_slide_trial["robot_hold_action"], dtype=np.float32).copy()
        else:
            left_action = compute_pose(left_ik, left_target_pos, left_target_quat, left_gripper)
            right_action = compute_pose(right_ik, right_target_pos, right_target_quat, right_gripper)
            action = np.concatenate([left_action[:8], right_action[8:]], axis=0).astype(np.float32)
        obs, _, _, _, _ = env.step(action)

        left_ee = left_ik.get_ee_pos()
        left_ee_quat = left_ik.get_ee_quat()
        right_ee = right_ik.get_ee_pos()
        right_ee_quat = right_ik.get_ee_quat()
        pad_axes_w = _read_pad_axes_world(env, cfg)

        curr_debug_hydro_relative_poses = _read_hydro_relative_poses(env)
        debug_pad = commanded_pad
        if debug_pad is None and object_slide_trial is not None:
            debug_pad = int(object_slide_trial.get("pad", -1))
        debug_hydro_delta = np.zeros(3, dtype=np.float64)
        if debug_pad is not None:
            curr_pose = curr_debug_hydro_relative_poses.get(int(debug_pad))
            prev_pose = prev_debug_hydro_relative_poses.get(int(debug_pad))
            if curr_pose is not None and prev_pose is not None:
                debug_hydro_delta = np.asarray(curr_pose[0], dtype=np.float64) - np.asarray(prev_pose[0], dtype=np.float64)
        debug_command_delta = commanded_object_delta_w + commanded_left_delta_w + commanded_right_delta_w
        debug_text = update_hydroshear_debug_vectors(
            handles,
            obs,
            pad_axes_w,
            pad=None if debug_pad is None or int(debug_pad) < 0 else int(debug_pad),
            command_delta_w=debug_command_delta,
            hydro_rel_delta_p=debug_hydro_delta,
            visible=True,
        )
        if handles.get("hydro_debug_md") is not None:
            try:
                handles["hydro_debug_md"].content = (
                    f"**Vectors:** blue=command, orange=HydroShear relative motion, purple=force shear.  \n"
                    f"**Current:** {debug_text}"
                )
            except Exception:
                pass
        prev_debug_hydro_relative_poses = curr_debug_hydro_relative_poses

        if recording:
            recorder.record(
                obs,
                left_target_pos,
                left_target_quat,
                left_gripper,
                right_target_pos,
                right_target_quat,
                right_gripper,
                left_ee_pos=left_ee,
                left_ee_quat=left_ee_quat,
                right_ee_pos=right_ee,
                right_ee_quat=right_ee_quat,
                joint_commands=action,
            )
        if shear_csv_recording:
            shear_csv_recorder.record(
                obs,
                step,
                left_target_pos=left_target_pos,
                right_target_pos=right_target_pos,
                left_ee_pos=left_ee,
                right_ee_pos=right_ee,
                pad_axes_w=pad_axes_w,
                commanded_left_delta_w=commanded_left_delta_w,
                commanded_right_delta_w=commanded_right_delta_w,
                commanded_pad=commanded_pad,
                commanded_axis=commanded_axis,
                trial_phase=trial_phase,
                plug_pose=obs.get("plug_pose"),
                socket_pose=obs.get("socket_pose"),
                commanded_object_name=commanded_object_name,
                commanded_object_delta_w=commanded_object_delta_w,
                hydro_target_poses_w=_read_hydro_target_poses_world(env),
                hydro_relative_poses=_read_hydro_relative_poses(env),
            )

        update_scene_fast(
            handles,
            obs,
            left_ee,
            left_ee_quat,
            right_ee,
            right_ee_quat,
            joint_name_map,
            urdf_joint_names,
        )

        force_live_shear_refresh = bool(
            WORKING_PROBE_DEMO
            or shear_csv_recording
            or object_slide_trial is not None
        )
        if force_live_shear_refresh or step % slow_interval == 0:
            update_scene_slow(
                handles,
                obs,
                left_ee,
                right_ee,
                left_gripper,
                right_gripper,
                step,
                recording,
                recorder.num_steps,
                shear_csv_recording,
                shear_csv_recorder.num_frames,
                shear_csv_recorder.num_rows,
                hydroshear_elastomer_source_summary(env),
            )

        if step % 120 == 0:
            tac = [float(obs["tactile"][i].max()) for i in range(4)]
            rec_tag = f" REC({recorder.num_steps})" if recording else ""
            shear_csv_tag = f" SHEAR_CSV({shear_csv_recorder.num_frames})" if shear_csv_recording else ""
            print(
                f"[step {step:06d}]"
                f" L_ee=({left_ee[0]:.3f},{left_ee[1]:.3f},{left_ee[2]:.3f})"
                f" R_ee=({right_ee[0]:.3f},{right_ee[1]:.3f},{right_ee[2]:.3f})"
                f" L_grip={left_gripper:.2f}"
                f" R_grip={right_gripper:.2f}"
                f" tactile={','.join(f'{v:.4f}' for v in tac)}"
                f"{rec_tag}"
                f"{shear_csv_tag}",
                flush=True,
            )

        if (
            auto_probe_complete
            and not auto_probe_visual_demo
            and object_slide_trial is None
            and not shear_csv_recorder.active
        ):
            print("[AUTO] Object slide probe complete; exiting.", flush=True)
            break

        step += 1

    if recorder.num_steps > 0:
        recorder.flush()
    if shear_csv_recorder.active:
        shear_csv_recorder.stop()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
