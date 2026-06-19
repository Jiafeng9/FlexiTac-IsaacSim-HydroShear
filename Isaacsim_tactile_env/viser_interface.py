from __future__ import annotations

import argparse
import os
import site
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

# --- SimulationApp MUST be created before any omni/isaaclab imports ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ALOHA bimanual drag control and visualization")
parser.add_argument("--port", type=int, default=8080, help="Viser server port")
parser.add_argument("--slow_interval", type=int, default=8, help="UI refresh interval for expensive updates")
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
    "--shear_arrow_scale",
    type=float,
    default=750.0,
    help="3D shear arrow display scale in meters per shear unit.",
)
parser.add_argument(
    "--shear_arrow_max_len",
    type=float,
    default=0.15,
    help="Maximum 3D shear arrow length in meters after display scaling.",
)
parser.add_argument(
    "--shear_arrow_min_mag",
    type=float,
    default=0.0,
    help="Hide 3D shear arrows with magnitude below this value.",
)
parser.add_argument(
    "--shear_arrow_normal_lift",
    type=float,
    default=0.008,
    help="Lift 3D shear arrows along the tactile pad normal in meters.",
)
parser.add_argument(
    "--shear_arrow_world_z_lift",
    type=float,
    default=0.25,
    help="Additional world +Z lift for the 3D shear arrow field in meters.",
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
    "--save_camera_renders",
    action="store_true",
    help="Save camera RGB frames from the simulation loop.",
)
parser.add_argument(
    "--render_output_dir",
    type=str,
    default="",
    help="Directory for saved camera frames. Default: output/renders.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Stop after this many interactive-loop steps. Use -1 to run until closed.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


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
from aloha.tactile import HydroShearTactileBackendCfg


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
    '''
        Based on the normal axis, get the other two tangential axis
    '''
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


def _elastomer_meshes_from_urdf(urdf_path: str) -> list[str]:
    urdf_path = os.path.expanduser(urdf_path)
    if not os.path.isfile(urdf_path):
        return []
    meshes: list[str] = []
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if "elastomer" not in link_name:
            continue
        for tag in ("visual", "collision"):
            for mesh_el in link.findall(f"./{tag}/geometry/mesh"):
                filename = mesh_el.get("filename", "")
                if filename:
                    meshes.append(f"{link_name}:{tag}:{filename}")
    return meshes


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


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
        self._show_shear_arrows = True

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

    def set_show_shear_arrows(self, v):
        with self._lock:
            self._show_shear_arrows = bool(v)

    def is_show_shear_arrows(self):
        with self._lock:
            return self._show_shear_arrows


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

    show_shear_arrows = server.gui.add_checkbox("Show shear arrows", initial_value=True)

    @show_shear_arrows.on_update
    def _(_):
        shared.set_show_shear_arrows(bool(show_shear_arrows.value))

    shared.set_show_shear_arrows(bool(show_shear_arrows.value))

    main_shear_enabled = bool(getattr(cfg.tactile.backend, "include_force_observations", False))
    bump_enabled = bool(getattr(cfg.tactile.backend, "bump_enabled", False))
    main_backend_is_hydroshear = "HydroShear" in type(cfg.tactile.backend).__name__
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
    main_shear_x_handles = []
    main_shear_y_handles = []
    main_shear_mag_handles = []
    if main_shear_enabled:
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

    with server.gui.add_folder("State", expand_by_default=False):
        state_md = server.gui.add_markdown("**Loading...**")

    with server.gui.add_folder("Sim Camera", expand_by_default=False):
        camera_img = server.gui.add_image(
            np.zeros((480, 640, 3), dtype=np.uint8),
            label="Camera Feed", format="jpeg", jpeg_quality=70,
        )

    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, -0.05))
    tactile_shear_arrows = server.scene.add_arrows(
        "/tactile_shear_vectors",
        points=np.zeros((128, 2, 3), dtype=np.float32),
        colors=np.tile(np.array([[80, 120, 255]], dtype=np.uint8), (128, 1)),
        shaft_radius=0.002,
        head_radius=0.006,
        head_length=0.012,
        visible=True,
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
        "tactile_shear_arrows": tactile_shear_arrows,
        "tac_handles": tac_handles,
        "main_shear_x_handles": main_shear_x_handles,
        "main_shear_y_handles": main_shear_y_handles,
        "main_shear_mag_handles": main_shear_mag_handles,
        "state_md": state_md,
        "camera_img": camera_img,
        "bump_enabled": bump_enabled,
        "main_backend_is_hydroshear": main_backend_is_hydroshear,
        "last_shear_arrow_count": 0,
        "last_shear_arrow_max_mag": 0.0,
        "shear_arrow_capacity": 128,
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


def _grid_center_offsets(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    pitch_m = 0.004
    u = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0) * pitch_m
    v = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0) * pitch_m
    return u, v


def _shear_arrow_colors(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float64).reshape(-1)
    if mag.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmax = float(np.max(mag))
    if vmax <= 1.0e-12:
        t = np.zeros_like(mag)
    else:
        t = np.clip(mag / vmax, 0.0, 1.0)

    anchors = np.array(
        [
            [80, 120, 255],
            [0, 210, 255],
            [70, 220, 110],
            [255, 210, 60],
            [255, 70, 45],
        ],
        dtype=np.float64,
    )
    scaled = t * float(len(anchors) - 1)
    idx = np.floor(scaled).astype(np.int64)
    idx = np.clip(idx, 0, len(anchors) - 2)
    frac = (scaled - idx).reshape(-1, 1)
    colors = anchors[idx] * (1.0 - frac) + anchors[idx + 1] * frac
    return colors.astype(np.uint8)


def update_tactile_shear_vector_visuals(
    handles,
    obs,
    pad_axes_w: dict[int, dict[str, np.ndarray]],
    *,
    visible: bool,
):
    handle = handles.get("tactile_shear_arrows")
    if handle is None:
        return
    max_arrows = int(handles.get("shear_arrow_capacity", 128))
    zero_points = np.zeros((max_arrows, 2, 3), dtype=np.float32)
    zero_colors = np.tile(np.array([[80, 120, 255]], dtype=np.uint8), (max_arrows, 1))
    shear = tactile_shear_uv(obs.get("tactile_shear"))
    if not visible or shear is None or not pad_axes_w:
        handles["last_shear_arrow_count"] = 0
        handles["last_shear_arrow_max_mag"] = 0.0
        try:
            handle.points = zero_points
            handle.colors = zero_colors
            handle.visible = bool(visible)
        except Exception:
            pass
        return

    scale = max(0.0, float(getattr(args, "shear_arrow_scale", 50.0)))
    max_len = max(0.0, float(getattr(args, "shear_arrow_max_len", 0.015)))
    min_mag = max(0.0, float(getattr(args, "shear_arrow_min_mag", 1.0e-8)))
    normal_lift = float(getattr(args, "shear_arrow_normal_lift", 0.008))
    world_z_lift = float(getattr(args, "shear_arrow_world_z_lift", 0.02))
    global_lift = np.array([0.0, 0.0, world_z_lift], dtype=np.float64)

    segments = []
    magnitudes = []
    for pad in range(min(4, shear.shape[0])):
        axes = pad_axes_w.get(pad)
        if axes is None:
            continue
        pos = np.asarray(axes.get("position_w", np.zeros(3)), dtype=np.float64).reshape(3)
        normal = _unit_vec(axes.get("normal_w", np.zeros(3)))
        axis_x = _unit_vec(axes.get("axis_x_w", np.zeros(3)))
        axis_y = _unit_vec(axes.get("axis_y_w", np.zeros(3)))
        if not np.any(axis_x) or not np.any(axis_y):
            continue

        grid = np.asarray(shear[pad], dtype=np.float64)
        if grid.ndim != 3 or grid.shape[-1] != 2:
            continue
        rows, cols = int(grid.shape[0]), int(grid.shape[1])
        offsets_u, offsets_v = _grid_center_offsets(rows, cols)
        for r in range(rows):
            for c in range(cols):
                sx = float(grid[r, c, 0])
                sy = float(grid[r, c, 1])
                mag = float(np.hypot(sx, sy))
                if mag < min_mag or scale <= 0.0 or max_len <= 0.0:
                    continue
                vec = axis_x * sx + axis_y * sy
                length = float(np.linalg.norm(vec)) * scale
                if length <= 1.0e-12:
                    continue
                direction = vec / float(np.linalg.norm(vec))
                length = min(length, max_len)
                center = pos + axis_x * offsets_u[r] + axis_y * offsets_v[c]
                start = center + normal * normal_lift + global_lift
                end = start + direction * length
                segments.append((start, end))
                magnitudes.append(mag)

    if not segments:
        handles["last_shear_arrow_count"] = 0
        handles["last_shear_arrow_max_mag"] = (
            float(np.linalg.norm(np.asarray(shear)[..., :2], axis=-1).max()) if shear is not None else 0.0
        )
        try:
            handle.points = zero_points
            handle.colors = zero_colors
            handle.visible = bool(visible)
        except Exception:
            pass
        return

    count = min(len(segments), max_arrows)
    points = zero_points
    colors = zero_colors
    points[:count] = np.asarray(segments[:count], dtype=np.float32)
    colors[:count] = _shear_arrow_colors(np.asarray(magnitudes[:count], dtype=np.float64))
    handles["last_shear_arrow_count"] = int(count)
    handles["last_shear_arrow_max_mag"] = float(np.max(magnitudes)) if magnitudes else 0.0
    try:
        handle.points = points
        handle.colors = colors
        handle.visible = True
    except Exception:
        pass


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
):
    tactile = obs["tactile"]
    tactile_force = obs.get("tactile_force")
    tactile_shear = tactile_shear_uv(obs.get("tactile_shear"))
    contact_count = np.asarray(obs.get("tactile_contact_count", np.zeros(4)), dtype=np.float32).reshape(-1)
    bump_contact_count = np.asarray(obs.get("tactile_bump_contact_count", np.zeros(4)), dtype=np.float32).reshape(-1)
    max_penetration = np.asarray(obs.get("tactile_max_penetration", np.zeros(4)), dtype=np.float32).reshape(-1)
    min_sdf = np.asarray(obs.get("tactile_min_sdf", np.zeros(4)), dtype=np.float32).reshape(-1)

    tac_maxes = []
    force_normal_maxes = []
    shear_x_ranges = []
    shear_y_ranges = []
    shear_mag_maxes = []

    for i, handle in enumerate(handles["tac_handles"]):
        grid = tactile[i]
        grid_normal = tactile_normal_channel(grid)
        tac_maxes.append(float(grid_normal.max()))
        if tactile_force is not None:
            force_normal_grid = tactile_normal_channel(tactile_force[i])
            force_normal_maxes.append(float(force_normal_grid.max()))

        if bool(handles.get("bump_enabled", False)):
            normal_label = "bump normal"
        elif bool(handles.get("main_backend_is_hydroshear", False)):
            normal_label = "hydro normal"
        else:
            normal_label = "normal"
        try:
            handle.image = _label_rgb(
                tactile_to_rgb(grid_normal),
                f"{normal_label} max={_fmt_scalar(float(grid_normal.max()))}",
            )
        except Exception:
            pass

        shear_grid = tactile_shear[i] if tactile_shear is not None else None
        if shear_grid is None:
            continue
        sx = shear_grid[:, :, 0]
        sy = shear_grid[:, :, 1]
        mag = np.linalg.norm(shear_grid, axis=-1)
        shear_x_ranges.append((float(sx.min()), float(sx.max())))
        shear_y_ranges.append((float(sy.min()), float(sy.max())))
        shear_mag_maxes.append(float(mag.max()))

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

    if "rgb" in obs:
        try:
            handles["camera_img"].image = obs["rgb"]
        except Exception:
            pass

    tac_str = ", ".join(_fmt_scalar(m) for m in tac_maxes)
    contact_str = ", ".join(_fmt_scalar(float(v)) for v in contact_count[:4])
    bump_contact_str = ", ".join(_fmt_scalar(float(v)) for v in bump_contact_count[:4])
    max_pen_str = ", ".join(_fmt_scalar(float(v)) for v in max_penetration[:4])
    min_sdf_str = ", ".join(_fmt_scalar(float(v)) for v in min_sdf[:4])
    force_normal_str = ", ".join(_fmt_scalar(m) for m in force_normal_maxes) if force_normal_maxes else "disabled"
    shear_x_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in shear_x_ranges)
        if shear_x_ranges
        else "disabled"
    )
    shear_y_str = (
        ", ".join(f"[{_fmt_scalar(lo)},{_fmt_scalar(hi)}]" for lo, hi in shear_y_ranges)
        if shear_y_ranges
        else "disabled"
    )
    shear_mag_str = ", ".join(_fmt_scalar(m) for m in shear_mag_maxes) if shear_mag_maxes else "disabled"
    force_label = "Bump force" if bool(handles.get("bump_enabled", False)) else "Hydro force"
    shear_vis_vmax = max(0.0, float(getattr(args, "shear_vis_vmax", 0.0)))
    shear_vis_vmax_str = "auto" if shear_vis_vmax <= 0.0 else _fmt_scalar(shear_vis_vmax)
    shear_vis_deadband_str = _fmt_scalar(_shear_vis_deadband())
    shear_arrow_count = int(handles.get("last_shear_arrow_count", 0))
    shear_arrow_max = float(handles.get("last_shear_arrow_max_mag", 0.0))

    handles["state_md"].content = (
        f"**Left EE:** ({left_ee[0]:.4f}, {left_ee[1]:.4f}, {left_ee[2]:.4f})  \n"
        f"**Right EE:** ({right_ee[0]:.4f}, {right_ee[1]:.4f}, {right_ee[2]:.4f})  \n"
        f"**Left gripper:** {left_gripper:.2f}  \n"
        f"**Right gripper:** {right_gripper:.2f}  \n"
        f"**Shear vis:** vmax={shear_vis_vmax_str}, deadband={shear_vis_deadband_str} display-only  \n"
        f"**Shear arrows:** count={shear_arrow_count}, max={_fmt_scalar(shear_arrow_max)}  \n"
        f"**Pad axes:** cyan=shear_x, orange=shear_y, green=normal  \n"
        f"**Contact samples [4 pads]:** [{contact_str}]  \n"
        f"**Contact bumps [4 pads]:** [{bump_contact_str}]  \n"
        f"**Max penetration [4 pads]:** [{max_pen_str}]  \n"
        f"**Min SDF [4 pads]:** [{min_sdf_str}]  \n"
        f"**Tactile max [4 pads]:** [{tac_str}]  \n"
        f"**{force_label} normal max [4 pads]:** [{force_normal_str}]  \n"
        f"**{force_label} shear-x min/max [4 pads]:** [{shear_x_str}]  \n"
        f"**{force_label} shear-y min/max [4 pads]:** [{shear_y_str}]  \n"
        f"**{force_label} |shear| max [4 pads]:** [{shear_mag_str}]  \n"
        f"**Step:** {step}"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = int(args.port)
    camera_enabled = bool(getattr(args, "enable_cameras", False) or bool(args.save_camera_renders))

    cfg = AlohaTactileEnvCfg(
        camera=AlohaCameraCfg(
            enable_camera=camera_enabled,
            save_renders=bool(args.save_camera_renders),
            render_output_dir=str(args.render_output_dir),
        ),
        sim=AlohaSimCfg(
            headless=bool(getattr(args, "headless", True)),
            device=str(getattr(args, "device", "cuda:0")),
        ),
    )
    if args.robot_urdf:
        cfg.robot.urdf_path = str(Path(os.path.expanduser(args.robot_urdf)).resolve())
    elif bool(args.use_bump_pad):
        cfg.robot.urdf_path = str((Path(__file__).resolve().parent / "assets" / "aloha_tactile_bump.urdf").resolve())
    cfg.tactile.backend = HydroShearTactileBackendCfg(
        include_force_observations=True,
        bump_enabled=bool(args.use_bump_pad),
    )

    elastomer_meshes = _elastomer_meshes_from_urdf(str(cfg.robot.urdf_path))
    print(
        f"[INFO] Robot URDF: {cfg.robot.urdf_path} "
        f"use_bump_pad={bool(args.use_bump_pad)} "
        f"bump_enabled={bool(getattr(cfg.tactile.backend, 'bump_enabled', False))}",
        flush=True,
    )
    print(f"[INFO] Elastomer meshes: {elastomer_meshes}", flush=True)

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

    handles["state_md"].content = "**Ready. Drag either gizmo to control the bimanual arms.**"

    print('\n' + '=' * 72, flush=True)
    print(f'[VISER URL] http://localhost:{port}', flush=True)
    print(f'[VISER URL] Open this in your browser: http://localhost:{port}', flush=True)
    print('=' * 72 + '\n', flush=True)

    step = 0
    slow_interval = max(1, int(args.slow_interval))

    while simulation_app.is_running():
        if shared.consume_reset():
            obs, _ = env.reset()
            for _ in range(warmup_steps):
                obs, _, _, _, _ = env.step(init_joint_pos)

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

        # use diffusion IK DifferentialIKController.compute
        left_action = compute_pose(left_ik, left_target_pos, left_target_quat, left_gripper)
        right_action = compute_pose(right_ik, right_target_pos, right_target_quat, right_gripper)
        action = np.concatenate([left_action[:8], right_action[8:]], axis=0).astype(np.float32)
        obs, _, _, _, _ = env.step(action)

        left_ee = left_ik.get_ee_pos()
        left_ee_quat = left_ik.get_ee_quat()
        right_ee = right_ik.get_ee_pos()
        right_ee_quat = right_ik.get_ee_quat()
        pad_axes_w = _read_pad_axes_world(env, cfg)
        update_tactile_shear_vector_visuals(
            handles,
            obs,
            pad_axes_w,
            visible=shared.is_show_shear_arrows(),
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

        if step % slow_interval == 0:
            update_scene_slow(
                handles,
                obs,
                left_ee,
                right_ee,
                left_gripper,
                right_gripper,
                step,
            )

        if step % 120 == 0:
            tac = [float(obs["tactile"][i].max()) for i in range(4)]
            force_shear = tactile_shear_uv(obs.get("tactile_shear"))
            contact_count = np.asarray(obs.get("tactile_contact_count", np.zeros(4)), dtype=np.float32).reshape(-1)
            bump_contact_count = np.asarray(obs.get("tactile_bump_contact_count", np.zeros(4)), dtype=np.float32).reshape(-1)
            max_penetration = np.asarray(obs.get("tactile_max_penetration", np.zeros(4)), dtype=np.float32).reshape(-1)
            min_sdf = np.asarray(obs.get("tactile_min_sdf", np.zeros(4)), dtype=np.float32).reshape(-1)
            force_shear_tag = ""
            if force_shear is not None:
                force_shear_tag = (
                    f" force_shear_absmax="
                    f"({float(np.max(np.abs(force_shear[..., 0]))):.4g},"
                    f"{float(np.max(np.abs(force_shear[..., 1]))):.4g})"
                )
            shape_tag = ""
            if step == 0:
                shape_tag = " shapes=" + ",".join(
                    f"{key}:{tuple(np.asarray(value).shape)}"
                    for key, value in obs.items()
                    if key.startswith("tactile")
                )
            print(
                f"[step {step:06d}]"
                f" L_ee=({left_ee[0]:.3f},{left_ee[1]:.3f},{left_ee[2]:.3f})"
                f" R_ee=({right_ee[0]:.3f},{right_ee[1]:.3f},{right_ee[2]:.3f})"
                f" L_grip={left_gripper:.2f}"
                f" R_grip={right_gripper:.2f}"
                f" tactile={','.join(f'{v:.4f}' for v in tac)}"
                f" contact={','.join(f'{float(v):.0f}' for v in contact_count[:4])}"
                f" bump_contact={','.join(f'{float(v):.0f}' for v in bump_contact_count[:4])}"
                f" max_pen={','.join(f'{float(v):.4g}' for v in max_penetration[:4])}"
                f" min_sdf={','.join(f'{float(v):.4g}' for v in min_sdf[:4])}"
                f"{force_shear_tag}"
                f"{shape_tag}",
                flush=True,
            )

        step += 1
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            print(f"[INFO] Reached --max_steps={int(args.max_steps)}; stopping.", flush=True)
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
