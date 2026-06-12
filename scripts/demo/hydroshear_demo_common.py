from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np


warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 .*charset_normalizer.*",
    category=Warning,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TACTILE_ROOT = REPO_ROOT / "Isaacsim_tactile_env"
TOOLS_ROOT = TACTILE_ROOT / "tools"


def add_repo_paths() -> None:
    for path in (str(TACTILE_ROOT), str(TOOLS_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def make_probe_args(**overrides):
    defaults = dict(
        normal_axis=0,
        num_rows=12,
        num_cols=32,
        point_distance=0.002,
        normal_offset=0.0,
        surface_rows=30,
        surface_cols=70,
        clearance=0.001,
        penetration=0.002,
        slide=0.004,
        press_steps=10,
        slide_steps=36,
        hold_steps=8,
        gif_stride=2,
        lambda_s=10_800.0,
        mu=10.0,
        output_dir=str(TACTILE_ROOT / "output" / "hydroshear_demo"),
        basename="hydroshear_demo",
        num_points=2048,
        seed=7,
        cube_side=0.018,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def build_case(*, complex_geometry: bool, channel: int, **overrides) -> dict:
    add_repo_paths()
    args = make_probe_args(**overrides)
    print(
        f"[HydroShear demo] building pytorch_volumetric SDF case "
        f"({'complex cube' if complex_geometry else 'flat slab'}, channel={int(channel)})...",
        flush=True,
    )
    if complex_geometry:
        from visualize_hydroshear_cube_axis_probe import run_dynamic_case
    else:
        from visualize_hydroshear_dynamic_axis_probe import run_dynamic_case
    case = run_dynamic_case(args, int(channel))
    print(f"[HydroShear demo] generated {len(case['frames'])} frames.", flush=True)
    return case


def select_frame(case: dict, step: int | None = None) -> dict:
    frames = case["frames"]
    if not frames:
        raise RuntimeError("HydroShear demo produced no frames")
    if step is None:
        for i, frame in enumerate(frames):
            if frame.get("phase") == "slide":
                return frames[min(i + max(1, len(frames) // 4), len(frames) - 1)]
        return frames[-1]
    return frames[max(0, min(int(step), len(frames) - 1))]


def signed_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vmax = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0e-12)
    signed = np.clip(values / vmax, -1.0, 1.0)
    mag = np.abs(signed)
    base = (255.0 * (1.0 - mag)).astype(np.uint8)
    colors = np.empty((values.shape[0], 3), dtype=np.uint8)
    positive = signed >= 0.0
    colors[:, 0] = np.where(positive, 255, base)
    colors[:, 1] = base
    colors[:, 2] = np.where(positive, base, 255)
    return colors


def penetration_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vmax = max(float(np.max(values)) if values.size else 0.0, 1.0e-12)
    x = np.clip(values / vmax, 0.0, 1.0)
    colors = np.zeros((values.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = (255.0 * x).astype(np.uint8)
    colors[:, 1] = (80.0 + 120.0 * (1.0 - x)).astype(np.uint8)
    colors[:, 2] = (255.0 * (1.0 - x)).astype(np.uint8)
    return colors


def shear_segments(frame: dict, *, max_vectors: int = 220, length: float = 0.006) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(frame["contact_points"], dtype=np.float64)
    shear = np.asarray(frame["surface_shear_e"], dtype=np.float64)
    mask = np.asarray(frame["contact_mask"], dtype=bool)
    mag = np.linalg.norm(shear, axis=-1)
    ids = np.flatnonzero(mask & (mag > 1.0e-12))
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    if ids.size > max_vectors:
        ids = ids[:: max(1, ids.size // max_vectors)]
    scale = float(length) / max(float(np.max(mag[ids])), 1.0e-12)
    starts = points[ids]
    ends = starts + shear[ids] * scale
    return starts, ends


def taxel_points_and_colors(frame: dict, *, channel: int) -> tuple[np.ndarray, np.ndarray]:
    taxel_points = np.asarray(frame["taxel_points"], dtype=np.float64).reshape(-1, 3)
    shear_grid = np.asarray(frame["shear_grid"], dtype=np.float64)
    colors = signed_colors(shear_grid[..., int(channel)].reshape(-1))
    return taxel_points, colors


def contact_points_and_colors(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(frame["contact_points"], dtype=np.float64)
    mask = np.asarray(frame["contact_mask"], dtype=bool)
    penetration = np.asarray(frame["penetration"], dtype=np.float64)
    return points[mask], penetration_colors(penetration[mask])
