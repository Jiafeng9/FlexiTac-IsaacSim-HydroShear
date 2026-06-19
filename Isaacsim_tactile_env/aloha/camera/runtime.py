from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..helpers.spatial import look_at_quat


@dataclass
class AlohaCameraCfg:
    enable_camera: bool = True
    width: int = 640
    height: int = 480
    prim_path: str = "/World/Camera"
    eye: tuple[float, float, float] = (-0.9, 0.0, 0.4)
    target: tuple[float, float, float] = (0.0, 0.0, 0.1)
    save_renders: bool = False
    render_output_dir: str = ""


@dataclass
class AlohaCameraOutput:
    rgb: np.ndarray | None


class AlohaCamera:
    """Optional scene camera and render dumping."""

    def __init__(self, cfg, sim_utils, base_dir: str):
        self.cfg = cfg
        self.camera_cfg = cfg.camera
        self.camera = None
        self.render_output_dir = None
        self._base_dir = base_dir
        self._warned_unready = False
        self._setup(sim_utils)

    def _setup(self, sim_utils):
        camera_cfg = self.camera_cfg
        if not camera_cfg.enable_camera:
            return

        from isaacsim.core.utils.viewports import set_camera_view
        from isaaclab.sensors.camera import Camera, CameraCfg

        set_camera_view(list(camera_cfg.eye), list(camera_cfg.target))
        cam_rot = look_at_quat(camera_cfg.eye, camera_cfg.target)

        self.camera = Camera(
            CameraCfg(
                prim_path=camera_cfg.prim_path,
                update_period=0.0,
                height=camera_cfg.height,
                width=camera_cfg.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 1.0e5),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=camera_cfg.eye,
                    rot=cam_rot,
                    convention="world",
                ),
            )
        )

        if camera_cfg.save_renders:
            self.render_output_dir = camera_cfg.render_output_dir or os.path.join(self._base_dir, "output", "renders")
            os.makedirs(self.render_output_dir, exist_ok=True)

    def reset(self):
        if not self._camera_ready():
            return
        try:
            self.camera.reset()
        except Exception as e:
            self._warn_camera_unready(f"reset failed: {e}")

    def update(self, dt: float):
        if not self._camera_ready():
            return
        try:
            self.camera.update(dt=dt)
        except Exception as e:
            self._warn_camera_unready(f"update failed: {e}")

    def save_render(self, rgb: np.ndarray | None, step: int):
        if rgb is None or not self.render_output_dir:
            return
        try:
            from PIL import Image

            Image.fromarray(rgb).save(os.path.join(self.render_output_dir, f"frame_{step:06d}.png"))
        except ImportError:
            np.save(os.path.join(self.render_output_dir, f"frame_{step:06d}.npy"), rgb)

    def output(self) -> AlohaCameraOutput:
        if not self._camera_ready():
            return AlohaCameraOutput(rgb=None)
        try:
            rgb = self.camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
        except Exception:
            rgb = None
        return AlohaCameraOutput(rgb=rgb)

    def _camera_ready(self) -> bool:
        return bool(self.camera is not None and getattr(self.camera, "is_initialized", False) and hasattr(self.camera, "_timestamp"))

    def _warn_camera_unready(self, reason: str):
        if self._warned_unready:
            return
        self._warned_unready = True
        print(f"[WARN] Camera sensor unavailable; continuing without live RGB ({reason})", flush=True)
