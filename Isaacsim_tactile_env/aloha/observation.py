from __future__ import annotations

import gymnasium
import numpy as np


class AlohaObservationBuilder:
    """Observation and Gym space construction."""

    def __init__(self, cfg, robot, objects, tactile, camera):
        self.cfg = cfg
        self.robot = robot
        self.objects = objects
        self.tactile = tactile
        self.camera = camera

    def build(self) -> dict:
        cfg = self.cfg
        tactile_cfg = cfg.tactile
        camera_cfg = cfg.camera

        robot_out = self.robot.output()
        objects_out = self.objects.output()
        tactile_out = self.tactile.output()
        camera_out = self.camera.output()

        obs = {
            **tactile_out.observations,
            "joint_pos": robot_out.joint_pos,
            "joint_vel": robot_out.joint_vel,
            "plug_pose": objects_out.plug_pose,
            "socket_pose": objects_out.socket_pose,
        }

        if self.camera.camera:
            obs["rgb"] = (
                camera_out.rgb
                if camera_out.rgb is not None
                else np.zeros((camera_cfg.height, camera_cfg.width, 3), dtype=np.uint8)
            )

        return obs

    def build_spaces(self):
        tactile_cfg = self.cfg.tactile
        camera_cfg = self.cfg.camera
        output_key = tactile_cfg.output_key
        primary_shape = self._tactile_primary_shape(tactile_cfg.backend, tactile_cfg)
        rows, cols = self._tactile_grid_shape(tactile_cfg.backend, tactile_cfg)
        obs_spaces = {
            output_key: gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                shape=(4,) + primary_shape,
                dtype=np.float32,
            ),
            f"{output_key}_contact_count": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            f"{output_key}_bump_contact_count": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            f"{output_key}_max_penetration": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            f"{output_key}_min_sdf": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "joint_pos": gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "joint_vel": gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "plug_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "socket_pose": gymnasium.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        }
        if getattr(tactile_cfg.backend, "include_force_observations", False):
            obs_spaces[f"{output_key}_force"] = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                shape=(4, rows, cols, 3),
                dtype=np.float32,
            )
            obs_spaces[f"{output_key}_shear"] = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                shape=(4, rows, cols, 2),
                dtype=np.float32,
            )
            if getattr(tactile_cfg.backend, "include_taxel_shear_debug_observations", False):
                obs_spaces[f"{output_key}_slip_ratio"] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(4, rows, cols),
                    dtype=np.float32,
                )
                obs_spaces[f"{output_key}_shear_vector_w"] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(4, rows, cols, 3),
                    dtype=np.float32,
                )
        if getattr(tactile_cfg.backend, "include_marker_observations", False):
            for suffix in ("marker", "marker_dilation", "marker_shear"):
                obs_spaces[f"{output_key}_{suffix}"] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(4, rows, cols, 3),
                    dtype=np.float32,
                )
        if getattr(tactile_cfg, "enable_hydro_normal_observation", False):
            hydro_key = tactile_cfg.hydro_normal_output_key
            hydro_primary_shape = self._tactile_primary_shape(tactile_cfg.hydro_normal_backend, tactile_cfg)
            hydro_rows, hydro_cols = self._tactile_grid_shape(tactile_cfg.hydro_normal_backend, tactile_cfg)
            obs_spaces[hydro_key] = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                shape=(4,) + hydro_primary_shape,
                dtype=np.float32,
            )
            obs_spaces[f"{hydro_key}_contact_count"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            obs_spaces[f"{hydro_key}_bump_contact_count"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            obs_spaces[f"{hydro_key}_max_penetration"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            obs_spaces[f"{hydro_key}_min_sdf"] = gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            if getattr(tactile_cfg.hydro_normal_backend, "include_force_observations", False):
                obs_spaces[f"{hydro_key}_force"] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(4, hydro_rows, hydro_cols, 3),
                    dtype=np.float32,
                )
                obs_spaces[f"{hydro_key}_shear"] = gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(4, hydro_rows, hydro_cols, 2),
                    dtype=np.float32,
                )
            if getattr(tactile_cfg.hydro_normal_backend, "include_marker_observations", False):
                for suffix in ("marker", "marker_dilation", "marker_shear"):
                    obs_spaces[f"{hydro_key}_{suffix}"] = gymnasium.spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=(4, hydro_rows, hydro_cols, 3),
                        dtype=np.float32,
                    )
        if self.camera.camera:
            obs_spaces["rgb"] = gymnasium.spaces.Box(
                0,
                255,
                shape=(camera_cfg.height, camera_cfg.width, 3),
                dtype=np.uint8,
            )

        observation_space = gymnasium.spaces.Dict(obs_spaces)
        action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        return observation_space, action_space

    @staticmethod
    def _tactile_grid_shape(backend_cfg, tactile_cfg) -> tuple[int, int]:
        if getattr(backend_cfg, "bump_enabled", False):
            return int(getattr(backend_cfg, "bump_rows", 4)), int(getattr(backend_cfg, "bump_cols", 8))
        return int(tactile_cfg.num_rows), int(tactile_cfg.num_cols)

    @staticmethod
    def _tactile_primary_shape(backend_cfg, tactile_cfg) -> tuple[int, ...]:
        rows, cols = AlohaObservationBuilder._tactile_grid_shape(backend_cfg, tactile_cfg)
        output_mode = getattr(backend_cfg, "output_mode", "force_grid")
        if getattr(backend_cfg, "bump_enabled", False) and output_mode == "marker_field":
            output_mode = "bump_force_grid"
        if output_mode in ("marker_field", "bump_force_grid"):
            return (rows, cols, 3)
        return (rows, cols)
