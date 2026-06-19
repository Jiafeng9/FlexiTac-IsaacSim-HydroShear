from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..helpers.spatial import resolve_mesh_prim, rigid_prim_world_pose
from ..helpers.structure import infer_arm, infer_finger, sensor_slot
from .backend import (
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
)


@dataclass
class AlohaTactileOutput:
    observations: dict[str, np.ndarray]
    selected_links: tuple[str, ...]
    target_query_paths: tuple[str, ...]
    sensor_slot_order: tuple[int, ...]


class AlohaPatchTransform:
    """Compute the sensor patch transform in each elastomer body frame."""

    def __init__(self, tactile_cfg, urdf_origins: dict, math_utils):
        self.tactile_cfg = tactile_cfg
        self.urdf_origins = urdf_origins
        self.math_utils = math_utils

    def compute(self, link_path: str):
        import torch

        lp = link_path.lower()
        if "_left_finger_link" in lp:
            side = "left"
        elif "_right_finger_link" in lp:
            side = "right"
        else:
            return self.tactile_cfg.patch_offset_pos, self.tactile_cfg.patch_offset_quat

        base_xyz, base_rpy = self.urdf_origins[side]

        base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
        base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
        user_pos_t = torch.tensor(self.tactile_cfg.patch_offset_pos, dtype=torch.float32).unsqueeze(0)
        user_quat_t = torch.tensor(self.tactile_cfg.patch_offset_quat, dtype=torch.float32).unsqueeze(0)

        q_be = self.math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
        pos_bp = (base_xyz_t + self.math_utils.quat_apply(q_be, user_pos_t)).squeeze(0)
        quat_bp = self.math_utils.quat_mul(q_be, user_quat_t).squeeze(0)

        return tuple(float(v) for v in pos_bp), tuple(float(v) for v in quat_bp)


class AlohaTactileBinding:
    """Discover ALOHA tactile links and bind each link to a target mesh."""

    def __init__(self, cfg, sim_utils, prim_utils, objects, UsdPhysics, PhysxSchema):
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self.sim_utils = sim_utils
        self.prim_utils = prim_utils
        self.objects = objects
        self.UsdPhysics = UsdPhysics
        self.PhysxSchema = PhysxSchema

    def build(self) -> tuple[list[str], list[str], list[str]]:
        selected_links = self.sort_elastomer_links(self.find_elastomer_links())
        target_root_paths, target_query_paths = self.build_target_query_paths(selected_links)
        for i, (link, root, query) in enumerate(zip(selected_links, target_root_paths, target_query_paths)):
            print(
                f"  [{i}] slot={sensor_slot(link)} arm={infer_arm(link) or '?':>5s}"
                f" elastomer={link} -> query={query}",
                flush=True,
            )
        return selected_links, target_root_paths, target_query_paths

    def find_elastomer_links(self) -> list[str]:
        bodies = self.sim_utils.get_all_matching_child_prims(
            self.robot_cfg.prim_path,
            predicate=lambda p: p.HasAPI(self.UsdPhysics.RigidBodyAPI)
            and p.HasAPI(self.PhysxSchema.PhysxContactReportAPI),
            traverse_instance_prims=False,
        )
        name_token = self.tactile_cfg.link_name_contains.lower()
        elastomers = sorted(
            [p.GetPath().pathString for p in bodies if name_token in p.GetPath().pathString.lower()]
        )[: self.tactile_cfg.max_elastomers]
        if not elastomers:
            raise RuntimeError("No elastomer links found on robot.")
        return elastomers

    def sort_elastomer_links(self, elastomers: list[str]) -> list[str]:
        def sort_key(path: str):
            arm = infer_arm(path)
            arm_order = 0 if arm == "left" else 1

            finger = infer_finger(path)
            finger_order = 0 if finger == "left_finger" else 1

            return arm_order, finger_order, path

        return sorted(elastomers, key=sort_key)

    def build_target_query_paths(self, selected_links: list[str]) -> tuple[list[str], list[str]]:
        target_root_paths = [self.objects.target_root_for_arm(infer_arm(link_path)) for link_path in selected_links]

        target_query_paths: list[str] = []
        for root in target_root_paths:
            try:
                qp, _ = resolve_mesh_prim(root, prim_utils=self.prim_utils, sim_utils=self.sim_utils)
            except RuntimeError:
                qp = root
            target_query_paths.append(qp)

        return target_root_paths, target_query_paths


@dataclass
class TrackInfo:
    rp: object
    p_rel: Any
    q_rel: Any
    rb_path: str


class AlohaTargetTracker:
    """Track target mesh poses and push them into tactile sensors."""

    def __init__(self, target_query_paths: list[str], sim_utils, math_utils, UsdPhysics, device: str):
        self.target_query_paths = target_query_paths
        self.sim_utils = sim_utils
        self.math_utils = math_utils
        self.UsdPhysics = UsdPhysics
        self.device = device
        self.stage = None
        self.per_sensor_target_prims = []
        self.dynamic_track_map = {}

    def initialize(self, stage):
        self.stage = stage
        self.per_sensor_target_prims = [stage.GetPrimAtPath(p) for p in self.target_query_paths]
        self.setup_dynamic_tracking()

    def setup_dynamic_tracking(self):
        RigidPrim = None
        for mod in ("omni.isaac.core.prims", "isaacsim.core.prims"):
            try:
                RigidPrim = __import__(mod, fromlist=["RigidPrim"]).RigidPrim
                break
            except ImportError:
                continue

        self.dynamic_track_map = {}
        if RigidPrim is None:
            return

        def make_rigid_prim(path: str):
            candidates = (
                ((), {}),
                ((), {"prim_path": path}),
                ((), {"path": path}),
                ((), {"name": path.replace("/", "_")}),
            )

            try:
                return RigidPrim(path)
            except TypeError:
                pass

            last_err = None
            for _, kwargs in candidates:
                try:
                    if "prim_path" not in kwargs and "path" not in kwargs:
                        continue
                    return RigidPrim(**kwargs)
                except TypeError as e:
                    last_err = e
                    continue

            if last_err is not None:
                return RigidPrim(prim_path=path)
            return RigidPrim(prim_path=path)

        def to_t(x):
            import torch

            return torch.tensor(x, device=self.device, dtype=torch.float32)

        for query_path in self.target_query_paths:
            if not query_path:
                continue

            prim = self.stage.GetPrimAtPath(query_path)
            if not prim.IsValid():
                continue

            curr = prim
            rb_prim = None
            while curr.IsValid() and not curr.IsPseudoRoot():
                if curr.HasAPI(self.UsdPhysics.RigidBodyAPI) or curr.HasAPI(self.UsdPhysics.MassAPI):
                    rb_prim = curr
                    break
                curr = curr.GetParent()
            if rb_prim is None:
                continue

            rb_path = rb_prim.GetPath().pathString

            try:
                rp = make_rigid_prim(rb_path)
                if hasattr(rp, "initialize"):
                    rp.initialize()

                p_m, q_m = self.sim_utils.resolve_prim_pose(prim)
                p_b, q_b = self.sim_utils.resolve_prim_pose(rb_prim)

                q_b_inv = self.math_utils.quat_inv(to_t(q_b))
                p_rel = self.math_utils.quat_apply(q_b_inv, to_t(p_m) - to_t(p_b))
                q_rel = self.math_utils.quat_mul(q_b_inv, to_t(q_m))

                self.dynamic_track_map[query_path] = TrackInfo(rp=rp, p_rel=p_rel, q_rel=q_rel, rb_path=rb_path)

            except Exception as e:
                print(f"[WARN] Dynamic tracking init failed for {rb_path}: {e}", flush=True)

    def tracked_target_pose(self, info: TrackInfo):
        import torch

        pos_b, quat_b = rigid_prim_world_pose(info.rp)

        pos_b_t = torch.tensor(pos_b, device=self.device, dtype=torch.float32)
        quat_b_t = torch.tensor(quat_b, device=self.device, dtype=torch.float32)

        pos_t = pos_b_t + self.math_utils.quat_apply(quat_b_t, info.p_rel)
        quat_t = self.math_utils.quat_mul(quat_b_t, info.q_rel)

        return pos_t.detach().cpu().numpy(), quat_t.detach().cpu().numpy()

    def target_pose_for_sensor(self, sensor_index: int):
        tgt_prim = self.per_sensor_target_prims[sensor_index]
        if not tgt_prim or not tgt_prim.IsValid():
            raise RuntimeError(f"Invalid target prim for tactile sensor {sensor_index}")

        path = tgt_prim.GetPath().pathString
        info = self.dynamic_track_map.get(path)
        if info is not None:
            try:
                return self.tracked_target_pose(info)
            except Exception:
                pass
        return self.sim_utils.resolve_prim_pose(tgt_prim)

    def update_target_poses(self, sensors):
        for i, (sensor, tgt_prim) in enumerate(zip(sensors, self.per_sensor_target_prims)):
            if not tgt_prim or not tgt_prim.IsValid():
                continue

            pos, quat = self.target_pose_for_sensor(i)
            sensor.set_target_pose(pos, quat)


class AlohaTactileSetup:
    """ALOHA adapter for selectable tactile backends."""

    def __init__(
        self,
        cfg,
        sim_utils,
        prim_utils,
        math_utils,
        robot_asset,
        objects,
        urdf_origins: dict,
        UsdPhysics,
        PhysxSchema,
        device: str,
    ):
        self.cfg = cfg
        self.sim_utils = sim_utils
        self.robot_asset = robot_asset
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self._device = device

        self.binding = AlohaTactileBinding(cfg, sim_utils, prim_utils, objects, UsdPhysics, PhysxSchema)
        self.selected_links, self.target_root_paths, self.target_query_paths = self.binding.build()

        self.patch_transform = AlohaPatchTransform(self.tactile_cfg, urdf_origins, math_utils)
        self.backend = self._create_backend()
        self.sensors, self.sensor_slot_order = self.backend.create_sensors(
            self.selected_links,
            self.target_query_paths,
        )

        self.target_tracker = AlohaTargetTracker(self.target_query_paths, sim_utils, math_utils, UsdPhysics, device)
        self.per_sensor_target_prims = []
        self.dynamic_track_map = {}

    def _create_backend(self):
        backend_cfg = self.tactile_cfg.backend
        if isinstance(backend_cfg, HydroShearTactileBackendCfg):
            return HydroShearTactileBackend(self.cfg, self.patch_transform, self.robot_asset, self._device)
        raise TypeError(f"Unsupported tactile backend cfg: {type(backend_cfg).__name__}")

    def initialize_after_sim_reset(self, stage):
        self.target_tracker.initialize(stage)
        self.per_sensor_target_prims = self.target_tracker.per_sensor_target_prims
        self.dynamic_track_map = self.target_tracker.dynamic_track_map
        self.backend.initialize_after_sim_reset(self.sensors, stage, self.target_tracker)

    def update_target_poses(self):
        pass

    def update(self, dt: float):
        self.backend.update(dt, self.sensors, self.target_tracker)

    def reset(self):
        self.backend.reset(self.sensors)

    def close(self):
        self.backend.close(self.sensors)

    def normal_force_grid(self) -> np.ndarray:
        return self.backend.observations(self.sensors, self.sensor_slot_order)[self.tactile_cfg.output_key]

    def output(self) -> AlohaTactileOutput:
        observations = self.backend.observations(self.sensors, self.sensor_slot_order)
        return AlohaTactileOutput(
            observations=observations,
            selected_links=tuple(self.selected_links),
            target_query_paths=tuple(self.target_query_paths),
            sensor_slot_order=tuple(self.sensor_slot_order),
        )
