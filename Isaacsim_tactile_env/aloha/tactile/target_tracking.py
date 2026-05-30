from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..helpers.spatial import rigid_prim_world_pose


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
