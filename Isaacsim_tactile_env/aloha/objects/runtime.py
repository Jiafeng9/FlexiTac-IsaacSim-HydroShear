from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..helpers.spatial import obj_pose_numpy, xyzw_to_wxyz


@dataclass
class AlohaObjectsCfg:
    enable_plug: bool = True
    enable_socket: bool = True
    use_cube_objects: bool = False
    asset_root: str = Path(__file__).resolve().parent.parent.parent / "assets"
    automate_asset_id: str = "00186"
    plug_fix_base: bool = False
    socket_fix_base: bool = False
    plug_scale: float = 1.06
    socket_scale: float = 1.0
    cube_size: tuple[float, float, float] = (0.026, 0.026, 0.026)
    cube_mass: float = 0.04
    plug_collider_type: str = "convex_decomposition"
    socket_collider_type: str = "convex_decomposition"
    force_urdf_conversion: bool = True
    plug_default_pose: tuple[float, float, float, float, float, float, float] = (
        0.0,
        +0.05,
        +0.003,
        0.0,
        0.0,
        +1.0,
        0.0,
    )
    socket_default_pose: tuple[float, float, float, float, float, float, float] = (
        0.0,
        -0.05,
        +0.003,
        0.0,
        0.0,
        +1.0,
        0.0,
    )


@dataclass
class AlohaObjectsOutput:
    plug_obj: object | None
    socket_obj: object | None
    plug_pose: np.ndarray
    socket_pose: np.ndarray


class PlugSocketObjects:
    """Plug/socket spawning and lifecycle."""

    def __init__(self, cfg, sim_utils, RigidObject, RigidObjectCfg, base_dir: str):
        self.cfg = cfg
        self.objects_cfg = cfg.objects
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self.plug_obj = None
        self.socket_obj = None
        self._spawn(sim_utils, RigidObject, RigidObjectCfg, base_dir)

    def _spawn(self, sim_utils, RigidObject, RigidObjectCfg, base_dir: str):
        obj_cfg = self.objects_cfg
        objs_out_dir = os.path.join(base_dir, "output", "automate_scaled_urdf")
        os.makedirs(objs_out_dir, exist_ok=True)

        if not (obj_cfg.enable_plug or obj_cfg.enable_socket):
            return

        if bool(getattr(obj_cfg, "use_cube_objects", False)):
            self._spawn_cube_objects(sim_utils, RigidObject, RigidObjectCfg)
            return

        automate_dir = os.path.join(os.path.expanduser(obj_cfg.asset_root), "automate_scaled", "urdf")
        plug_urdf = os.path.join(automate_dir, f"{obj_cfg.automate_asset_id}_plug.urdf")
        socket_urdf = os.path.join(automate_dir, f"{obj_cfg.automate_asset_id}_socket.urdf")

        def make_spawn_cfg(urdf_path: str, scale: float, fix_base: bool, collider_type: str):
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                scale=(scale,) * 3 if scale != 1.0 else None,
                fix_base=fix_base,
                joint_drive=None,
                link_density=1000.0,
                usd_dir=objs_out_dir,
                force_usd_conversion=obj_cfg.force_urdf_conversion,
                collider_type=collider_type,
                activate_contact_sensors=False,
            )

            try:
                art_root_cfg = None
                try:
                    from isaaclab.sim.schemas import ArticulationRootPropertiesCfg

                    art_root_cfg = ArticulationRootPropertiesCfg(articulation_enabled=False)
                except Exception:
                    try:
                        from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg

                        art_root_cfg = ArticulationRootPropertiesCfg(articulation_enabled=False)
                    except Exception:
                        art_root_cfg = None

                if art_root_cfg is not None:
                    if hasattr(spawn_cfg, "articulation_props"):
                        spawn_cfg.articulation_props = art_root_cfg
                    elif hasattr(spawn_cfg, "articulation_root_props"):
                        spawn_cfg.articulation_root_props = art_root_cfg
                    elif hasattr(spawn_cfg, "usd_config") and hasattr(spawn_cfg.usd_config, "articulation_props"):
                        spawn_cfg.usd_config.articulation_props = art_root_cfg
            except Exception as e:
                print(f"[WARN] Could not attach articulation-root fix for {os.path.basename(urdf_path)}: {e}", flush=True)

            return spawn_cfg

        def spawn_one(urdf_path: str, pose, prim_path: str, scale: float, fix_base: bool, collider_type: str):
            if not os.path.isfile(urdf_path):
                return None
            pos = tuple(float(v) for v in pose[:3])
            rot = xyzw_to_wxyz(pose[3:7])
            return RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=make_spawn_cfg(urdf_path, scale, fix_base, collider_type),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            )

        if obj_cfg.enable_plug:
            self.plug_obj = spawn_one(
                plug_urdf,
                obj_cfg.plug_default_pose,
                "/World/Plug",
                obj_cfg.plug_scale,
                obj_cfg.plug_fix_base,
                obj_cfg.plug_collider_type,
            )

        if obj_cfg.enable_socket:
            self.socket_obj = spawn_one(
                socket_urdf,
                obj_cfg.socket_default_pose,
                "/World/Socket",
                obj_cfg.socket_scale,
                obj_cfg.socket_fix_base,
                obj_cfg.socket_collider_type,
            )

    def _spawn_cube_objects(self, sim_utils, RigidObject, RigidObjectCfg):
        obj_cfg = self.objects_cfg

        def make_cube_spawn_cfg(fix_base: bool, color: tuple[float, float, float]):
            return sim_utils.MeshCuboidCfg(
                size=tuple(float(v) for v in obj_cfg.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=bool(fix_base),
                    disable_gravity=bool(fix_base),
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(obj_cfg.cube_mass)),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    roughness=0.65,
                ),
            )

        def spawn_one(pose, prim_path: str, fix_base: bool, color: tuple[float, float, float]):
            pos = tuple(float(v) for v in pose[:3])
            rot = xyzw_to_wxyz(pose[3:7])
            return RigidObject(
                RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=make_cube_spawn_cfg(fix_base, color),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            )

        if obj_cfg.enable_plug:
            self.plug_obj = spawn_one(
                obj_cfg.plug_default_pose,
                "/World/Plug",
                obj_cfg.plug_fix_base,
                (0.18, 0.38, 0.95),
            )

        if obj_cfg.enable_socket:
            self.socket_obj = spawn_one(
                obj_cfg.socket_default_pose,
                "/World/Socket",
                obj_cfg.socket_fix_base,
                (0.95, 0.48, 0.16),
            )

    def reset(self):
        for obj in (self.plug_obj, self.socket_obj):
            if obj:
                obj.write_root_state_to_sim(obj.data.default_root_state)
                obj.reset()

    def update(self, dt: float):
        for obj in (self.plug_obj, self.socket_obj):
            if obj:
                obj.update(dt)

    def target_root_for_arm(self, arm: str | None) -> str:
        if arm == "left" and self.socket_obj:
            return self.tactile_cfg.left_arm_target_mesh_prim
        if arm == "right" and self.plug_obj:
            return self.tactile_cfg.right_arm_target_mesh_prim
        return self.robot_cfg.prim_path

    def output(self) -> AlohaObjectsOutput:
        return AlohaObjectsOutput(
            plug_obj=self.plug_obj,
            socket_obj=self.socket_obj,
            plug_pose=obj_pose_numpy(self.plug_obj),
            socket_pose=obj_pose_numpy(self.socket_obj),
        )
