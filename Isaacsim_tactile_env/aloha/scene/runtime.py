from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlohaSimCfg:
    physics_dt: float = 1.0 / 120.0
    device: str = "cuda:0"
    headless: bool = True


@dataclass
class AlohaSceneOutput:
    sim: object
    device: str


class AlohaScene:
    """Simulation context and static world setup."""

    def __init__(self, cfg, SimulationContext):
        sim_cfg = cfg.sim
        self.sim = SimulationContext(
            physics_dt=sim_cfg.physics_dt,
            rendering_dt=sim_cfg.physics_dt,
            backend="torch",
            device=sim_cfg.device,
        )
        self.device = sim_cfg.device

    def output(self) -> AlohaSceneOutput:
        return AlohaSceneOutput(sim=self.sim, device=self.device)

    def spawn_basic_world(self, sim_utils) -> None:
        sim_utils.spawn_mesh_cuboid(
            prim_path="/World/defaultGroundPlane",
            cfg=sim_utils.MeshCuboidCfg(
                size=(10.0, 10.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.004,
                    rest_offset=0.0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            ),
            translation=(0.0, 0.0, -0.05),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        sim_utils.spawn_light(
            prim_path="/World/Light/DomeLight",
            cfg=sim_utils.DomeLightCfg(intensity=2000),
            translation=(-4.5, 3.5, 10.0),
        )
