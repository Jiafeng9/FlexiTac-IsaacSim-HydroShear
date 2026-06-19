from __future__ import annotations

from dataclasses import dataclass, field

from .backend import HydroShearTactileBackendCfg


@dataclass
class AlohaTactileCfg:
    """ALOHA tactile binding, shared taxel geometry, and selected backend config."""

    backend: HydroShearTactileBackendCfg = field(default_factory=HydroShearTactileBackendCfg)

    # ALOHA binding
    link_name_contains: str = "elastomer"
    left_arm_target_mesh_prim: str = "/World/Socket"
    right_arm_target_mesh_prim: str = "/World/Plug"
    max_elastomers: int = 4

    # Taxel grid geometry
    num_rows: int = 12
    num_cols: int = 32
    point_distance: float = 0.002
    normal_axis: int = 0
    normal_offset: float = 0.0036
    patch_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    patch_offset_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, -0.7071068)

    # Observation
    output_key: str = "tactile"
