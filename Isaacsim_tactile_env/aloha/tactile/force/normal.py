from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LinearNormalForceCfg:
    stiffness: float = 5_000.0
    max_force: float = 10.0
    normalize_forces: bool = True


@dataclass
class NormalForceOutput:
    normal_force: Any


class LinearNormalForceModel:
    """Penalty normal force: fn = clamp(stiffness * penetration, 0, max_force)."""

    def __init__(self, cfg: LinearNormalForceCfg):
        self.cfg = cfg

    def compute(self, penetration) -> NormalForceOutput:
        force = self.cfg.stiffness * penetration
        if hasattr(force, "clamp"):
            force = force.clamp(0.0, self.cfg.max_force)
        else:
            force = np.clip(force, 0.0, self.cfg.max_force)

        if self.cfg.normalize_forces:
            force = force / self.cfg.max_force
        return NormalForceOutput(normal_force=force)
