from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ShearOutput:
    shear_force: Any
    state: Any | None = None


@dataclass
class NoShearTrackerCfg:
    output_dim: int = 2


class NoShearTracker:
    """Null shear model for normal-only tactile pipelines."""

    def __init__(self, cfg: NoShearTrackerCfg | None = None):
        self.cfg = cfg or NoShearTrackerCfg()

    def reset(self):
        pass

    def update(self, tangent_delta, normal_force, contact_mask=None) -> ShearOutput:
        base = np.asarray(normal_force)
        shear = np.zeros(base.shape + (self.cfg.output_dim,), dtype=base.dtype)
        return ShearOutput(shear_force=shear)


@dataclass
class TaxelShearTrackerCfg:
    tangential_stiffness: float = 1_000.0
    friction_coefficient: float = 1.0


class TaxelShearTracker:
    """Per-taxel shear state with Coulomb clipping."""

    def __init__(self, cfg: TaxelShearTrackerCfg):
        self.cfg = cfg
        self._shear = None

    def reset(self, shape=None):
        self._shear = None if shape is None else np.zeros(shape, dtype=np.float32)

    def update(self, tangent_delta, normal_force, contact_mask=None) -> ShearOutput:
        delta = np.asarray(tangent_delta, dtype=np.float32)
        normal = np.asarray(normal_force, dtype=np.float32)

        if self._shear is None or self._shear.shape != delta.shape:
            self._shear = np.zeros_like(delta, dtype=np.float32)

        shear = self._shear + self.cfg.tangential_stiffness * delta
        limit = self.cfg.friction_coefficient * np.maximum(normal, 0.0)
        while limit.ndim < shear.ndim:
            limit = limit[..., None]

        norm = np.linalg.norm(shear, axis=-1, keepdims=True)
        scale = np.minimum(1.0, limit / (norm + 1.0e-12))
        shear = shear * scale

        if contact_mask is not None:
            mask = np.asarray(contact_mask, dtype=bool)
            while mask.ndim < shear.ndim:
                mask = mask[..., None]
            shear = np.where(mask, shear, 0.0)

        self._shear = shear.astype(np.float32, copy=False)
        return ShearOutput(shear_force=self._shear, state=self._shear)
