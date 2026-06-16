"""Generic tactile-model utilities shared across robot-specific adapters.

The implementation is grouped into three modules:

- :mod:`tactile.geometry` for mesh sampling, SDFs, transforms, and contact queries.
- :mod:`tactile.readout` for taxel grids, projections, and taxel-level shear.
- :mod:`tactile.hydroshear` for the recurrent HydroShear model and backend pipeline.

Legacy submodule names are kept as import aliases for existing scripts.
"""

from __future__ import annotations

import sys as _sys

from . import geometry as _geometry
from . import hydroshear as _hydroshear
from . import readout as _readout
from .geometry import (
    ElastomerSdfResult,
    MeshPatchElastomerSdf,
    ObjectMeshSdfResult,
    ObjectSurfaceSampler,
    ObjectSurfaceSamplerCfg,
    ObjectSurfaceSamples,
    PytorchVolumetricMeshSdf,
    SurfacePointContactQuery,
    SurfacePointContactState,
    first_visual_mesh_from_urdf,
    inverse_transform_points,
    load_trimesh,
    normalize_quat_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    rotate_vectors,
    transform_points,
)
from .hydroshear import (
    BumpHydroShearCfg,
    BumpHydroShearOutput,
    BumpHydroShearState,
    BumpHydroShearTracker,
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
    HydroShearTactileBackendOutput,
    SurfacePointHydroShearCfg,
    SurfacePointHydroShearOutput,
    SurfacePointHydroShearTracker,
    create_bump_grid_centers,
    contact_segment_fraction,
)
from .readout import (
    HydroShearMarkerProjector,
    HydroShearMarkerReadout,
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointOutput,
    ProjectedSurfacePointState,
    ProjectedSurfacePointTracker,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjector,
    SurfacePointForceProjectorCfg,
    TaxelForceReadout,
    TaxelGridCfg,
    TaxelShearOutput,
    TaxelShearState,
    TaxelShearTracker,
    TaxelShearTrackerCfg,
    compute_per_taxel_delta_tangent,
    create_taxel_grid_points,
    tangential_axes,
    update_taxel_shear_force,
)

_LEGACY_SUBMODULES = {
    "backend": _hydroshear,
    "contact": _geometry,
    "elastomer": _geometry,
    "surface": _geometry,
    "readout": _readout,
    "taxel_shear": _readout,
}
for _name, _module in _LEGACY_SUBMODULES.items():
    _sys.modules[f"{__name__}.{_name}"] = _module
    if __name__ != "tactile":
        _sys.modules.setdefault(f"tactile.{_name}", _module)

__all__ = [
    "ElastomerSdfResult",
    "BumpHydroShearCfg",
    "BumpHydroShearOutput",
    "BumpHydroShearState",
    "BumpHydroShearTracker",
    "HydroShearMarkerProjector",
    "HydroShearMarkerReadout",
    "HydroShearMarkerReadoutCfg",
    "HydroShearTactileBackend",
    "HydroShearTactileBackendCfg",
    "HydroShearTactileBackendOutput",
    "MeshPatchElastomerSdf",
    "ObjectMeshSdfResult",
    "ObjectSurfaceSampler",
    "ObjectSurfaceSamplerCfg",
    "ObjectSurfaceSamples",
    "PytorchVolumetricMeshSdf",
    "ProjectedSurfacePointOutput",
    "ProjectedSurfacePointState",
    "ProjectedSurfacePointTracker",
    "ProjectedSurfacePointTrackerCfg",
    "SurfacePointContactQuery",
    "SurfacePointContactState",
    "SurfacePointForceProjector",
    "SurfacePointForceProjectorCfg",
    "SurfacePointHydroShearCfg",
    "SurfacePointHydroShearOutput",
    "SurfacePointHydroShearTracker",
    "TaxelForceReadout",
    "TaxelGridCfg",
    "TaxelShearOutput",
    "TaxelShearState",
    "TaxelShearTracker",
    "TaxelShearTrackerCfg",
    "compute_per_taxel_delta_tangent",
    "contact_segment_fraction",
    "create_bump_grid_centers",
    "create_taxel_grid_points",
    "first_visual_mesh_from_urdf",
    "inverse_transform_points",
    "load_trimesh",
    "normalize_quat_wxyz",
    "quat_apply_wxyz",
    "quat_conjugate_wxyz",
    "rotate_vectors",
    "tangential_axes",
    "transform_points",
    "update_taxel_shear_force",
]
