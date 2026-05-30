"""Generic tactile-model utilities shared across robot-specific adapters."""

from .backend import HydroShearTactileBackend, HydroShearTactileBackendCfg, HydroShearTactileBackendOutput
from .contact import SurfacePointContactQuery, SurfacePointContactState
from .elastomer import FlatPatchElastomerSdf, FlatPatchElastomerSdfCfg, MeshPatchElastomerSdf
from .hydroshear import (
    SurfacePointHydroShearCfg,
    SurfacePointHydroShearOutput,
    SurfacePointHydroShearTracker,
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
    create_taxel_grid_points,
    tangential_axes,
)
from .surface import (
    ObjectMeshSdfResult,
    ObjectSurfaceSampler,
    ObjectSurfaceSamplerCfg,
    ObjectSurfaceSamples,
    signed_distance_to_mesh,
)
from .taxel_shear import (
    TaxelShearOutput,
    TaxelShearState,
    TaxelShearTracker,
    TaxelShearTrackerCfg,
    compute_per_taxel_delta_tangent,
    update_taxel_shear_force,
)

__all__ = [
    "HydroShearTactileBackend",
    "HydroShearTactileBackendCfg",
    "HydroShearTactileBackendOutput",
    "FlatPatchElastomerSdf",
    "FlatPatchElastomerSdfCfg",
    "MeshPatchElastomerSdf",
    "SurfacePointContactState",
    "SurfacePointContactQuery",
    "SurfacePointHydroShearCfg",
    "SurfacePointHydroShearOutput",
    "SurfacePointHydroShearTracker",
    "contact_segment_fraction",
    "ObjectSurfaceSamples",
    "ObjectMeshSdfResult",
    "ObjectSurfaceSamplerCfg",
    "ObjectSurfaceSampler",
    "signed_distance_to_mesh",
    "ProjectedSurfacePointOutput",
    "ProjectedSurfacePointState",
    "ProjectedSurfacePointTracker",
    "ProjectedSurfacePointTrackerCfg",
    "HydroShearMarkerProjector",
    "HydroShearMarkerReadout",
    "HydroShearMarkerReadoutCfg",
    "SurfacePointForceProjector",
    "SurfacePointForceProjectorCfg",
    "TaxelForceReadout",
    "TaxelGridCfg",
    "create_taxel_grid_points",
    "tangential_axes",
    "TaxelShearTrackerCfg",
    "TaxelShearState",
    "TaxelShearOutput",
    "TaxelShearTracker",
    "compute_per_taxel_delta_tangent",
    "update_taxel_shear_force",
]
