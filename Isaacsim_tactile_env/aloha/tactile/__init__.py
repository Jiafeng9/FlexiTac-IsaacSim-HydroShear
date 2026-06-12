from .backend import (
    HydroShearTactileBackend,
    HydroShearTactileBackendCfg,
    TaxelShearTactileBackend,
    TaxelShearTactileBackendCfg,
    WarpSdfTactileBackend,
    WarpSdfTactileBackendCfg,
)
from .cfg import AlohaTactileCfg
from .runtime import AlohaTactileOutput, TrackInfo


def __getattr__(name: str):
    if name == "AlohaTactileSetup":
        from .runtime import AlohaTactileSetup

        return AlohaTactileSetup
    raise AttributeError(name)


__all__ = [
    "AlohaTactileCfg",
    "AlohaTactileOutput",
    "AlohaTactileSetup",
    "TrackInfo",
    "WarpSdfTactileBackendCfg",
    "WarpSdfTactileBackend",
    "HydroShearTactileBackend",
    "HydroShearTactileBackendCfg",
    "TaxelShearTactileBackend",
    "TaxelShearTactileBackendCfg",
]
