from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlohaTactileOutput:
    observations: dict[str, np.ndarray]
    selected_links: tuple[str, ...]
    target_query_paths: tuple[str, ...]
    sensor_slot_order: tuple[int, ...]
