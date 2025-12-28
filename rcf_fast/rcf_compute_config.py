# rcf_fast/rcf_compute_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class RCFComputeConfig:
    # -----------------------------
    # Sensor / slicing
    # -----------------------------
    resolution: Tuple[int, int] = (346, 260)   # (W, H)
    bin_us: int = 10_000                       # 10 ms

    # -----------------------------
    # Score1 (TS-based support)
    # -----------------------------
    radius: int = 2        # 5x5 neighborhood
    T_us: int = 3_000      # 3 ms
    K_sat: float = 3.0     # saturation constant

    # -----------------------------
    # Block / score2
    # -----------------------------
    block_size: int = 16
    min_events_per_block: int = 5
    anchor_ratio: float = 0.10     # top / bottom 10%
    score2_sparse_value: float = 0.5
    eps: float = 1e-12

    # -----------------------------
    # Fusion / threshold
    # -----------------------------
    eta_list: List[float] = None   # e.g. [0.05, ..., 0.30]

    def __post_init__(self):
        if self.eta_list is None:
            object.__setattr__(
                self,
                "eta_list",
                [round(v, 2) for v in np.linspace(0.05, 0.30, 6)]
            )
