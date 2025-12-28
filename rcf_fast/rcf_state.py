# rcf_fast/rcf_state.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class TimeSurface:
    last_ts: np.ndarray   # (H, W), float64

    @staticmethod
    def create(resolution: Tuple[int, int]) -> "TimeSurface":
        W, H = resolution
        ts = np.full((H, W), -np.inf, dtype=np.float64)
        return TimeSurface(ts)


@dataclass
class RCFState:
    ts: TimeSurface

    @staticmethod
    def create(resolution: Tuple[int, int]) -> "RCFState":
        return RCFState(TimeSurface.create(resolution))
