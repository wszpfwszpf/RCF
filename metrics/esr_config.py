# metrics/esr_config.py
# -*- coding: utf-8 -*-
"""
ESR/MESR evaluation configuration.

This module centralizes evaluation protocol constants so that:
- tools scripts stay simple
- results are reproducible
- protocol is explicit (N packet size, M reference size, resolution)

Current phase:
- Evaluate ESR on NPZ-based event streams (txyp), where time unit can be us.
- Future: can be reused for AEDAT4 streaming without changing ESR core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ESRConfig:
    # Sensor resolution (W, H)
    resolution: Tuple[int, int] = (346, 260)

    # Packet size for MESR: fixed number of events per packet
    n_events_packet: int = 30_000

    # Reference event count used in LN interpolation (fixed across all methods)
    m_events_ref: int = 20_000

    # Drop the last packet if it contains fewer than n_events_packet events
    drop_last: bool = True

    # Optional: ignore polarity in ESR computation (ESR typically uses spatial counts)
    ignore_polarity: bool = True

    # Optional: if you later implement hot-pixel handling, store a boolean mask here.
    # Shape should be (H, W), True indicates "valid pixel"; False indicates "masked out".
    hot_pixel_valid_mask: Optional[object] = None  # keep generic to avoid hard numpy dependency here

    # Safety: if True, validate x/y in [0, W) x [0, H) and drop invalid events.
    validate_xy: bool = True


# A convenient default config (E-MLB-style protocol)
DEFAULT_ESR_CONFIG = ESRConfig()
