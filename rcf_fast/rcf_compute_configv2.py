# rcf_fast/rcf_compute_configv2.py
# ============================================================
# 功能：RCF v2 计算配置（回退版：仅改 3ms->10ms + 指数衰减）
# 日期：2026-01-07
# 说明：
# - score1 使用 TS + 指数时间核：w(dt)=exp(-dt/tau)，并在 dt<=T_us 内累加
# - 默认 T_us=10ms，tau_us=3ms（更强调近邻一致性，ESR 更可能受益）
# - 其余配置保持 v1 逻辑与接口风格一致
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class RCFComputeConfigv2:
    # -----------------------------
    # Sensor / slicing
    # -----------------------------
    resolution: Tuple[int, int] = (346, 260)   # (W, H)
    bin_us: int = 10_000                       # 10 ms（bin 不改）

    # -----------------------------
    # Score1 (TS-based support)
    # -----------------------------
    radius: int = 2        # 5x5 neighborhood
    T_us: int = 10_000     # 10 ms（由 3ms 改为 10ms）
    tau_us: int = 3_000    # 指数衰减时间常数 τ（默认 3ms，更可能利于 ESR）
    K_sat: float = 3.0     # saturation constant

    # -----------------------------
    # Block / score2
    # -----------------------------
    block_size: int = 16
    min_events_per_block: int = 20
    anchor_ratio: float = 0.10     # top / bottom 10%
    score2_sparse_value: float = 0.3
    eps: float = 1e-12

    # -----------------------------
    # Fusion / threshold
    # -----------------------------
    eta_list: List[float] = None   # e.g. [0.05, ..., 0.30] 或你自己在 step4 扫描

    def __post_init__(self):
        if self.eta_list is None:
            object.__setattr__(
                self,
                "eta_list",
                [round(v, 2) for v in np.linspace(0.25, 0.50, 6)]
            )

        # 基本健壮性检查（轻量）
        if self.T_us <= 0:
            raise ValueError("T_us must be > 0")
        if self.tau_us <= 0:
            raise ValueError("tau_us must be > 0")
        if self.K_sat <= 0:
            raise ValueError("K_sat must be > 0")
