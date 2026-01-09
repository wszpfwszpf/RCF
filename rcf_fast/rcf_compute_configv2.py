# rcf_fast/rcf_compute_configv2.py
# ============================================================
# 功能：RCF v2 计算配置（score2 默认 uonly 版本）
# 日期：2026-01-08
#
# 说明：
# - 本文件只提供参数，不改 step4 的调用方式
# - score2 默认使用 uonly（只用 ub），并使用 RBF 相似度做“原型对比”
# - score1 核函数由 core 决定（你当前 core 仍是线性 TS）
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
    bin_us: int = 10_000                       # bin size (step4 会覆盖)

    # -----------------------------
    # Score1 (TS-based support)
    # -----------------------------
    radius: int = 2        # 5x5 neighborhood
    T_us: int = 10_000     # 10 ms
    tau_us: int = 3_000    # 兼容保留（当前 core 的线性 TS 不使用它）
    K_sat: float = 3.0     # saturation constant

    # -----------------------------
    # Block / score2
    # -----------------------------
    block_size: int = 16
    min_events_per_block: int = 20
    anchor_ratio: float = 0.10     # top/bottom 10%
    score2_sparse_value: float = 0.5
    eps: float = 1e-12

    # score2 feature switch
    # - False: uonly（默认）
    # - True : 使用 [ub, sigmab] 的原始 2D cosine 版本（保留做对照）
    score2_use_sigma: bool = False

    # uonly 相似度：RBF 带宽（tau）
    # tau = score2_tau_scale * std(ub_valid)，并做最小值保护
    score2_tau_scale: float = 1.0
    score2_tau_min: float = 1e-3  # 防止 std 极小导致过硬

    # -----------------------------
    # Fusion / threshold
    # -----------------------------
    eta_list: List[float] = None   # step4 侧会直接用 cfg.eta_list

    def __post_init__(self):
        if self.eta_list is None:
            object.__setattr__(
                self,
                "eta_list",
                [round(v, 2) for v in np.linspace(0.2, 0.7, 6)]
            )

        # 基本健壮性检查（轻量）
        if self.T_us <= 0:
            raise ValueError("T_us must be > 0")
        if self.K_sat <= 0:
            raise ValueError("K_sat must be > 0")
        if self.anchor_ratio <= 0.0 or self.anchor_ratio >= 0.5:
            raise ValueError("anchor_ratio should be in (0, 0.5) for top/bottom selection")
        if self.score2_tau_scale <= 0:
            raise ValueError("score2_tau_scale must be > 0")
        if self.score2_tau_min <= 0:
            raise ValueError("score2_tau_min must be > 0")
