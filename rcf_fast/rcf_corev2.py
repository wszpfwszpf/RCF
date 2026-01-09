# rcf_fast/rcf_corev2.py
# ============================================================
# 日期：2026-01-08
#
# 核心：
#   score1：局部时空一致性（Time-Surface + 线性衰减核）
#   score2：跨块相对统计对比（默认 uonly：只用 ub；RBF 原型相似度）
#
# 说明：
# - 不改 step4 调用方式：rcf_process_bin(events, state, cfg)
# - cfg 来自 rcf_compute_configv2.RCFComputeConfigv2
# ============================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_configv2 import RCFComputeConfigv2


# -------------------------------------------------
# 输出结果结构体：便于上层 step4 读取/保存
# -------------------------------------------------
@dataclass
class RCFResult:
    score1: np.ndarray
    score2: np.ndarray
    score: np.ndarray
    keep_masks: Dict[float, np.ndarray]
    block_id: np.ndarray
    ub: np.ndarray
    sigmab: np.ndarray


# -------------------------------------------------
# 工具：从 dv EventStore 中抽取 t/x/y/p
# -------------------------------------------------
def _extract_txyp(events) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = events.numpy()
    names = arr.dtype.names

    x = arr["x"].astype(np.int32, copy=False)
    y = arr["y"].astype(np.int32, copy=False)

    if "t" in names:
        t = arr["t"].astype(np.int64, copy=False)
    elif "timestamp" in names:
        t = arr["timestamp"].astype(np.int64, copy=False)
    else:
        raise KeyError(f"EventStore has no time field. Available fields: {names}")

    if "p" in names:
        p = arr["p"].astype(np.int8, copy=False)
    elif "polarity" in names:
        p = arr["polarity"].astype(np.int8, copy=False)
    else:
        p = np.zeros_like(x, dtype=np.int8)

    return t, x, y, p


def _safe_event_count(events) -> int:
    sz = getattr(events, "size", None)
    try:
        return int(sz() if callable(sz) else sz)
    except Exception:
        return 0


# -------------------------------------------------
# score1：Time-Surface + 线性衰减核
# -------------------------------------------------
try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap


@njit(cache=True, fastmath=True)
def _score1_ts_numba(t, x, y, ts_map, radius, T_us, K_sat):
    H, W = ts_map.shape
    out = np.zeros(t.shape[0], dtype=np.float32)

    T = float(T_us)
    K = float(K_sat)

    for i in range(t.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        ti = float(t[i])

        s = 0.0

        for dy in range(-radius, radius + 1):
            yy = yi + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-radius, radius + 1):
                xx = xi + dx
                if xx < 0 or xx >= W:
                    continue

                last = ts_map[yy, xx]
                dt = ti - last

                if dt > 0.0 and dt <= T:
                    s += (1.0 - dt / T)

        s = s / K
        if s > 1.0:
            s = 1.0
        out[i] = np.float32(s)

        ts_map[yi, xi] = ti

    return out


def compute_score1_ts(t, x, y, state: RCFState, cfg: RCFComputeConfigv2) -> np.ndarray:
    ts_map = state.ts.last_ts

    if _NUMBA_OK:
        return _score1_ts_numba(t, x, y, ts_map, cfg.radius, cfg.T_us, cfg.K_sat)

    # fallback：纯 python（debug 用）
    H, W = ts_map.shape
    r = int(cfg.radius)
    T = float(cfg.T_us)
    Ksat = float(cfg.K_sat)

    score1 = np.zeros_like(t, dtype=np.float32)

    for i in range(len(t)):
        xi, yi, ti = int(x[i]), int(y[i]), float(t[i])

        s = 0.0
        for dy in range(-r, r + 1):
            yy = yi + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-r, r + 1):
                xx = xi + dx
                if xx < 0 or xx >= W:
                    continue
                dt = ti - ts_map[yy, xx]
                if 0.0 < dt <= T:
                    s += (1.0 - dt / T)

        s = s / Ksat
        if s > 1.0:
            s = 1.0

        score1[i] = np.float32(s)
        ts_map[yi, xi] = ti

    return score1


# -------------------------------------------------
# 每个 block 的统计量：counts / ub / sigmab
# -------------------------------------------------
def compute_block_stats(score1: np.ndarray, block_id: np.ndarray, n_blocks: int):
    counts = np.bincount(block_id, minlength=n_blocks).astype(np.int32)

    sum1 = np.bincount(block_id, weights=score1, minlength=n_blocks)
    ub = np.zeros(n_blocks, dtype=np.float64)
    valid = counts > 0
    ub[valid] = sum1[valid] / counts[valid]

    sum2 = np.bincount(block_id, weights=score1 ** 2, minlength=n_blocks)
    sigmab = np.zeros(n_blocks, dtype=np.float64)
    sigmab[valid] = np.sqrt(
        np.maximum(sum2[valid] / counts[valid] - ub[valid] ** 2, 0.0)
    )

    return counts, ub, sigmab


def _cosine_vec(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


# -------------------------------------------------
# score2：跨块相对统计对比（默认 uonly）
#
# uonly（默认）：
#   - block 特征只用 ub
#   - 仍使用 top/bottom anchors 构造 signal/noise 原型（标量）
#   - 相似度用 RBF（避免 1D cosine 退化）
#   - score2_b = c_sig / (c_sig + c_noise + eps)
#
# 兼容对照：
#   - cfg.score2_use_sigma=True 时，用 [ub,sigmab] + cosine（2D 原始）
# -------------------------------------------------
def compute_score2(
    score1: np.ndarray,
    block_id: np.ndarray,
    counts: np.ndarray,
    ub: np.ndarray,
    sigmab: np.ndarray,
    cfg: RCFComputeConfigv2,
) -> np.ndarray:

    n_blocks = ub.shape[0]
    score2_block = np.full(n_blocks, cfg.score2_sparse_value, dtype=np.float64)

    # 只对事件数足够的 block 计算 score2
    valid = counts >= cfg.min_events_per_block
    idx = np.where(valid)[0]
    if idx.size < 2:
        return score2_block[block_id].astype(np.float32)

    # 按 ub 排序，构造上下锚点集合
    order = idx[np.argsort(ub[idx])]
    k = max(1, int(cfg.anchor_ratio * order.size))
    bot = order[:k]
    top = order[-k:]

    # 权重：弱化超大块支配性
    w = np.log1p(counts).astype(np.float64)

    if bool(getattr(cfg, "score2_use_sigma", False)):
        # -----------------------------
        # 原始 2D cosine 版本（对照用）
        # -----------------------------
        phi = np.stack([ub, sigmab], axis=1)  # (n_blocks, 2)

        phi_sig = np.average(phi[top], axis=0, weights=w[top])
        phi_noise = np.average(phi[bot], axis=0, weights=w[bot])

        for b in idx:
            c_sig = max(_cosine_vec(phi[b], phi_sig, cfg.eps), 0.0)
            c_noise = max(_cosine_vec(phi[b], phi_noise, cfg.eps), 0.0)
            score2_block[b] = c_sig / (c_sig + c_noise + cfg.eps)

        return score2_block[block_id].astype(np.float32)

    # -----------------------------
    # 默认：uonly + RBF
    # -----------------------------
    u_sig = float(np.average(ub[top], weights=w[top]))
    u_noise = float(np.average(ub[bot], weights=w[bot]))

    # tau：按 valid blocks 的 ub 分布自适应（避免过硬/过软）
    ub_valid = ub[idx]
    std = float(np.std(ub_valid))
    tau = max(float(getattr(cfg, "score2_tau_min", 1e-3)),
              float(getattr(cfg, "score2_tau_scale", 1.0)) * std)

    inv_2tau2 = 1.0 / (2.0 * tau * tau + cfg.eps)

    for b in idx:
        du_sig = float(ub[b] - u_sig)
        du_noi = float(ub[b] - u_noise)

        # RBF similarity in 1D (always positive, continuous)
        c_sig = np.exp(-(du_sig * du_sig) * inv_2tau2)
        c_noise = np.exp(-(du_noi * du_noi) * inv_2tau2)

        score2_block[b] = c_sig / (c_sig + c_noise + cfg.eps)

    return score2_block[block_id].astype(np.float32)


# -------------------------------------------------
# 主入口：处理一个 bin 内的事件
# -------------------------------------------------
def rcf_process_bin(
    events,
    state: RCFState,
    cfg: RCFComputeConfigv2,
) -> Optional[RCFResult]:

    N = _safe_event_count(events)
    if N <= 0:
        return None

    t, x, y, p = _extract_txyp(events)

    # 1) score1
    score1 = compute_score1_ts(t, x, y, state, cfg)

    # 2) block id
    W, H = cfg.resolution
    B = int(cfg.block_size)
    nbx = (W + B - 1) // B
    nby = (H + B - 1) // B

    bx = x // B
    by = y // B
    block_id = bx + by * nbx
    n_blocks = nbx * nby

    # 3) block stats
    counts, ub, sigmab = compute_block_stats(score1, block_id, n_blocks)

    # 4) score2 (default uonly)
    score2 = compute_score2(score1, block_id, counts, ub, sigmab, cfg)

    # 5) fusion
    score = (score1 * score2).astype(np.float32)

    # 6) keep masks
    keep_masks = {eta: (score >= eta) for eta in cfg.eta_list}

    return RCFResult(
        score1=score1,
        score2=score2,
        score=score,
        keep_masks=keep_masks,
        block_id=block_id,
        ub=ub,
        sigmab=sigmab,
    )
