# rcf_fast/rcf_core.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_config import RCFComputeConfig


# -------------------------------------------------
# Result container
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
# Utilities
# -------------------------------------------------
def _extract_txyp(events) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = events.numpy()
    names = arr.dtype.names

    # x, y 一定有
    x = arr["x"].astype(np.int32, copy=False)
    y = arr["y"].astype(np.int32, copy=False)

    # time field: t or timestamp
    if "t" in names:
        t = arr["t"].astype(np.int64, copy=False)
    elif "timestamp" in names:
        t = arr["timestamp"].astype(np.int64, copy=False)
    else:
        raise KeyError(f"EventStore has no time field. Available fields: {names}")

    # polarity: p or polarity (optional)
    if "p" in names:
        p = arr["p"].astype(np.int8, copy=False)
    elif "polarity" in names:
        p = arr["polarity"].astype(np.int8, copy=False)
    else:
        # polarity not required by RCF, fill zeros
        p = np.zeros_like(x, dtype=np.int8)

    return t, x, y, p



def _safe_event_count(events) -> int:
    sz = getattr(events, "size", None)
    try:
        return int(sz() if callable(sz) else sz)
    except Exception:
        return 0


# -------------------------------------------------
# Score1: TS + linear decay (npz-consistent)
# -------------------------------------------------
try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
    def njit(*args, **kwargs):
        def wrap(fn): return fn
        return wrap


@njit(cache=True, fastmath=True)
def _score1_ts_numba(t, x, y, ts_map, radius, T_us, K_sat):
    H, W = ts_map.shape
    out = np.zeros(t.shape[0], dtype=np.float32)
    T = float(T_us)
    K = float(K_sat)

    for i in range(t.shape[0]):
        xi = int(x[i]); yi = int(y[i]); ti = float(t[i])
        s = 0.0

        # 5x5 neighborhood
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


def compute_score1_ts(t, x, y, state, cfg):
    # ts_map is float64; numba supports it fine
    ts_map = state.ts.last_ts
    if _NUMBA_OK:
        return _score1_ts_numba(t, x, y, ts_map, cfg.radius, cfg.T_us, cfg.K_sat)
    # fallback: original python loop (slow)
    H, W = ts_map.shape
    r = cfg.radius
    T = float(cfg.T_us)
    Ksat = float(cfg.K_sat)
    score1 = np.zeros_like(t, dtype=np.float32)
    for i in range(len(t)):
        xi, yi, ti = int(x[i]), int(y[i]), float(t[i])
        s = 0.0
        for dy in range(-r, r + 1):
            yy = yi + dy
            if yy < 0 or yy >= H: continue
            for dx in range(-r, r + 1):
                xx = xi + dx
                if xx < 0 or xx >= W: continue
                dt = ti - ts_map[yy, xx]
                if 0.0 < dt <= T:
                    s += (1.0 - dt / T)
        s = min(1.0, s / Ksat)
        score1[i] = s
        ts_map[yi, xi] = ti
    return score1

# def compute_score1_ts(
#     t: np.ndarray,
#     x: np.ndarray,
#     y: np.ndarray,
#     state: RCFState,
#     cfg: RCFComputeConfig,
# ) -> np.ndarray:
#     ts_map = state.ts.last_ts
#     H, W = ts_map.shape
#     r = cfg.radius
#     T = float(cfg.T_us)
#     Ksat = float(cfg.K_sat)
#
#     score1 = np.zeros_like(t, dtype=np.float32)
#
#     for i in range(len(t)):
#         xi, yi, ti = int(x[i]), int(y[i]), float(t[i])
#         s = 0.0
#
#         for dy in range(-r, r + 1):
#             yy = yi + dy
#             if yy < 0 or yy >= H:
#                 continue
#             for dx in range(-r, r + 1):
#                 xx = xi + dx
#                 if xx < 0 or xx >= W:
#                     continue
#                 dt = ti - ts_map[yy, xx]
#                 if 0.0 < dt <= T:
#                     s += (1.0 - dt / T)
#
#         s = min(1.0, s / Ksat)
#         score1[i] = s
#         ts_map[yi, xi] = ti
#
#     return score1


# -------------------------------------------------
# Block statistics
# -------------------------------------------------
def compute_block_stats(score1, block_id, n_blocks):
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


def cosine(a, b, eps):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


# -------------------------------------------------
# Score2: prototype consistency (npz-consistent)
# -------------------------------------------------
def compute_score2(
    score1,
    block_id,
    counts,
    ub,
    sigmab,
    cfg: RCFComputeConfig,
):
    n_blocks = ub.shape[0]
    score2_block = np.full(n_blocks, cfg.score2_sparse_value, dtype=np.float64)

    valid = counts >= cfg.min_events_per_block
    idx = np.where(valid)[0]
    if idx.size < 2:
        return score2_block[block_id].astype(np.float32)

    # sort by ub
    order = idx[np.argsort(ub[idx])]
    k = max(1, int(cfg.anchor_ratio * order.size))
    bot = order[:k]
    top = order[-k:]

    phi = np.stack([ub, sigmab], axis=1)
    w = np.log1p(counts)

    phi_sig = np.average(phi[top], axis=0, weights=w[top])
    phi_noise = np.average(phi[bot], axis=0, weights=w[bot])

    for b in idx:
        c_sig = max(cosine(phi[b], phi_sig, cfg.eps), 0.0)
        c_noise = max(cosine(phi[b], phi_noise, cfg.eps), 0.0)
        score2_block[b] = c_sig / (c_sig + c_noise + cfg.eps)

    return score2_block[block_id].astype(np.float32)


# -------------------------------------------------
# Main entry
# -------------------------------------------------
def rcf_process_bin(
    events,
    state: RCFState,
    cfg: RCFComputeConfig,
) -> Optional[RCFResult]:

    N = _safe_event_count(events)
    if N <= 0:
        return None

    t, x, y, p = _extract_txyp(events)

    # score1
    score1 = compute_score1_ts(t, x, y, state, cfg)

    # block id
    W, H = cfg.resolution
    B = cfg.block_size
    nbx = (W + B - 1) // B
    bx = x // B
    by = y // B
    block_id = bx + by * nbx
    n_blocks = nbx * ((H + B - 1) // B)

    counts, ub, sigmab = compute_block_stats(score1, block_id, n_blocks)

    # score2
    score2 = compute_score2(score1, block_id, counts, ub, sigmab, cfg)

    # fusion
    score = (score1 * score2).astype(np.float32)

    # keep masks
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
