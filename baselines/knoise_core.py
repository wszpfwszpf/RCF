# baselines/knoise_core.py
# 日期：2026-01-08
# 中文说明：
# KNoise（O(N)-space spatiotemporal filter）baseline 核心实现（软件等价版）。
# 核心思想：不维护每个像素的时间戳表（O(W*H)），而是维护“行/列级”最近事件记忆（O(W+H)）。
# 对于新事件 e=(t,x,y,p)，在其所在行 y 与所在列 x 的 FIFO 记忆中查找支持：
# - 时间支持：t - t_hist <= support_us
# - 空间支持：行记忆中要求 |x - x_hist|<=1；列记忆中要求 |y - y_hist|<=1
# - 可选极性一致：use_polarity=True 时，要求同一极性通道匹配
# 满足任一支持则保留，否则丢弃。
#
# 输出：keepmask (bool, N,) + stats
# 依赖：numba（默认强依赖以保证速度；如需纯 numpy 版可再写一个慢速 fallback）

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import numba as nb
    _NUMBA_OK = True
except Exception:
    nb = None
    _NUMBA_OK = False


# -----------------------------
# Config / Result（RCF-like）
# -----------------------------
@dataclass
class KNoiseComputeConfig:
    resolution: Tuple[int, int]          # (W, H)
    bin_us: int                          # for pipeline consistency/logging
    support_us: int = 3000               # ΔT: support time window
    use_polarity: bool = True            # 是否按极性分通道
    fifo_k: int = 2                      # 行/列 FIFO 深度（论文“两块 memory”可用 K=2 对应）
    clamp_xy: bool = False               # 可选鲁棒性
    init_ts: int = -10**18               # 初始时间戳（表示无效）


@dataclass
class KNoiseResult:
    keepmask: np.ndarray                 # bool (N,)
    stats: Dict[str, Any]


# -----------------------------
# State：行/列 FIFO 记忆（O(W+H)）
# -----------------------------
class KNoiseState:
    def __init__(self, resolution: Tuple[int, int], cfg: KNoiseComputeConfig):
        W, H = int(resolution[0]), int(resolution[1])
        C = 2 if cfg.use_polarity else 1
        K = int(cfg.fifo_k)

        if K < 1:
            raise ValueError("fifo_k must be >= 1")

        self.resolution = (W, H)
        self.use_polarity = bool(cfg.use_polarity)
        self.init_ts = int(cfg.init_ts)
        self.fifo_k = K

        # 行记忆：row_ts[y, k, c] & row_x[y, k, c]
        self.row_ts = np.full((H, K, C), self.init_ts, dtype=np.int64)
        self.row_x = np.full((H, K, C), -1, dtype=np.int32)

        # 列记忆：col_ts[x, k, c] & col_y[x, k, c]
        self.col_ts = np.full((W, K, C), self.init_ts, dtype=np.int64)
        self.col_y = np.full((W, K, C), -1, dtype=np.int32)

    @staticmethod
    def create(resolution: Tuple[int, int], cfg: KNoiseComputeConfig) -> "KNoiseState":
        return KNoiseState(resolution, cfg)

    def reset(self):
        self.row_ts.fill(self.init_ts)
        self.row_x.fill(-1)
        self.col_ts.fill(self.init_ts)
        self.col_y.fill(-1)


# -----------------------------
# Extraction utility（沿用 BAF/DWF 风格）
# -----------------------------
def _extract_txyp(events):
    """
    从 dv.EventStore-like 或 numpy structured array/dict 中提取 (t, x, y, p)。
    """
    if hasattr(events, "numpy"):
        arr = events.numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            x = arr["x"].astype(np.int32, copy=False)
            y = arr["y"].astype(np.int32, copy=False)
            if "t" in arr.dtype.fields:
                t = arr["t"].astype(np.int64, copy=False)
            else:
                t = arr["timestamp"].astype(np.int64, copy=False)
            if "p" in arr.dtype.fields:
                p = arr["p"].astype(np.int8, copy=False)
            elif "polarity" in arr.dtype.fields:
                p = arr["polarity"].astype(np.int8, copy=False)
            else:
                p = np.zeros_like(x, dtype=np.int8)
            return t, x, y, p
        events = arr

    if isinstance(events, dict):
        x = np.asarray(events["x"], dtype=np.int32)
        y = np.asarray(events["y"], dtype=np.int32)
        t = np.asarray(events.get("t", events.get("timestamp", np.zeros_like(x))), dtype=np.int64)
        p = np.asarray(events.get("p", events.get("polarity", np.zeros_like(x))), dtype=np.int8)
        return t, x, y, p

    if isinstance(events, np.ndarray) and events.dtype.fields is not None:
        x = events["x"].astype(np.int32, copy=False)
        y = events["y"].astype(np.int32, copy=False)
        if "t" in events.dtype.fields:
            t = events["t"].astype(np.int64, copy=False)
        else:
            t = events["timestamp"].astype(np.int64, copy=False)
        if "p" in events.dtype.fields:
            p = events["p"].astype(np.int8, copy=False)
        elif "polarity" in events.dtype.fields:
            p = events["polarity"].astype(np.int8, copy=False)
        else:
            p = np.zeros_like(x, dtype=np.int8)
        return t, x, y, p

    raise TypeError("Unsupported events container type for _extract_txyp().")


def _pol_to_c(p: np.ndarray) -> np.ndarray:
    """
    polarity -> {0,1}，支持 p in {0,1} 或 {-1,+1}
    """
    if p.size == 0:
        return p.astype(np.int32)
    if int(p.min()) < 0:
        return (p > 0).astype(np.int32)
    return p.astype(np.int32)


# -----------------------------
# Numba kernel（行/列 FIFO 支持搜索 + 更新）
# -----------------------------
if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False)
    def _knoise_kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        row_ts: np.ndarray,  # (H,K,C)
        row_x: np.ndarray,   # (H,K,C)
        col_ts: np.ndarray,  # (W,K,C)
        col_y: np.ndarray,   # (W,K,C)
        W: int,
        H: int,
        C: int,
        K: int,
        T_us: int,
    ):
        n = x.shape[0]
        keep_u8 = np.zeros(n, dtype=np.uint8)
        n_keep = 0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t[i])
            ci = int(c[i]) if C == 2 else 0

            # ---------
            # Step A：在“行/列 FIFO”里查支持（先查后写，避免自支持）
            # ---------
            supported = False

            # A1) 行 y：找历史事件 (t_hist, x_hist)
            # 条件：ti - t_hist <= T_us 且 |xi - x_hist| <= 1
            for k in range(K):
                th = int(row_ts[yi, k, ci])
                if th == -10**18:
                    continue
                dt = ti - th
                if dt <= T_us:
                    xh = int(row_x[yi, k, ci])
                    if xh >= 0:
                        dx = xi - xh
                        if dx < 0:
                            dx = -dx
                        if dx <= 1:
                            supported = True
                            break

            # A2) 列 x：找历史事件 (t_hist, y_hist)
            if not supported:
                for k in range(K):
                    th = int(col_ts[xi, k, ci])
                    if th == -10**18:
                        continue
                    dt = ti - th
                    if dt <= T_us:
                        yh = int(col_y[xi, k, ci])
                        if yh >= 0:
                            dy = yi - yh
                            if dy < 0:
                                dy = -dy
                            if dy <= 1:
                                supported = True
                                break

            if supported:
                keep_u8[i] = 1
                n_keep += 1

            # ---------
            # Step B：更新 FIFO（无论 keep/drop，都写入记忆，符合“可恢复/可唤醒”的工程逻辑）
            # FIFO 规则：最新放在 k=0，其他右移，丢弃最旧
            # ---------
            # row FIFO
            for k in range(K - 1, 0, -1):
                row_ts[yi, k, ci] = row_ts[yi, k - 1, ci]
                row_x[yi, k, ci] = row_x[yi, k - 1, ci]
            row_ts[yi, 0, ci] = ti
            row_x[yi, 0, ci] = xi

            # col FIFO
            for k in range(K - 1, 0, -1):
                col_ts[xi, k, ci] = col_ts[xi, k - 1, ci]
                col_y[xi, k, ci] = col_y[xi, k - 1, ci]
            col_ts[xi, 0, ci] = ti
            col_y[xi, 0, ci] = yi

        return keep_u8, n_keep


# -----------------------------
# Core：KNoise per-bin processing
# -----------------------------
def knoise_process_bin(events, state: KNoiseState, cfg: KNoiseComputeConfig) -> Optional[KNoiseResult]:
    t, x, y, p = _extract_txyp(events)
    n = int(x.shape[0])
    if n == 0:
        return None

    if not _NUMBA_OK:
        raise RuntimeError(
            "Numba is not available, but this KNoise implementation uses numba for speed. "
            "Install numba or ask for a pure-numpy fallback (slower)."
        )

    # optional clamp
    if cfg.clamp_xy:
        W, H = cfg.resolution
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

    W, H = cfg.resolution
    C = 2 if cfg.use_polarity else 1
    K = int(cfg.fifo_k)
    T_us = int(cfg.support_us)

    if cfg.use_polarity:
        c = _pol_to_c(p)
    else:
        c = np.zeros_like(x, dtype=np.int32)

    keep_u8, n_keep = _knoise_kernel(
        t.astype(np.int64, copy=False),
        x.astype(np.int32, copy=False),
        y.astype(np.int32, copy=False),
        c.astype(np.int32, copy=False),
        state.row_ts,
        state.row_x,
        state.col_ts,
        state.col_y,
        int(W), int(H), int(C),
        int(K),
        int(T_us),
    )

    keep = keep_u8.view(np.bool_)  # 0/1 -> bool（zero-copy view）

    stats = {
        "n_in": n,
        "n_keep": int(n_keep),
        "n_drop": int(n - n_keep),
        "keep_rate": float(n_keep) / float(n) if n > 0 else 0.0,
        "support_us": int(cfg.support_us),
        "use_polarity": bool(cfg.use_polarity),
        "fifo_k": int(cfg.fifo_k),
        "numba": True,
    }
    return KNoiseResult(keepmask=keep, stats=stats)
