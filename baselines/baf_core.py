# baselines/baf_core.py
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
# Config / Result (RCF-like)
# -----------------------------
@dataclass
class BAFComputeConfig:
    resolution: Tuple[int, int]          # (W, H)
    bin_us: int                          # for pipeline consistency/logging
    support_us: int = 3000               # T in paper (support time window)
    use_polarity: bool = True            # paper uses 128x128x2 map
    clamp_xy: bool = False               # optional robustness
    init_ts: int = -10**18               # initial "no support"


@dataclass
class BAFResult:
    keepmask: np.ndarray                 # bool (N,)
    stats: Dict[str, Any]


# -----------------------------
# State: timestamp map
# -----------------------------
class BAFState:
    def __init__(self, resolution: Tuple[int, int], cfg: BAFComputeConfig):
        W, H = int(resolution[0]), int(resolution[1])
        C = 2 if cfg.use_polarity else 1
        self.resolution = (W, H)
        self.use_polarity = bool(cfg.use_polarity)
        self.init_ts = int(cfg.init_ts)

        # ts_map[x, y, c] stores the most recent support timestamp written by NEIGHBORS
        self.ts_map = np.full((W, H, C), self.init_ts, dtype=np.int64)

    @staticmethod
    def create(resolution: Tuple[int, int], cfg: BAFComputeConfig) -> "BAFState":
        return BAFState(resolution, cfg)

    def reset(self):
        self.ts_map.fill(self.init_ts)


# -----------------------------
# Extraction utility (copy DWF style)
# -----------------------------
def _extract_txyp(events):
    """
    Extract (t, x, y, p) from dv.EventStore-like or numpy structured array/dict.
    Input order in pipeline is (t, x, y, p) conceptually, but dv stores as fields.
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
    Map polarity to {0,1}.
    Supports p in {0,1} or {-1,+1}.
    """
    if p.size == 0:
        return p.astype(np.int32)
    pmin = int(p.min())
    if pmin < 0:
        return (p > 0).astype(np.int32)
    return p.astype(np.int32)


# -----------------------------
# Numba kernel (faithful to Delbruck BAF)
# -----------------------------
if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False)
    def _baf_kernel(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        ts_map: np.ndarray,   # (W,H,C) int64
        W: int,
        H: int,
        C: int,
        T_us: int,
    ):
        n = x.shape[0]
        keep_u8 = np.zeros(n, dtype=np.uint8)
        n_keep = 0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ci = int(c[i]) if C == 2 else 0
            ti = int(t[i])

            # Step 1 (paper): store event timestamp into ALL 8 NEIGHBORING pixels' timestamp memory
            # (avoids iterating neighbors on check; we do fixed 8 writes)
            # Neighbors:
            # (-1,-1) (0,-1) (1,-1)
            # (-1, 0)        (1, 0)
            # (-1, 1) (0, 1) (1, 1)

            # row -1
            y0 = yi - 1
            if y0 >= 0:
                x0 = xi - 1
                if x0 >= 0:
                    ts_map[x0, y0, ci] = ti
                x0 = xi
                ts_map[x0, y0, ci] = ti
                x0 = xi + 1
                if x0 < W:
                    ts_map[x0, y0, ci] = ti

            # row 0
            x0 = xi - 1
            if x0 >= 0:
                ts_map[x0, yi, ci] = ti
            x0 = xi + 1
            if x0 < W:
                ts_map[x0, yi, ci] = ti

            # row +1
            y0 = yi + 1
            if y0 < H:
                x0 = xi - 1
                if x0 >= 0:
                    ts_map[x0, y0, ci] = ti
                x0 = xi
                ts_map[x0, y0, ci] = ti
                x0 = xi + 1
                if x0 < W:
                    ts_map[x0, y0, ci] = ti

            # Step 2 (paper): check if present timestamp is within T of the previous value at THIS location
            prev = int(ts_map[xi, yi, ci])
            if ti - prev <= T_us:
                keep_u8[i] = 1
                n_keep += 1

        return keep_u8, n_keep


# -----------------------------
# Core: BAF per-bin processing
# -----------------------------
def baf_process_bin(events, state: BAFState, cfg: BAFComputeConfig) -> Optional[BAFResult]:
    t, x, y, p = _extract_txyp(events)
    n = int(x.shape[0])
    if n == 0:
        return None

    # optional clamp (do it once per bin)
    if cfg.clamp_xy:
        W, H = cfg.resolution
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

    if not _NUMBA_OK:
        raise RuntimeError(
            "Numba is not available, but this BAF implementation uses numba for speed. "
            "Install numba or ask for a pure-numpy fallback (slower)."
        )

    W, H = cfg.resolution
    C = 2 if cfg.use_polarity else 1
    T_us = int(cfg.support_us)

    if cfg.use_polarity:
        c = _pol_to_c(p)
    else:
        c = np.zeros_like(x, dtype=np.int32)

    keep_u8, n_keep = _baf_kernel(
        t.astype(np.int64, copy=False),
        x.astype(np.int32, copy=False),
        y.astype(np.int32, copy=False),
        c.astype(np.int32, copy=False),
        state.ts_map,
        int(W), int(H), int(C),
        int(T_us),
    )

    keep = keep_u8.view(np.bool_)  # 0/1 -> bool, zero-copy view

    stats = {
        "n_in": n,
        "n_keep": int(n_keep),
        "n_drop": int(n - n_keep),
        "keep_rate": float(n_keep) / float(n),
        "support_us": int(cfg.support_us),
        "use_polarity": bool(cfg.use_polarity),
        "numba": True,
    }
    return BAFResult(keepmask=keep, stats=stats)
