# baselines/dwf_core.py
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
class DWFComputeConfig:
    resolution: Tuple[int, int]          # (W, H) for optional clamp
    bin_us: int                          # for pipeline consistency/logging
    bufferSize: int = 36
    searchRadius: int = 9                # L1 radius
    intThreshold: int = 1                # support threshold
    init_mode: str = "zeros"             # "zeros" or "empty"
    clamp_xy: bool = False               # optional robustness


@dataclass
class DWFResult:
    keepmask: np.ndarray                 # bool (N,)
    stats: Dict[str, Any]                # lightweight stats


# -----------------------------
# State: two ring buffers
# -----------------------------
class DWFState:
    def __init__(self, capacity: int, init_mode: str = "zeros"):
        cap = int(capacity)
        if cap <= 0:
            raise ValueError("bufferSize must be > 0")
        self.capacity = cap
        self.init_mode = init_mode

        # ring buffers
        self.real_x = np.empty(cap, dtype=np.int32)
        self.real_y = np.empty(cap, dtype=np.int32)
        self.noise_x = np.empty(cap, dtype=np.int32)
        self.noise_y = np.empty(cap, dtype=np.int32)

        if init_mode == "zeros":
            self.real_x.fill(0); self.real_y.fill(0)
            self.noise_x.fill(0); self.noise_y.fill(0)
            self.real_size = cap
            self.noise_size = cap
            self.real_head = 0
            self.noise_head = 0
        elif init_mode == "empty":
            self.real_size = 0
            self.noise_size = 0
            self.real_head = 0
            self.noise_head = 0
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    @staticmethod
    def create(resolution: Tuple[int, int], cfg: DWFComputeConfig) -> "DWFState":
        _ = resolution
        return DWFState(cfg.bufferSize, cfg.init_mode)

    def reset(self):
        self.__init__(self.capacity, self.init_mode)


# -----------------------------
# Extraction utility
# -----------------------------
def _extract_txyp(events):
    """
    Extract (t, x, y, p) from dv.EventStore-like or numpy structured array/dict.
    DWF uses only x,y but keep consistent signature.
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


# -----------------------------
# Numba kernel
# -----------------------------
if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False)
    def _dwf_kernel(
        x: np.ndarray,
        y: np.ndarray,
        real_x: np.ndarray,
        real_y: np.ndarray,
        noise_x: np.ndarray,
        noise_y: np.ndarray,
        real_size: int,
        noise_size: int,
        real_head: int,
        noise_head: int,
        cap: int,
        R: int,
        thr: int,
    ):
        n = x.shape[0]
        keep_u8 = np.zeros(n, dtype=np.uint8)
        n_keep = 0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])

            support = 0
            kept = False

            # scan real
            for j in range(real_size):
                dx = real_x[j] - xi
                if dx < 0:
                    dx = -dx
                dy = real_y[j] - yi
                if dy < 0:
                    dy = -dy
                if dx + dy <= R:
                    support += 1
                    if support >= thr:
                        kept = True
                        break

            # scan noise if needed
            if not kept:
                for j in range(noise_size):
                    dx = noise_x[j] - xi
                    if dx < 0:
                        dx = -dx
                    dy = noise_y[j] - yi
                    if dy < 0:
                        dy = -dy
                    if dx + dy <= R:
                        support += 1
                        if support >= thr:
                            kept = True
                            break

            if kept:
                keep_u8[i] = 1
                n_keep += 1
                # push to real ring
                if real_size < cap:
                    real_x[real_size] = xi
                    real_y[real_size] = yi
                    real_size += 1
                else:
                    real_x[real_head] = xi
                    real_y[real_head] = yi
                    real_head += 1
                    if real_head == cap:
                        real_head = 0
            else:
                # push to noise ring
                if noise_size < cap:
                    noise_x[noise_size] = xi
                    noise_y[noise_size] = yi
                    noise_size += 1
                else:
                    noise_x[noise_head] = xi
                    noise_y[noise_head] = yi
                    noise_head += 1
                    if noise_head == cap:
                        noise_head = 0

        return keep_u8, n_keep, real_size, noise_size, real_head, noise_head


# -----------------------------
# Core: DWF per-bin processing
# -----------------------------
def dwf_process_bin(events, state: DWFState, cfg: DWFComputeConfig) -> Optional[DWFResult]:
    _t, x, y, _p = _extract_txyp(events)
    n = int(x.shape[0])
    if n == 0:
        return None

    # optional clamp (do it once per bin; cost is negligible)
    if cfg.clamp_xy:
        W, H = cfg.resolution
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

    R = int(cfg.searchRadius)
    thr = int(cfg.intThreshold)
    if thr <= 0:
        thr = 1

    cap = state.capacity

    if not _NUMBA_OK:
        raise RuntimeError(
            "Numba is not available, but this DWF implementation requires numba for speed. "
            "Install numba or switch to the C++ wrapper."
        )

    keep_u8, n_keep, real_size, noise_size, real_head, noise_head = _dwf_kernel(
        x, y,
        state.real_x, state.real_y,
        state.noise_x, state.noise_y,
        int(state.real_size), int(state.noise_size),
        int(state.real_head), int(state.noise_head),
        int(cap), int(R), int(thr)
    )

    # write back state
    state.real_size = int(real_size)
    state.noise_size = int(noise_size)
    state.real_head = int(real_head)
    state.noise_head = int(noise_head)

    keep = keep_u8.view(np.bool_)  # 0/1 -> bool, zero-copy view

    stats = {
        "n_in": n,
        "n_keep": int(n_keep),
        "n_drop": int(n - n_keep),
        "keep_rate": float(n_keep) / float(n),
        "bufferSize": cfg.bufferSize,
        "searchRadius": cfg.searchRadius,
        "intThreshold": cfg.intThreshold,
        "init_mode": cfg.init_mode,
        "numba": True,
    }
    return DWFResult(keepmask=keep, stats=stats)
