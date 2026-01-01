# baselines/ynoise_core.py
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
class YNoiseComputeConfig:
    resolution: Tuple[int, int]          # (W, H)
    bin_us: int                          # used by runner only
    duration_us: int = 10000             # ΔT, microseconds (module default)
    searchRadius: int = 1                # R
    intThreshold: int = 2                # θ (module default)
    include_center: bool = True          # depends on Offsets; keep True to align typical impl
    clamp_xy: bool = False               # robustness


@dataclass
class YNoiseResult:
    keepmask: np.ndarray                 # bool (N,)
    stats: Dict[str, Any]


# -----------------------------
# State: last timestamp + last polarity (per pixel)
# -----------------------------
class YNoiseState:
    """
    YNoise maintains per-pixel last event time and polarity:
      - tlast[y,x]: int64 us, 0 means "never seen"
      - plast[y,x]: uint8 {0,1} polarity of last event
    """
    def __init__(self, resolution: Tuple[int, int]):
        W, H = resolution
        self.W = int(W)
        self.H = int(H)
        self.tlast = np.zeros((H, W), dtype=np.int64)
        self.plast = np.zeros((H, W), dtype=np.uint8)

        self._radius = None
        self._include_center = None
        self.off_dx = None
        self.off_dy = None

    @staticmethod
    def create(resolution: Tuple[int, int], cfg: YNoiseComputeConfig) -> "YNoiseState":
        st = YNoiseState(resolution)
        st.ensure_offsets(cfg.searchRadius, cfg.include_center)
        return st

    def reset(self):
        self.tlast.fill(0)
        self.plast.fill(0)

    def ensure_offsets(self, radius: int, include_center: bool):
        radius = int(radius)
        if self._radius == radius and self._include_center == include_center and self.off_dx is not None:
            return

        dx_list = []
        dy_list = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if not include_center and dx == 0 and dy == 0:
                    continue
                dx_list.append(dx)
                dy_list.append(dy)

        self.off_dx = np.asarray(dx_list, dtype=np.int16)
        self.off_dy = np.asarray(dy_list, dtype=np.int16)
        self._radius = radius
        self._include_center = include_center


# -----------------------------
# Extraction utility (robust)
# -----------------------------
def _empty_txyp():
    return (
        np.empty((0,), dtype=np.int64),   # t_us
        np.empty((0,), dtype=np.int32),   # x
        np.empty((0,), dtype=np.int32),   # y
        np.empty((0,), dtype=np.uint8),   # p01
    )


def _to_p01(p: np.ndarray) -> np.ndarray:
    """
    Convert polarity array to uint8 {0,1}.
    Supports p in {-1,+1}, {0,1}, bool, int.
    """
    p = np.asarray(p)
    if p.size == 0:
        return np.empty((0,), dtype=np.uint8)
    if p.dtype == np.bool_:
        return p.astype(np.uint8, copy=False)
    # numeric: >0 => 1
    return (p.astype(np.int8, copy=False) > 0).astype(np.uint8)


def _extract_txyp(events):
    """
    Extract (t_us, x, y, p01) where p01 is {0,1}.
    Supports dv.EventStore-like .numpy(), numpy structured arrays, and dict.
    Assumes timestamps are already in microseconds int64 (as in your RCF pipeline).
    """
    # dv EventStore-like
    if hasattr(events, "numpy"):
        arr = events.numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
            x = arr["x"].astype(np.int32, copy=False)
            y = arr["y"].astype(np.int32, copy=False)
            if x.size == 0:
                return _empty_txyp()

            if "t" in arr.dtype.fields:
                t = arr["t"].astype(np.int64, copy=False)
            else:
                t = arr["timestamp"].astype(np.int64, copy=False)

            if "p" in arr.dtype.fields:
                p = arr["p"]
            elif "polarity" in arr.dtype.fields:
                p = arr["polarity"]
            else:
                p = np.zeros_like(x, dtype=np.int8)

            p01 = _to_p01(p)
            return t, x, y, p01

        events = arr

    # dict
    if isinstance(events, dict):
        x = np.asarray(events.get("x", []), dtype=np.int32)
        y = np.asarray(events.get("y", []), dtype=np.int32)
        if x.size == 0:
            return _empty_txyp()
        t = np.asarray(events.get("t", events.get("timestamp", np.zeros_like(x))), dtype=np.int64)
        p = np.asarray(events.get("p", events.get("polarity", np.zeros_like(x))), dtype=np.int8)
        p01 = _to_p01(p)
        return t, x, y, p01

    # numpy structured array
    if isinstance(events, np.ndarray) and events.dtype.fields is not None:
        x = events["x"].astype(np.int32, copy=False)
        y = events["y"].astype(np.int32, copy=False)
        if x.size == 0:
            return _empty_txyp()

        if "t" in events.dtype.fields:
            t = events["t"].astype(np.int64, copy=False)
        else:
            t = events["timestamp"].astype(np.int64, copy=False)

        if "p" in events.dtype.fields:
            p = events["p"]
        elif "polarity" in events.dtype.fields:
            p = events["polarity"]
        else:
            p = np.zeros_like(x, dtype=np.int8)

        p01 = _to_p01(p)
        return t, x, y, p01

    raise TypeError("Unsupported events container type for _extract_txyp().")


# -----------------------------
# Numba kernel
# -----------------------------
if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False)
    def _ynoise_kernel(
        t_us: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p01: np.ndarray,
        tlast: np.ndarray,
        plast: np.ndarray,
        off_dx: np.ndarray,
        off_dy: np.ndarray,
        W: int,
        H: int,
        duration_us: int,
        thr_int: int,
    ):
        n = x.shape[0]
        keep_u8 = np.zeros(n, dtype=np.uint8)
        n_keep = 0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t_us[i])
            pol = np.uint8(p01[i])

            density = 0

            # count "recent active pixels with same polarity" in neighborhood
            for k in range(off_dx.shape[0]):
                xx = xi + int(off_dx[k])
                yy = yi + int(off_dy[k])
                if 0 <= xx < W and 0 <= yy < H:
                    tl = int(tlast[yy, xx])
                    if tl != 0:
                        dt = ti - tl  # usually >= 0
                        if dt <= duration_us and plast[yy, xx] == pol:
                            density += 1

            if density >= thr_int:
                keep_u8[i] = 1
                n_keep += 1

            # update current pixel regardless keep/drop
            tlast[yi, xi] = ti
            plast[yi, xi] = pol

        return keep_u8, n_keep


# -----------------------------
# Public API (rcf_core-like)
# -----------------------------
def ynoise_process_bin(events, state: YNoiseState, cfg: YNoiseComputeConfig) -> Optional[YNoiseResult]:
    """
    Process a batch/bin of events and return keepmask.
    State is carried across bins (event-driven).
    """
    if not _NUMBA_OK:
        raise RuntimeError("Numba is required for YNoise baseline speed. Please install numba.")

    t_us, x, y, p01 = _extract_txyp(events)
    n = int(x.shape[0])
    if n == 0:
        return None

    state.ensure_offsets(cfg.searchRadius, cfg.include_center)

    if cfg.clamp_xy:
        W, H = cfg.resolution
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

    W, H = cfg.resolution
    dur = int(cfg.duration_us)
    if dur <= 0:
        dur = 1
    thr = int(cfg.intThreshold)
    if thr < 0:
        thr = 0

    keep_u8, n_keep = _ynoise_kernel(
        t_us, x, y, p01,
        state.tlast, state.plast,
        state.off_dx, state.off_dy,
        int(W), int(H),
        int(dur),
        int(thr),
    )

    keep = keep_u8.view(np.bool_)  # zero-copy view

    stats = {
        "n_in": n,
        "n_keep": int(n_keep),
        "n_drop": int(n - n_keep),
        "keep_rate": float(n_keep) / float(n),
        "duration_us": int(cfg.duration_us),
        "searchRadius": int(cfg.searchRadius),
        "intThreshold": int(cfg.intThreshold),
        "include_center": bool(cfg.include_center),
        "numba": True,
    }
    return YNoiseResult(keepmask=keep, stats=stats)
