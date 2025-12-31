# baselines/ts_core.py
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
class TSComputeConfig:
    resolution: Tuple[int, int]          # (W, H)
    bin_us: int                          # used by runner, TS itself is event-driven
    decay_us: int = 30000                # tau, microseconds
    searchRadius: int = 1                # neighborhood radius
    floatThreshold: float = 0.3          # keep if avg(exp(-(dt)/tau)) >= threshold
    include_center: bool = True          # whether offsets include (0,0)
    clamp_xy: bool = False               # robustness


@dataclass
class TSResult:
    keepmask: np.ndarray                 # bool (N,)
    stats: Dict[str, Any]


# -----------------------------
# State: two SAE (last-timestamp maps) for polarity
# -----------------------------
class TSState:
    """
    SAE-like state: last timestamp per pixel, separated by polarity.
    tpos[y,x], tneg[y,x] store last event timestamp (int64 us); 0 means "never seen".
    """
    def __init__(self, resolution: Tuple[int, int]):
        W, H = resolution
        self.W = int(W)
        self.H = int(H)
        self.tpos = np.zeros((H, W), dtype=np.int64)
        self.tneg = np.zeros((H, W), dtype=np.int64)

        self._radius = None
        self._include_center = None
        self.off_dx = None
        self.off_dy = None

    @staticmethod
    def create(resolution: Tuple[int, int], cfg: TSComputeConfig) -> "TSState":
        st = TSState(resolution)
        st.ensure_offsets(cfg.searchRadius, cfg.include_center)
        return st

    def reset(self):
        self.tpos.fill(0)
        self.tneg.fill(0)

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
    For empty p, returns empty uint8.
    """
    p = np.asarray(p)
    if p.size == 0:
        return np.empty((0,), dtype=np.uint8)

    # bool polarity: True as positive
    if p.dtype == np.bool_:
        return p.astype(np.uint8, copy=False)

    # numeric polarity: treat >0 as positive
    # works for {-1,+1} and {0,1} and any int/float
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

        # fallback
        events = arr

    # dict container
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
    def _ts_kernel(
        t_us: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p01: np.ndarray,
        tpos: np.ndarray,
        tneg: np.ndarray,
        off_dx: np.ndarray,
        off_dy: np.ndarray,
        W: int,
        H: int,
        tau_us: int,
        thr: float,
    ):
        n = x.shape[0]
        keep_u8 = np.zeros(n, dtype=np.uint8)
        n_keep = 0

        inv_tau = 1.0 / float(tau_us) if tau_us > 0 else 1.0

        for i in range(n):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t_us[i])
            pol = int(p01[i])  # 1=pos, 0=neg

            sae = tpos if pol == 1 else tneg

            acc = 0.0
            cnt = 0

            for k in range(off_dx.shape[0]):
                xx = xi + int(off_dx[k])
                yy = yi + int(off_dy[k])
                if 0 <= xx < W and 0 <= yy < H:
                    t_last = int(sae[yy, xx])
                    if t_last != 0:
                        dt = ti - t_last  # usually >= 0
                        acc += np.exp(-float(dt) * inv_tau)
                        cnt += 1

            surface = acc / cnt if cnt > 0 else 0.0

            if surface >= thr:
                keep_u8[i] = 1
                n_keep += 1

            # update regardless keep/drop
            sae[yi, xi] = ti

        return keep_u8, n_keep


# -----------------------------
# Public API (rcf_core-like)
# -----------------------------
def ts_process_bin(events, state: TSState, cfg: TSComputeConfig) -> Optional[TSResult]:
    """
    Process a batch/bin of events and return keepmask.
    State is carried across bins (event-driven).
    """
    if not _NUMBA_OK:
        raise RuntimeError("Numba is required for TS baseline speed. Please install numba.")

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
    tau = int(cfg.decay_us)
    if tau <= 0:
        tau = 1

    keep_u8, n_keep = _ts_kernel(
        t_us, x, y, p01,
        state.tpos, state.tneg,
        state.off_dx, state.off_dy,
        int(W), int(H),
        int(tau),
        float(cfg.floatThreshold),
    )

    keep = keep_u8.view(np.bool_)  # zero-copy view

    stats = {
        "n_in": n,
        "n_keep": int(n_keep),
        "n_drop": int(n - n_keep),
        "keep_rate": float(n_keep) / float(n),
        "decay_us": int(cfg.decay_us),
        "searchRadius": int(cfg.searchRadius),
        "floatThreshold": float(cfg.floatThreshold),
        "include_center": bool(cfg.include_center),
        "numba": True,
    }
    return TSResult(keepmask=keep, stats=stats)
