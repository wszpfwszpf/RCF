# step5_run_rcf_on_roinpz_keepmask.py
# ------------------------------------------------------------
# Input : ROI npz files (t,x,y,p), timestamps in microseconds (int64)
# Task  : slice by 10ms bins, run RCF per bin, output keepmask for 6 etas
# Output: DV_roi_npz_km/<same_name>_keepmask_10ms.npz
#         - eta_list: (6,) float32
#         - keepmask: (N,6) uint8, aligned to raw event order
# ------------------------------------------------------------

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_config import RCFComputeConfig
from rcf_fast.rcf_core import rcf_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
ROI_NPZ_DIR = r"data\DV_roi_npz"              # input: ROI events npz folder
OUT_KM_DIR  = r"data\DV_roi_npz_km"           # output: keepmask folder

BIN_MS = 10                                   # slice by 10ms
RESOLUTION = (256, 224)                       # (W, H) for ROI

# Optional: only process first K files for quick test (0 = all)
MAX_FILES = 0

# Optional: sanity check bounds
CHECK_XY_BOUNDS = True
# ============================================================
ROI_X0 = 600 - 256 // 2   # 472
ROI_Y0 = 200 - 224 // 2   # 88


class NpzEventStore:
    """
    Minimal adapter to match the interface expected by rcf_process_bin():
    - .numpy() returns a structured array with fields: t/x/y/p (or timestamp/polarity)
    - .size or .size() returns number of events
    """
    __slots__ = ("_arr",)

    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: Optional[np.ndarray] = None):
        if p is None:
            p = np.zeros_like(x, dtype=np.int8)

        # structured array to satisfy rcf_core._extract_txyp()
        arr = np.empty(t.shape[0], dtype=[("t", "i8"), ("x", "i4"), ("y", "i4"), ("p", "i1")])
        arr["t"] = t.astype(np.int64, copy=False)
        arr["x"] = x.astype(np.int32, copy=False)
        arr["y"] = y.astype(np.int32, copy=False)
        arr["p"] = p.astype(np.int8, copy=False)
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def size(self) -> int:
        return int(self._arr.shape[0])


def collect_npz_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"ROI_NPZ_DIR not found: {root}")
    out = []
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith(".npz"):
            out.append(os.path.join(root, fn))
    return out


def _load_txyp(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)

    # Required fields
    if "t" not in data or "x" not in data or "y" not in data:
        raise KeyError(f"{os.path.basename(npz_path)} missing keys. Need at least t/x/y. Keys={list(data.keys())}")

    t = data["t"].astype(np.int64, copy=False)
    x = data["x"].astype(np.int32, copy=False)
    y = data["y"].astype(np.int32, copy=False)

    # Optional polarity
    if "p" in data:
        p = data["p"].astype(np.int8, copy=False)
    else:
        p = np.zeros_like(x, dtype=np.int8)

    # Ensure 1-D
    t = np.ravel(t)
    x = np.ravel(x)
    y = np.ravel(y)
    p = np.ravel(p)

    if t.size == 0:
        return t, x, y, p

    # Ensure time-sorted for bin slicing
    if not np.all(t[1:] >= t[:-1]):
        order = np.argsort(t, kind="mergesort")  # stable
        t = t[order]
        x = x[order]
        y = y[order]
        p = p[order]

    return t, x, y, p


def _slice_bins_by_time(t_us: np.ndarray, bin_us: int) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) for contiguous bins.
    Assumes t_us is sorted non-decreasing.
    """
    n = int(t_us.shape[0])
    if n == 0:
        return []

    t0 = int(t_us[0])
    # bin id per event
    bid = (t_us - t0) // bin_us
    # find boundaries where bid changes
    # indices where new bin starts
    change = np.nonzero(bid[1:] != bid[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return list(zip(starts.tolist(), ends.tolist()))


def compute_keepmask_for_file(npz_path: str, cfg: RCFComputeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      eta_list: (6,) float32
      keepmask: (N,6) uint8 aligned to raw event order (time-sorted order if input unsorted)
    """
    t, x, y, p = _load_txyp(npz_path)

    # shift to ROI-local coordinates
    x = x - ROI_X0
    y = y - ROI_Y0

    n_total = int(t.shape[0])
    eta_list = np.asarray(cfg.eta_list, dtype=np.float32)
    n_eta = int(len(cfg.eta_list))

    if n_total == 0:
        keepmask = np.zeros((0, n_eta), dtype=np.uint8)
        return eta_list, keepmask

    W, H = cfg.resolution
    if CHECK_XY_BOUNDS:
        if (x.min(initial=0) < 0) or (y.min(initial=0) < 0) or (x.max(initial=-1) >= W) or (y.max(initial=-1) >= H):
            raise ValueError(
                f"{os.path.basename(npz_path)} has out-of-bound events for resolution {cfg.resolution}. "
                f"x:[{x.min()}, {x.max()}], y:[{y.min()}, {y.max()}]"
            )

    # per-file state (IMPORTANT)
    state = RCFState.create(cfg.resolution)

    # slice into bins
    bins = _slice_bins_by_time(t, cfg.bin_us)

    keep_chunks: List[np.ndarray] = []

    for (s, e) in bins:
        if e <= s:
            continue

        ev_store = NpzEventStore(t[s:e], x[s:e], y[s:e], p[s:e])
        res = rcf_process_bin(ev_store, state, cfg)
        if res is None:
            # no events in this bin (shouldn't happen if e>s), skip
            continue

        n_bin = int(res.score.shape[0])
        km = np.empty((n_bin, n_eta), dtype=np.uint8)
        for j, eta in enumerate(cfg.eta_list):
            km[:, j] = res.keep_masks[eta].astype(np.uint8, copy=False)
        keep_chunks.append(km)

    if len(keep_chunks) == 0:
        keepmask = np.zeros((0, n_eta), dtype=np.uint8)
    else:
        keepmask = np.concatenate(keep_chunks, axis=0)

    # Safety: should align to total events after sorting (if any)
    if keepmask.shape[0] != n_total:
        raise RuntimeError(
            f"Keepmask length mismatch for {os.path.basename(npz_path)}: "
            f"keepmask={keepmask.shape[0]} vs events={n_total}. "
            f"Check bin slicing / event ordering."
        )

    return eta_list, keepmask


def main():
    os.makedirs(OUT_KM_DIR, exist_ok=True)

    files = collect_npz_files(ROI_NPZ_DIR)
    if MAX_FILES > 0:
        files = files[:MAX_FILES]

    print("=" * 100)
    print("[STEP5] RCF keepmask on ROI npz")
    print(f"[IN ] ROI_NPZ_DIR : {ROI_NPZ_DIR}")
    print(f"[OUT] OUT_KM_DIR  : {OUT_KM_DIR}")
    print(f"[CFG] BIN_MS      : {BIN_MS} ms")
    print(f"[CFG] RESOLUTION  : {RESOLUTION} (W,H)")
    print(f"[INFO] Files      : {len(files)}")
    print("=" * 100)

    # Build config (DO NOT modify class definition; override instance fields externally)
    cfg = RCFComputeConfig(bin_us=BIN_MS * 1000)

    # override resolution for ROI (dataclass is frozen -> use object.__setattr__)
    object.__setattr__(cfg, "resolution", RESOLUTION)

    eta_list = cfg.eta_list
    n_eta = len(eta_list)
    print(f"[CFG] eta_list ({n_eta}) : {eta_list}")

    for i, npz_path in enumerate(files, 1):
        base = os.path.basename(npz_path)
        stem = os.path.splitext(base)[0]
        print(f"[{i:03d}/{len(files):03d}] {base}")

        eta_arr, keepmask = compute_keepmask_for_file(npz_path, cfg)

        out_path = os.path.join(OUT_KM_DIR, f"{stem}_keepmask_{BIN_MS}ms.npz")
        np.savez_compressed(
            out_path,
            eta_list=eta_arr.astype(np.float32),
            keepmask=keepmask.astype(np.uint8),
            resolution=np.asarray(RESOLUTION, dtype=np.int32),
            bin_ms=np.int32(BIN_MS),
            src_file=base,
        )

        keep_rates = keepmask.mean(axis=0) if keepmask.size > 0 else np.zeros((n_eta,), dtype=np.float32)
        print(f"  saved: {os.path.basename(out_path)} | keepmask={keepmask.shape} | keep_rate={keep_rates}")

    print("-" * 100)
    print("[DONE] All keepmasks saved.")
    print("-" * 100)


if __name__ == "__main__":
    main()
