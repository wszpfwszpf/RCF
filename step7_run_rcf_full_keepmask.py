# stepX_run_rcf_on_full_dat_keepmask.py
# -*- coding: utf-8 -*-
"""
[STEPX] Run RCF on full-resolution DV .dat files (first 6s only), output keepmask for 6 etas,
and profile per-bin processing time.

Input : <proj_root>/data/DV/*.dat
Output: <proj_root>/data/DV_rcf_full/*.npz   (eta_list + keepmask only)
        <proj_root>/data/DV_rcf_full/profile_fullrcf_bin_time10ms.csv

Notes:
- Full resolution fixed to (W,H)=(1280,720).
- Slice by 10ms (BIN_MS=10).
- Only process first 6 seconds (MAX_TIME_S=6).
- Keepmask is concatenated in raw event order across bins (same design as your ROI keepmask).
"""

from __future__ import annotations
import os
import time
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# ------------------------------------------------------------
# Imports (match your existing project layout)
# ------------------------------------------------------------
# RCF core
from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_config import RCFComputeConfig
from rcf_fast.rcf_core import rcf_process_bin

# PSEE loader (same as your visualization script)
from beam.utils.io.psee_loader import PSEELoader


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
BIN_MS = 33
BIN_US = BIN_MS * 1000

MAX_TIME_S = 6
MAX_TIME_US = int(MAX_TIME_S * 1_000_000)

FULL_RESOLUTION = (1280, 720)  # (W,H)

RECURSIVE = False  # if True: scan data/DV/**.dat


# ------------------------------------------------------------
# Small adapter: make numpy structured array look like dv.EventStore
# so that rcf_core._extract_txyp(events) can call .numpy() and .size()
# ------------------------------------------------------------
class NumpyEventStoreAdapter:
    """Wrap numpy structured array (fields: t,x,y,p) to mimic EventStore interface expected by rcf_core."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr

    def size(self) -> int:
        return int(self._arr.shape[0])


# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------
def find_proj_root(start: Path) -> Path:
    """
    Walk upwards until a folder containing 'data' is found.
    This makes the script robust whether placed under RCF/ or project root.
    """
    cur = start.resolve()
    for _ in range(6):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    # fallback: current folder
    return start.resolve()


def list_dat_files(dv_dir: Path) -> List[Path]:
    if RECURSIVE:
        return sorted([p for p in dv_dir.rglob("*.dat") if p.is_file()])
    else:
        return sorted([p for p in dv_dir.glob("*.dat") if p.is_file()])


def percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_one_file(dat_path: Path, out_dir: Path, cfg: RCFComputeConfig) -> None:
    """
    Process one .dat:
    - slice into 10ms bins (up to first 6s)
    - compute keepmask for each eta
    - concatenate keepmask along raw event order
    - record per-bin RCF processing time
    """
    print(f"  file: {dat_path.name}")

    # reset TS per file (important)
    state = RCFState.create(cfg.resolution)

    loader = PSEELoader(str(dat_path))

    # We only process first 6 seconds => at most 600 bins for 10ms
    max_bins = int(np.ceil(MAX_TIME_US / BIN_US))

    eta_list = list(cfg.eta_list)
    n_eta = len(eta_list)

    keep_chunks: List[np.ndarray] = []   # each: (n_events_in_bin, n_eta) uint8
    time_ms_list: List[float] = []       # per-bin processing time
    n_events_list: List[int] = []        # per-bin event count

    for b in range(max_bins):
        # This yields all events in [current_time, current_time+BIN_US)
        ev_np = loader.load_delta_t(BIN_US)

        # Stop if we've reached beyond 6 seconds in time axis
        # (PSEELoader updates current_time internally)
        if loader.current_time > MAX_TIME_US and ev_np.size == 0:
            break

        if ev_np is None or ev_np.size == 0:
            # Still count bin time as 0? Here we skip RCF to reflect true processing load.
            continue

        # Adapter -> rcf_process_bin expects an object with .numpy() and .size()
        ev_store = NumpyEventStoreAdapter(ev_np)

        t0 = time.perf_counter()
        res = rcf_process_bin(ev_store, state, cfg)
        t1 = time.perf_counter()

        if res is None:
            continue

        elapsed_ms = (t1 - t0) * 1000.0
        time_ms_list.append(elapsed_ms)

        n = int(res.score.shape[0])
        n_events_list.append(n)

        km = np.empty((n, n_eta), dtype=np.uint8)
        for j, eta in enumerate(eta_list):
            km[:, j] = res.keep_masks[eta].astype(np.uint8)

        keep_chunks.append(km)

        # Optional early stop if time exceeded and events become empty soon after
        if loader.current_time >= MAX_TIME_US and b >= max_bins - 1:
            break

    # save keepmask
    if len(keep_chunks) == 0:
        print("    [WARN] no events/bins in first 6s -> skip keepmask saving")
        return

    keepmask = np.concatenate(keep_chunks, axis=0)  # (N_total, 6)

    out_npz = out_dir / f"{dat_path.stem}_keepmask_{BIN_MS}ms.npz"
    np.savez_compressed(
        out_npz,
        eta_list=np.asarray(eta_list, dtype=np.float32),
        keepmask=keepmask.astype(np.uint8),
    )

    # print speed stats
    tarr = np.asarray(time_ms_list, dtype=np.float64)
    mean_ms = float(tarr.mean()) if tarr.size > 0 else float("nan")
    p95_ms = percentile(tarr, 95)
    p99_ms = percentile(tarr, 99)

    # keep rates (overall) for a quick sanity check
    keep_rates = keepmask.mean(axis=0)  # mean of 0/1
    keep_rates_str = " ".join([f"{kr:.4f}" for kr in keep_rates])

    print(f"    keepmask: {keepmask.shape} saved -> {out_npz.name}")
    print(f"    keep_rate(eta={eta_list}) : {keep_rates_str}")
    print(f"    speed(ms/bin): mean={mean_ms:.3f}, p95={p95_ms:.3f}, p99={p99_ms:.3f} | bins={tarr.size}")


def main():
    proj_root = find_proj_root(Path(__file__).resolve().parent)
    dv_dir = proj_root / "data" / "DV"
    out_dir = proj_root / "data" / "DV_rcf_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_dat_files(dv_dir)
    print("=" * 100)
    print("[STEPX] Full-res RCF keepmask on DV .dat (first 6s)")
    print(f"[IN ] DV_DIR   : {dv_dir}")
    print(f"[OUT] OUT_DIR  : {out_dir}")
    print(f"[CFG] BIN_MS   : {BIN_MS} ms")
    print(f"[CFG] MAX_TIME : {MAX_TIME_S} s")
    print(f"[CFG] RES      : {FULL_RESOLUTION} (W,H)")
    print(f"[INFO] Files   : {len(files)}")
    print("=" * 100)

    if len(files) == 0:
        print("[ERROR] No .dat files found under data/DV")
        return

    # Build cfg, then override resolution AFTER creation (as you要求的方式)
    cfg = RCFComputeConfig(bin_us=BIN_US)

    # robust override for frozen dataclass and non-frozen versions
    try:
        cfg.resolution = FULL_RESOLUTION
    except Exception:
        object.__setattr__(cfg, "resolution", FULL_RESOLUTION)

    eta_list = cfg.eta_list
    print(f"[CFG] eta_list ({len(eta_list)}) : {eta_list}")

    # Global CSV profiling (per-file aggregated speed is enough; if you want per-bin rows,
    # you can extend here easily)
    profile_csv = out_dir / "profile_fullrcf_bin_time33ms.csv"
    with open(profile_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "bin_ms", "max_time_s", "resolution_w", "resolution_h",
                         "eta_list", "mean_ms_per_bin", "p95_ms_per_bin", "p99_ms_per_bin", "n_bins_profiled"])

        for i, dat_path in enumerate(files, 1):
            print(f"[{i:03d}/{len(files):03d}] {dat_path.name}")

            # Run file and also capture timing summary by reading back printed arrays?
            # We keep it simple: re-run minimal speed collection inside a small wrapper.
            # -> To avoid duplication, we do a lightweight second pass in-memory is not desired.
            # Instead, we collect timing by temporarily capturing output inside run_one_file? too heavy.
            # Practical approach: compute summary inside a local block here.

            state = RCFState.create(cfg.resolution)
            loader = PSEELoader(str(dat_path))

            max_bins = int(np.ceil(MAX_TIME_US / BIN_US))
            time_ms_list: List[float] = []

            # Only profiling timings (not saving keepmask) would be faster, but you need keepmask.
            # So we do the full run and compute summary inside run_one_file, then do a profiling-only pass?
            # Not worth. We'll compute timings in a single pass by duplicating minimal logic here is messy.
            #
            # Therefore: for CSV, we re-open the saved keepmask and estimate bins? not possible.
            # Best: we do keepmask run AND timing summary in one loop here, then save keepmask and write CSV.
            #
            # -> We implement full loop inline here for clean single-pass behavior.

            eta_list = list(cfg.eta_list)
            n_eta = len(eta_list)

            keep_chunks: List[np.ndarray] = []
            for b in range(max_bins):
                ev_np = loader.load_delta_t(BIN_US)
                if loader.current_time > MAX_TIME_US and ev_np.size == 0:
                    break
                if ev_np is None or ev_np.size == 0:
                    continue

                ev_store = NumpyEventStoreAdapter(ev_np)

                t0 = time.perf_counter()
                res = rcf_process_bin(ev_store, state, cfg)
                t1 = time.perf_counter()

                if res is None:
                    continue

                time_ms_list.append((t1 - t0) * 1000.0)

                n = int(res.score.shape[0])
                km = np.empty((n, n_eta), dtype=np.uint8)
                for j, eta in enumerate(eta_list):
                    km[:, j] = res.keep_masks[eta].astype(np.uint8)
                keep_chunks.append(km)

            if len(keep_chunks) == 0:
                print("  [WARN] no events/bins in first 6s -> skip")
                continue

            keepmask = np.concatenate(keep_chunks, axis=0)
            out_npz = out_dir / f"{dat_path.stem}_keepmask_{BIN_MS}ms.npz"
            np.savez_compressed(
                out_npz,
                eta_list=np.asarray(eta_list, dtype=np.float32),
                keepmask=keepmask.astype(np.uint8),
            )

            tarr = np.asarray(time_ms_list, dtype=np.float64)
            mean_ms = float(tarr.mean()) if tarr.size > 0 else float("nan")
            p95_ms = percentile(tarr, 95)
            p99_ms = percentile(tarr, 99)

            keep_rates = keepmask.mean(axis=0)
            print(f"  saved: {out_npz.name} | keepmask={keepmask.shape} | keep_rate={keep_rates}")
            print(f"  speed: mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms p99={p99_ms:.3f}ms | bins={tarr.size}")

            writer.writerow([
                dat_path.stem,
                BIN_MS,
                MAX_TIME_S,
                FULL_RESOLUTION[0],
                FULL_RESOLUTION[1],
                ";".join([f"{e:.2f}" for e in eta_list]),
                f"{mean_ms:.6f}",
                f"{p95_ms:.6f}",
                f"{p99_ms:.6f}",
                int(tarr.size),
            ])

    print("-" * 100)
    print(f"[DONE] keepmasks saved to: {out_dir}")
    print(f"[DONE] profile CSV saved : {profile_csv}")
    print("-" * 100)


if __name__ == "__main__":
    main()
