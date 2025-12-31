# tools/step4_run_ts_emlb_nd00.py
# Verification stage (TS / Time-Surface):
# - stream aedat4
# - slice into BIN_MS chunks (batching only; TS itself is event-driven)
# - compute TS sequentially with persistent state across bins (reset per file)
# - write per-bin CSV (time_ms, keep_rate)
# - dump keepmask as 1-column 0/1 uint8 per file (aligned to raw event order)

import os
import time
import csv
from datetime import timedelta
from typing import List

import numpy as np
import dv_processing as dv

from baselines.ts_core import TSState, TSComputeConfig, ts_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
DATA_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\day"  # day / night
ND_FILTER = "ND64"                      # ND00 / ND04 / ND16 / ND64
BIN_MS = 33                             # batching only; TS is continuous
tag = "day"

# Quick dry-run:
MAX_BINS_PER_FILE = 0                   # 0 = full file

# Output directories
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_ts_verify"
CSV_DIR = os.path.join(OUT_DIR, "csv")
KEEP_DIR = os.path.join(OUT_DIR, "keepmask")

# TS parameters (match dv-processing module defaults unless you want to test)
TS_DECAY_US = 30000          # tau
TS_SEARCH_RADIUS = 1         # R
TS_FLOAT_THRESHOLD = 0.3     # theta
TS_INCLUDE_CENTER = True     # align to reference implementation
# ============================================================


def collect_files(root: str, nd_filter: str) -> List[str]:
    out = []
    for scene in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fn in sorted(os.listdir(scene_dir)):
            if fn.endswith(".aedat4") and (nd_filter in fn):
                out.append(os.path.join(scene_dir, fn))
    return out


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(KEEP_DIR, exist_ok=True)

    print("=" * 110)
    print("[STEP4-TS] TS verification run: per-bin CSV + keepmask(1 col uint8)")
    print(f"[STEP4-TS] Root      : {DATA_ROOT}")
    print(f"[STEP4-TS] ND filter : {ND_FILTER}")
    print(f"[STEP4-TS] Bin size  : {BIN_MS} ms (batching only; state is continuous)")
    print(f"[STEP4-TS] Out dir   : {OUT_DIR}")
    print(f"[STEP4-TS] Params    : decay_us={TS_DECAY_US}, searchRadius={TS_SEARCH_RADIUS}, floatThreshold={TS_FLOAT_THRESHOLD}, include_center={TS_INCLUDE_CENTER}")
    print("=" * 110)

    files = collect_files(DATA_ROOT, ND_FILTER)
    print(f"[STEP4-TS] Files found: {len(files)}")

    cfg = TSComputeConfig(
        resolution=(346, 260),
        bin_us=BIN_MS * 1000,
        decay_us=TS_DECAY_US,
        searchRadius=TS_SEARCH_RADIUS,
        floatThreshold=TS_FLOAT_THRESHOLD,
        include_center=TS_INCLUDE_CENTER,
        clamp_xy=False,
    )

    out_csv = os.path.join(CSV_DIR, f"ts_profile_{tag}_{ND_FILTER.lower()}.csv")
    header = ["scene", "file", "bin_idx", "n_events", "time_ms", "keep_rate"]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for fi, path in enumerate(files, 1):
            fname = os.path.basename(path)
            scene = os.path.basename(os.path.dirname(path))
            print(f"[{fi:04d}/{len(files):04d}] {scene} | {fname}")

            # IMPORTANT: reset state per file (continuous across bins within the file)
            state = TSState.create(cfg.resolution, cfg)

            reader = dv.io.MonoCameraRecording(path)
            slicer = dv.EventStreamSlicer()

            bin_counter = 0
            keep_chunks: List[np.ndarray] = []  # each: (n_bin,) uint8

            def on_bin(events: dv.EventStore):
                nonlocal bin_counter
                if MAX_BINS_PER_FILE > 0 and bin_counter >= MAX_BINS_PER_FILE:
                    return

                bin_counter += 1

                t0 = time.perf_counter()
                res = ts_process_bin(events, state, cfg)
                t1 = time.perf_counter()

                if res is None:
                    return

                km = res.keepmask  # bool, shape (n,)
                n = int(km.shape[0])
                elapsed_ms = (t1 - t0) * 1000.0
                keep_rate = float(km.sum()) / max(1, n)

                writer.writerow([scene, fname, bin_counter, n, f"{elapsed_ms:.3f}", f"{keep_rate:.6f}"])
                keep_chunks.append(km.astype(np.uint8))

            slicer.doEveryTimeInterval(timedelta(milliseconds=BIN_MS), on_bin)

            # stream
            while reader.isRunning():
                batch = reader.getNextEventBatch()
                if batch is not None:
                    slicer.accept(batch)
                else:
                    break

            if len(keep_chunks) == 0:
                print("  [WARN] No bins / no events produced; skip keepmask save.")
                continue

            keepmask = np.concatenate(keep_chunks, axis=0)  # (N_total,)

            out_npz = os.path.join(
                KEEP_DIR,
                fname.replace(".aedat4", f"_{ND_FILTER.lower()}_keepmask.npz")
            )
            np.savez_compressed(
                out_npz,
                keepmask=keepmask.astype(np.uint8),
                params=np.array(
                    [cfg.decay_us, cfg.searchRadius, cfg.floatThreshold, int(cfg.include_center)],
                    dtype=np.float32
                ),
            )

            print(f"  keepmask saved: {os.path.basename(out_npz)} | shape={keepmask.shape}")

    print("-" * 110)
    print(f"[STEP4-TS] CSV saved: {out_csv}")
    print(f"[STEP4-TS] keepmask dir: {KEEP_DIR}")
    print("-" * 110)


if __name__ == "__main__":
    main()
