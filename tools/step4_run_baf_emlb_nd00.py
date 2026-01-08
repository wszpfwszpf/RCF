# tools/step4_run_baf_emlb_nd00.py
# Verification stage (BAF):
# - stream aedat4
# - slice into BIN_MS chunks (for I/O convenience; BAF itself is event-driven)
# - compute BAF sequentially with persistent state across bins (reset per file)
# - write per-bin CSV (time_ms, keep_rate)
# - dump keepmask as 1-column 0/1 uint8 per file (aligned to raw event order)

import os
import time
import csv
from datetime import timedelta
from typing import List

import numpy as np
import dv_processing as dv

from baselines.baf_core import BAFState, BAFComputeConfig, baf_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
DATA_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\day"  # day / night
ND_FILTER = "ND00"                      # ND00 / ND04 / ND16 / ND64
BIN_MS = 33                             # slicing only for batching; BAF is continuous
tag = "day"

# Quick dry-run:
MAX_BINS_PER_FILE = 0                   # 0 = full file

# Output directories
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_baf_verify"
CSV_DIR = os.path.join(OUT_DIR, "csv")
KEEP_DIR = os.path.join(OUT_DIR, "keepmask")

# BAF parameters
BAF_SUPPORT_US = 3000                   # T (microseconds). Typical values: 1000~10000
BAF_USE_POLARITY = True                 # paper uses x*y*2 map
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
    print("[STEP4-BAF] BAF verification run: per-bin CSV + keepmask(1 col uint8)")
    print(f"[STEP4-BAF] Root      : {DATA_ROOT}")
    print(f"[STEP4-BAF] ND filter : {ND_FILTER}")
    print(f"[STEP4-BAF] Bin size  : {BIN_MS} ms (batching only; state is continuous)")
    print(f"[STEP4-BAF] Out dir   : {OUT_DIR}")
    print(f"[STEP4-BAF] Params    : support_us={BAF_SUPPORT_US}, use_polarity={BAF_USE_POLARITY}")
    print("=" * 110)

    files = collect_files(DATA_ROOT, ND_FILTER)
    print(f"[STEP4-BAF] Files found: {len(files)}")

    cfg = BAFComputeConfig(
        resolution=(346, 260),
        bin_us=BIN_MS * 1000,
        support_us=BAF_SUPPORT_US,
        use_polarity=BAF_USE_POLARITY,
        clamp_xy=False,
        init_ts=-10**18,
    )

    out_csv = os.path.join(CSV_DIR, f"baf_profile_{tag}_{ND_FILTER.lower()}.csv")
    header = ["scene", "file", "bin_idx", "n_events", "time_ms", "keep_rate"]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for fi, path in enumerate(files, 1):
            fname = os.path.basename(path)
            scene = os.path.basename(os.path.dirname(path))
            print(f"[{fi:04d}/{len(files):04d}] {scene} | {fname}")

            # reset state per file (continuous across bins within the file)
            state = BAFState.create(cfg.resolution, cfg)

            reader = dv.io.MonoCameraRecording(path)
            slicer = dv.EventStreamSlicer()

            bin_counter = 0
            keep_chunks: List[np.ndarray] = []

            def on_bin(events: dv.EventStore):
                nonlocal bin_counter
                if MAX_BINS_PER_FILE > 0 and bin_counter >= MAX_BINS_PER_FILE:
                    return

                bin_counter += 1

                t0 = time.perf_counter()
                res = baf_process_bin(events, state, cfg)
                t1 = time.perf_counter()

                if res is None:
                    return

                km = res.keepmask
                n = int(km.shape[0])
                elapsed_ms = (t1 - t0) * 1000.0
                keep_rate = float(km.sum()) / max(1, n)

                writer.writerow([scene, fname, bin_counter, n, f"{elapsed_ms:.3f}", f"{keep_rate:.6f}"])
                keep_chunks.append(km.astype(np.uint8))

            slicer.doEveryTimeInterval(timedelta(milliseconds=BIN_MS), on_bin)

            while reader.isRunning():
                batch = reader.getNextEventBatch()
                if batch is not None:
                    slicer.accept(batch)
                else:
                    break

            if len(keep_chunks) == 0:
                print("  [WARN] No bins / no events produced; skip keepmask save.")
                continue

            keepmask = np.concatenate(keep_chunks, axis=0)

            out_npz = os.path.join(
                KEEP_DIR,
                fname.replace(".aedat4", f"_{ND_FILTER.lower()}_keepmask.npz")
            )
            np.savez_compressed(
                out_npz,
                keepmask=keepmask.astype(np.uint8),
                params=np.array([cfg.support_us, int(cfg.use_polarity)], dtype=np.int32),
            )

            print(f"  keepmask saved: {os.path.basename(out_npz)} | shape={keepmask.shape}")

    print("-" * 110)
    print(f"[STEP4-BAF] CSV saved: {out_csv}")
    print(f"[STEP4-BAF] keepmask dir: {KEEP_DIR}")
    print("-" * 110)


if __name__ == "__main__":
    main()
