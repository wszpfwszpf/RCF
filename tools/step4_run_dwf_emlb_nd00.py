# tools/step4_run_dwf_emlb_nd00.py
# Verification stage (DWF):
# - stream aedat4
# - slice into BIN_MS chunks (for I/O convenience; DWF itself is event-driven)
# - compute DWF sequentially with persistent state across bins (reset per file)
# - write per-bin CSV (time_ms, keep_rate)
# - dump keepmask as 1-column 0/1 uint8 per file (aligned to raw event order)

import os
import time
import csv
from datetime import timedelta
from typing import List

import numpy as np
import dv_processing as dv

from baselines.dwf_core import DWFState, DWFComputeConfig, dwf_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
DATA_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\night"  # day / night
ND_FILTER = "ND64"                      # ND00 / ND04 / ND16 / ND64
BIN_MS = 33                             # slicing only for batching; DWF is continuous
tag = "night"

# Quick dry-run:
MAX_BINS_PER_FILE = 0                   # 0 = full file

# Output directories (YOU can change this)
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_dwf_verify"
CSV_DIR = os.path.join(OUT_DIR, "csv")
KEEP_DIR = os.path.join(OUT_DIR, "keepmask")

# DWF parameters (match C++ defaults unless you want to test)
DWF_BUFFER_SIZE = 36
DWF_SEARCH_RADIUS = 9
DWF_INT_THRESHOLD = 1
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
    print("[STEP4-DWF] DWF verification run: per-bin CSV + keepmask(1 col uint8)")
    print(f"[STEP4-DWF] Root      : {DATA_ROOT}")
    print(f"[STEP4-DWF] ND filter : {ND_FILTER}")
    print(f"[STEP4-DWF] Bin size  : {BIN_MS} ms (batching only; state is continuous)")
    print(f"[STEP4-DWF] Out dir   : {OUT_DIR}")
    print(f"[STEP4-DWF] Params    : bufferSize={DWF_BUFFER_SIZE}, searchRadius={DWF_SEARCH_RADIUS}, intThreshold={DWF_INT_THRESHOLD}")
    print("=" * 110)

    files = collect_files(DATA_ROOT, ND_FILTER)
    print(f"[STEP4-DWF] Files found: {len(files)}")

    # DWF config (resolution is inferred inside cfg in your code base? Here we set fixed 346x260)
    # If you prefer reading from your global constants, update here.
    cfg = DWFComputeConfig(
        resolution=(346, 260),
        bin_us=BIN_MS * 1000,
        bufferSize=DWF_BUFFER_SIZE,
        searchRadius=DWF_SEARCH_RADIUS,
        intThreshold=DWF_INT_THRESHOLD,
        init_mode="zeros",   # closest to C++ circular_buffer.resize() typical behavior
        clamp_xy=False,
    )

    # One global CSV for this ND
    out_csv = os.path.join(CSV_DIR, f"dwf_profile_{tag}_{ND_FILTER.lower()}.csv")
    header = ["scene", "file", "bin_idx", "n_events", "time_ms", "keep_rate"]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for fi, path in enumerate(files, 1):
            fname = os.path.basename(path)
            scene = os.path.basename(os.path.dirname(path))
            print(f"[{fi:04d}/{len(files):04d}] {scene} | {fname}")

            # IMPORTANT: reset state per file (continuous across bins within the file)
            state = DWFState.create(cfg.resolution, cfg)

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
                res = dwf_process_bin(events, state, cfg)
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
                params=np.array([cfg.bufferSize, cfg.searchRadius, cfg.intThreshold], dtype=np.int32),
            )

            print(f"  keepmask saved: {os.path.basename(out_npz)} | shape={keepmask.shape}")

    print("-" * 110)
    print(f"[STEP4-DWF] CSV saved: {out_csv}")
    print(f"[STEP4-DWF] keepmask dir: {KEEP_DIR}")
    print("-" * 110)


if __name__ == "__main__":
    main()
