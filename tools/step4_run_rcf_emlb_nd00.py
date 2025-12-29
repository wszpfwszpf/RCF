# tools/step4_run_rcf_emlb_nd00.py
# Verification stage:
# - stream aedat4
# - slice into 10ms bins
# - compute RCF per bin
# - write per-bin CSV (time_ms, keep_rate per eta)
# - dump keepmask as 6-column 0/1 uint8 per file (aligned to raw event order)

import os
import time
import csv
from datetime import timedelta
from typing import List

import numpy as np
import dv_processing as dv

from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_config import RCFComputeConfig
from rcf_fast.rcf_core import rcf_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
DATA_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\night"  # day / night
ND_FILTER = "ND64"                      # ND00 / ND04 / ND16 / ND64
BIN_MS = 33                             # 10ms slicing for RCF
tag='night'
# If you want a quick dry-run:
MAX_BINS_PER_FILE = 0                   # 0 = full file

# Output directories
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_rcf_verify"
CSV_DIR = os.path.join(OUT_DIR, "csv")
KEEP_DIR = os.path.join(OUT_DIR, "keepmask")
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
    print("[STEP4] RCF verification run: per-bin CSV + keepmask(6 cols uint8)")
    print(f"[STEP4] Root      : {DATA_ROOT}")
    print(f"[STEP4] ND filter : {ND_FILTER}")
    print(f"[STEP4] Bin size  : {BIN_MS} ms")
    print(f"[STEP4] Out dir   : {OUT_DIR}")
    print("=" * 110)

    files = collect_files(DATA_ROOT, ND_FILTER)
    print(f"[STEP4] Files found: {len(files)}")

    cfg = RCFComputeConfig(bin_us=BIN_MS * 1000)
    eta_list = cfg.eta_list
    n_eta = len(eta_list)

    # One global CSV for this ND
    out_csv = os.path.join(CSV_DIR, f"rcf_profile_{tag}_{ND_FILTER.lower()}.csv")
    header = ["scene", "file", "bin_idx", "n_events", "time_ms"] + [f"keep_rate_eta_{e:.2f}" for e in eta_list]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for fi, path in enumerate(files, 1):
            fname = os.path.basename(path)
            scene = os.path.basename(os.path.dirname(path))
            print(f"[{fi:04d}/{len(files):04d}] {scene} | {fname}")

            # IMPORTANT: reset TS per file
            state = RCFState.create(cfg.resolution)

            reader = dv.io.MonoCameraRecording(path)
            slicer = dv.EventStreamSlicer()

            bin_counter = 0
            keep_chunks: List[np.ndarray] = []  # each: (n_bin, n_eta) uint8

            def on_bin(events: dv.EventStore):
                nonlocal bin_counter
                if MAX_BINS_PER_FILE > 0 and bin_counter >= MAX_BINS_PER_FILE:
                    return

                bin_counter += 1

                t0 = time.perf_counter()
                res = rcf_process_bin(events, state, cfg)
                t1 = time.perf_counter()

                if res is None:
                    return

                n = int(res.score.shape[0])
                elapsed_ms = (t1 - t0) * 1000.0

                # keep rate row
                row = [scene, fname, bin_counter, n, f"{elapsed_ms:.3f}"]

                # keepmask 6 columns
                km = np.empty((n, n_eta), dtype=np.uint8)
                for j, eta in enumerate(eta_list):
                    keep = res.keep_masks[eta]
                    row.append(f"{float(keep.sum()) / max(1, n):.6f}")
                    km[:, j] = keep.astype(np.uint8)

                writer.writerow(row)

                # store keepmask chunk for later concatenation
                keep_chunks.append(km)

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

            keepmask = np.concatenate(keep_chunks, axis=0)  # (N_total, 6)

            # save keepmask as simple 6-col 0/1
            out_npz = os.path.join(
                KEEP_DIR,
                fname.replace(".aedat4", f"_{ND_FILTER.lower()}_keepmask.npz")
            )
            np.savez_compressed(
                out_npz,
                eta_list=np.asarray(eta_list, dtype=np.float32),
                keepmask=keepmask.astype(np.uint8),
            )

            print(f"  keepmask saved: {os.path.basename(out_npz)} | shape={keepmask.shape}")

    print("-" * 110)
    print(f"[STEP4] CSV saved: {out_csv}")
    print(f"[STEP4] keepmask dir: {KEEP_DIR}")
    print("-" * 110)


if __name__ == "__main__":
    main()
