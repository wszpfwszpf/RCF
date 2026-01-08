# tools/step4_run_knoise_emlb_all8.py
# 日期：2026-01-08
# 中文说明：
# 跑完整 E-MLB（8 个子集）：{day, night} × {ND00, ND04, ND16, ND64}
# - 流式读取 .aedat4
# - 按 BIN_MS 切成时间 bin（仅做批处理；KNoise 是事件级判别，但需要跨 bin 保持 state）
# - 每个文件 reset state；文件内跨 bin 连续更新行/列 FIFO 记忆
# - 输出 per-bin profile CSV（time_ms, keep_rate）
# - 输出 keepmask（按原始事件顺序对齐，一列 0/1 uint8）到 keepmask 子目录
#
# 依赖：
# - dv_processing
# - baselines/knoise_core.py（你已要求按 baf_core.py 风格实现）
#
# keepmask 文件命名对齐你当前 BAF/DWF 管线：
#   <orig>_<split>_<nd>_keepmask.npz

import os
import time
import csv
from datetime import timedelta
from typing import List

import numpy as np
import dv_processing as dv

from baselines.knoise_core import KNoiseState, KNoiseComputeConfig, knoise_process_bin


# ============================================================
# User configuration (edit here)
# ============================================================
DATA_ROOT_BASE = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb"  # contains day/ night folders
BIN_MS = 33

# Quick dry-run:
MAX_BINS_PER_FILE = 0  # 0 = full file

# Output base directory (will create csv/keepmask subfolders)
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_knoise_verify"

# KNoise parameters
KNOISE_SUPPORT_US = 3000
KNOISE_USE_POLARITY = True
KNOISE_FIFO_K = 2  # “two blocks of memory” 对应的 FIFO 深度（软件等价）

# Fixed sensor resolution for your pipeline
SENSOR_W, SENSOR_H = 346, 260

# Subsets to run
SPLITS = ["day", "night"]
NDS = ["ND00", "ND04", "ND16", "ND64"]

# SPLITS = ["day"]
# NDS = ["ND00"]
# ============================================================


def collect_files(root_split: str, nd_filter: str) -> List[str]:
    """
    root_split: .../emlb/day or .../emlb/night
    """
    out = []
    for scene in sorted(os.listdir(root_split)):
        scene_dir = os.path.join(root_split, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fn in sorted(os.listdir(scene_dir)):
            if fn.endswith(".aedat4") and (nd_filter in fn):
                out.append(os.path.join(scene_dir, fn))
    return out


def run_one_subset(split: str, nd: str, cfg: KNoiseComputeConfig, csv_dir: str, keep_dir: str):
    data_root = os.path.join(DATA_ROOT_BASE, split)
    assert os.path.isdir(data_root), f"Split folder not found: {data_root}"

    files = collect_files(data_root, nd)
    print(f"[STEP4-KNOISE] Subset: {split}/{nd} | files={len(files)}")

    out_csv = os.path.join(csv_dir, f"knoise_profile_{split}_{nd.lower()}.csv")
    header = ["scene", "file", "bin_idx", "n_events", "time_ms", "keep_rate"]

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)

        for fi, path in enumerate(files, 1):
            fname = os.path.basename(path)
            scene = os.path.basename(os.path.dirname(path))
            print(f"  [{fi:04d}/{len(files):04d}] {scene} | {fname}")

            # reset state per file (continuous across bins within the file)
            state = KNoiseState.create(cfg.resolution, cfg)

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
                res = knoise_process_bin(events, state, cfg)
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
                print("    [WARN] No bins / no events; skip keepmask save.")
                continue

            keepmask = np.concatenate(keep_chunks, axis=0)

            # keepmask filename: <orig>_<split>_<nd>_keepmask.npz
            out_npz = os.path.join(
                keep_dir,
                fname.replace(".aedat4", f"_{split}_{nd.lower()}_keepmask.npz")
            )
            np.savez_compressed(
                out_npz,
                keepmask=keepmask.astype(np.uint8),
                params=np.array(
                    [cfg.support_us, int(cfg.use_polarity), int(cfg.fifo_k)],
                    dtype=np.int32
                ),
            )

            print(f"    keepmask saved: {os.path.basename(out_npz)} | shape={keepmask.shape}")

    print(f"[STEP4-KNOISE] CSV saved: {out_csv}")


def main():
    csv_dir = os.path.join(OUT_DIR, "csv")
    keep_dir = os.path.join(OUT_DIR, "keepmask")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)

    print("=" * 110)
    print("[STEP4-KNOISE] Full E-MLB run (8 subsets): day/night x ND00/ND04/ND16/ND64")
    print(f"[STEP4-KNOISE] Data base : {DATA_ROOT_BASE}")
    print(f"[STEP4-KNOISE] Bin size  : {BIN_MS} ms (batching only; state is continuous)")
    print(f"[STEP4-KNOISE] Out dir   : {OUT_DIR}")
    print(f"[STEP4-KNOISE] Params    : support_us={KNOISE_SUPPORT_US}, use_polarity={KNOISE_USE_POLARITY}, fifo_k={KNOISE_FIFO_K}")
    print("=" * 110)

    cfg = KNoiseComputeConfig(
        resolution=(SENSOR_W, SENSOR_H),
        bin_us=BIN_MS * 1000,
        support_us=KNOISE_SUPPORT_US,
        use_polarity=KNOISE_USE_POLARITY,
        fifo_k=KNOISE_FIFO_K,
        clamp_xy=False,
        init_ts=-10**18,
    )

    for split in SPLITS:
        for nd in NDS:
            print("-" * 110)
            run_one_subset(split, nd, cfg, csv_dir, keep_dir)

    print("=" * 110)
    print("[STEP4-KNOISE] All done.")
    print(f"[STEP4-KNOISE] CSV dir    : {csv_dir}")
    print(f"[STEP4-KNOISE] Keepmask dir: {keep_dir}")
    print("=" * 110)


if __name__ == "__main__":
    main()
