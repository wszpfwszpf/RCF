# tools/step4_baf_mesr_rr_all8.py
# 日期：2026-01-08
# 中文说明：
# 该脚本用于在 E-MLB 的 8 个子集（day/night × ND00/04/16/64）上统计 ESR 指标，口径与“正常算法”一致：
# - 先按固定事件数 N_PER_PACKET 将每个文件切成多个 packet
# - 每个 packet 计算一次 ESR(v1)
# - 文件级 mesr = mean(packet_esr)
# - 子集级 mesr = sum(file_mesr) / 文件数（按你的要求：按文件平均，不做 packet 加权）
# 同时根据 keepmask 计算 rr（保留率）：文件 rr = mean(packet_rr)，子集 rr = mean(file_rr)
#
# keepmask：每个文件一个 npz，key="keepmask"，单列 0/1（与原始事件顺序对齐）

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rcf_fast.packet_slice import run_count_slicer, PacketInfo, _StopSlicing
import rcf_fast.esr_core as esr_core


# =========================
# 用户配置（只改这里）
# =========================
DATA_ROOT_BASE = Path(r"../data/emlb")                    # 含 day/ night
KEEP_DIR = Path(r"../data/emlb_baf_verify/keepmask")      # keepmask npz 目录
OUT_DIR = Path(r"../baselines/step4_baf_mesr_rr_all8")
OUT_CSV = OUT_DIR / "baf_mesr_rr_all8_filemean.csv"

SENSOR_W, SENSOR_H = 346, 260
KEEP_IS_ONE = True

# packet 配置：必须与你“正常算法”一致
N_PER_PACKET = 30000

SPLITS = ["day", "night"]
NDS = ["ND00", "ND04", "ND16", "ND64"]
# =========================
# SPLITS = ["day"]
# NDS = ["ND00"]


def _resolve(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def _extract_nd(stem: str) -> Optional[str]:
    m = re.search(r"(ND\d{2})", stem)
    return m.group(1) if m else None


def _collect_aedat4_files(split_dir: Path, nd: str) -> List[Path]:
    files = sorted(split_dir.rglob("*.aedat4"))
    return [p for p in files if _extract_nd(p.stem) == nd]


def _find_keepmask_npz(aedat4_path: Path, keep_dir: Path, split: str, nd: str) -> Optional[Path]:
    stem = aedat4_path.stem

    cand1 = keep_dir / f"{stem}_{split}_{nd.lower()}_keepmask.npz"
    if cand1.exists():
        return cand1

    cand2 = keep_dir / f"{stem}_{nd.lower()}_keepmask.npz"
    if cand2.exists():
        return cand2

    hits = sorted(keep_dir.glob(f"{stem}_*{nd.lower()}*keepmask*.npz"))
    return hits[0] if hits else None


def _load_keepmask_1d(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "keepmask" not in data:
        raise KeyError(f"npz missing 'keepmask', keys={list(data.keys())}")

    km = np.asarray(data["keepmask"])

    # object list -> concat
    if km.ndim == 1 and km.dtype == object:
        km = np.concatenate([np.asarray(x) for x in km], axis=0)

    # (N,1) -> (N,)
    if km.ndim == 2 and km.shape[1] == 1:
        km = km[:, 0]

    if km.ndim != 1:
        raise AssertionError(f"keepmask must be 1D, got {km.shape}")

    km = km.astype(np.uint8, copy=False)
    if not KEEP_IS_ONE:
        km = (1 - km).astype(np.uint8, copy=False)
    return km


def _eventstore_to_xy(events) -> tuple[np.ndarray, np.ndarray]:
    # 与你原 step5 习惯一致：优先 numpy 视图
    if hasattr(events, "numpy"):
        arr = events.numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            return arr["x"].astype(np.int32, copy=False), arr["y"].astype(np.int32, copy=False)

    # fallback：迭代（慢但稳）
    xs, ys = [], []
    for e in events:
        x = getattr(e, "x", None); y = getattr(e, "y", None)
        x = x() if callable(x) else x
        y = y() if callable(y) else y
        xs.append(int(x)); ys.append(int(y))
    return np.asarray(xs, np.int32), np.asarray(ys, np.int32)


def _esr_v1_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    N = int(x.size)
    if N < 2:
        return 0.0
    idx = y.astype(np.int64) * SENSOR_W + x.astype(np.int64)
    counts = np.bincount(idx, minlength=SENSOR_W * SENSOR_H).reshape(SENSOR_H, SENSOR_W)
    return float(esr_core._esr_v1_from_count_surface(counts, N=N, W=SENSOR_W, H=SENSOR_H))


def compute_file_mesr_rr(aedat4_path: Path, km_1d: np.ndarray) -> Tuple[int, float, float, float]:
    """
    文件级（但由 packet 聚合）：
      mesr_raw(file)  = mean(esr_raw(packet))
      mesr_keep(file) = mean(esr_keep(packet))
      rr(file)        = mean(rr(packet))
    返回：packets, mesr_raw, mesr_keep, rr
    """
    km_len = int(km_1d.shape[0])

    esr_raw_list: List[float] = []
    esr_keep_list: List[float] = []
    rr_list: List[float] = []

    def on_packet(events_packet, info: PacketInfo):
        # keepmask 覆盖不足则停止，避免错位
        if info.raw_end > km_len:
            raise _StopSlicing()

        x, y = _eventstore_to_xy(events_packet)

        km_slice = km_1d[info.raw_begin:info.raw_end]
        m = km_slice.astype(bool, copy=False)

        rr_list.append(float(m.mean()) if m.size > 0 else 0.0)

        # raw / keep 的 packet ESR
        esr_raw_list.append(_esr_v1_from_xy(x, y))
        esr_keep_list.append(_esr_v1_from_xy(x[m], y[m]))

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
        max_packets=None,
    )

    packets = int(len(esr_keep_list))
    if packets == 0:
        return 0, 0.0, 0.0, 0.0

    mesr_raw = float(np.mean(esr_raw_list))
    mesr_keep = float(np.mean(esr_keep_list))
    rr = float(np.mean(rr_list))
    return packets, mesr_raw, mesr_keep, rr


def main():
    data_root = _resolve(DATA_ROOT_BASE)
    keep_dir = _resolve(KEEP_DIR)
    out_csv = _resolve(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "split", "nd", "scene", "file",
        "packets",
        "mesr_raw_file", "mesr_keep_file", "rr_file",
    ]

    rows = []

    all_files = 0
    all_sum_raw = 0.0
    all_sum_keep = 0.0
    all_sum_rr = 0.0

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        for nd in NDS:
            files = _collect_aedat4_files(split_dir, nd)
            print(f"[RUN] {split}/{nd} files={len(files)}")

            sub_files = 0
            sub_sum_raw = 0.0
            sub_sum_keep = 0.0
            sub_sum_rr = 0.0

            for i, aedat4_path in enumerate(files, 1):
                scene = aedat4_path.parent.name
                fname = aedat4_path.name

                km_path = _find_keepmask_npz(aedat4_path, keep_dir, split=split, nd=nd)
                if km_path is None:
                    print(f"  [MISS] keepmask not found: {fname}")
                    continue

                km = _load_keepmask_1d(km_path)
                packets, mesr_raw, mesr_keep, rr = compute_file_mesr_rr(aedat4_path, km)

                rows.append([
                    split, nd, scene, fname,
                    packets,
                    f"{mesr_raw:.6f}", f"{mesr_keep:.6f}", f"{rr:.6f}",
                ])

                sub_files += 1
                sub_sum_raw += mesr_raw
                sub_sum_keep += mesr_keep
                sub_sum_rr += rr

                # if i % 10 == 0:
                #     print(f"  ...{i}/{len(files)} last={fname} mesr_raw={mesr_raw:.3f} mesr_keep={mesr_keep:.3f} rr={rr:.3f}")

            # 子集汇总：按文件平均（你要求的 sum/file_count）
            if sub_files > 0:
                sub_mesr_raw = sub_sum_raw / sub_files
                sub_mesr_keep = sub_sum_keep / sub_files
                sub_rr = sub_sum_rr / sub_files
            else:
                sub_mesr_raw = sub_mesr_keep = sub_rr = 0.0

            rows.append([
                split, nd, "TOTAL_FILES_MEAN", "TOTAL_FILES_MEAN",
                sub_files,
                f"{sub_mesr_raw:.6f}", f"{sub_mesr_keep:.6f}", f"{sub_rr:.6f}",
            ])

            print(f"[SUMMARY] {split}/{nd} files={sub_files} mesr_raw={sub_mesr_raw:.6f} mesr_keep={sub_mesr_keep:.6f} rr={sub_rr:.6f}")

            all_files += sub_files
            all_sum_raw += sub_sum_raw
            all_sum_keep += sub_sum_keep
            all_sum_rr += sub_sum_rr

    # 全局汇总
    if all_files > 0:
        all_mesr_raw = all_sum_raw / all_files
        all_mesr_keep = all_sum_keep / all_files
        all_rr = all_sum_rr / all_files
    else:
        all_mesr_raw = all_mesr_keep = all_rr = 0.0

    rows.append([
        "ALL", "ALL", "TOTAL_ALL_FILES_MEAN", "TOTAL_ALL_FILES_MEAN",
        all_files,
        f"{all_mesr_raw:.6f}", f"{all_mesr_keep:.6f}", f"{all_rr:.6f}",
    ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"[SAVED] {out_csv}")
    print(f"[TOTAL] files={all_files} mesr_raw={all_mesr_raw:.6f} mesr_keep={all_mesr_keep:.6f} rr={all_rr:.6f}")


if __name__ == "__main__":
    main()
