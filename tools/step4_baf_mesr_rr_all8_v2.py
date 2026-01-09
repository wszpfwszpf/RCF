# tools/step4_baf_mesr_rr_all8_v2.py
# 日期：2026-01-08
# 中文说明：
# v2：验证“原论文是否在降噪后的事件流上（denoised stream）按 30000 事件切包计算 ESR”。
# 输入仍为 raw aedat4 + keepmask（不写回文件）：
# - raw: 按固定事件数 N_PER_PACKET 切 packet，计算 esr_raw(packet)，文件 mesr_raw = mean(packet_esr_raw)
# - keep(v2): 先用 keepmask 从整条 raw 流构造 denoised stream（保留事件子集），
#            再在 denoised stream 上按 N_PER_PACKET 重新切 packet，计算 esr_keep(packet)，
#            文件 mesr_keep = mean(packet_esr_keep)
# rr 定义：
# - 文件级 rr = N_keep / N_raw（不再考虑 packet）
# 子集汇总：
# - mesr_raw / mesr_keep：按文件平均（sum(file_mesr)/files）
# - rr：按文件平均（mean(file_rr)）
#
# 参照：tools/step4_baf_mesr_rr_all8.py 的结构与参数（仅改 keep 的计算口径和 rr 定义）

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
#baf
# DATA_ROOT_BASE = Path(r"../data/emlb")                    # 含 day/ night
# KEEP_DIR = Path(r"../data/emlb_baf_verify/keepmask")      # keepmask npz 目录
# OUT_DIR = Path(r"../baselines/step4_baf_mesr_rr_all8_v2")
# OUT_CSV = OUT_DIR / "baf_mesr_rr_all8_v2_filemean.csv"

#knoise
DATA_ROOT_BASE = Path(r"../data/emlb")                    # 含 day/ night
KEEP_DIR = Path(r"../data/emlb_knoise_verify/keepmask")      # keepmask npz 目录
OUT_DIR = Path(r"../baselines/step4_knoise_mesr_rr_all8_v2")
OUT_CSV = OUT_DIR / "knoise_mesr_rr_all8_v2_filemean.csv"

SENSOR_W, SENSOR_H = 346, 260
KEEP_IS_ONE = True

# packet 配置：必须与你“正常算法”一致
N_PER_PACKET = 30000

SPLITS = ["day", "night"]
NDS = ["ND00", "ND04", "ND16", "ND64"]
# =========================
# SPLITS = ["night"]
# NDS = ["ND64"]


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
    if hasattr(events, "numpy"):
        arr = events.numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            return arr["x"].astype(np.int32, copy=False), arr["y"].astype(np.int32, copy=False)

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


def _mesr_raw_from_aedat4(aedat4_path: Path, km_len: int) -> Tuple[int, float]:
    """
    raw 口径不变：在 raw 流上按 N_PER_PACKET 切 packet，算 packet ESR，文件 mesr=mean。
    """
    esr_list: List[float] = []

    def on_packet(events_packet, info: PacketInfo):
        # keepmask 覆盖不足则停止，避免错位（保持与你原脚本一致）
        if info.raw_end > km_len:
            raise _StopSlicing()

        x, y = _eventstore_to_xy(events_packet)
        esr_list.append(_esr_v1_from_xy(x, y))

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
        max_packets=None,
    )

    packets = int(len(esr_list))
    if packets == 0:
        return 0, 0.0
    return packets, float(np.mean(esr_list))


def _mesr_keep_v2_from_inmemory_denoised(x_keep: np.ndarray, y_keep: np.ndarray) -> Tuple[int, float]:
    """
    v2 keep 口径：
      在 denoised stream（保留事件序列）上按 N_PER_PACKET 重新切 packet，
      每个 packet 计算 ESR，文件 mesr=mean(packet_esr)。
    """
    N = int(x_keep.size)
    if N <= 0:
        return 0, 0.0

    esr_list: List[float] = []
    for s in range(0, N, N_PER_PACKET):
        e = min(s + N_PER_PACKET, N)
        esr_list.append(_esr_v1_from_xy(x_keep[s:e], y_keep[s:e]))

    packets = int(len(esr_list))
    return packets, float(np.mean(esr_list)) if packets > 0 else 0.0


def compute_file_mesr_rr_v2(aedat4_path: Path, km_1d: np.ndarray) -> Tuple[int, int, float, float, float]:
    """
    文件级：
      mesr_raw(file): raw 流 packet ESR 均值（与原脚本一致）
      mesr_keep_v2(file): 先构造 denoised stream，再在 denoised 上切 packet 算 ESR 均值
      rr(file): N_keep / N_raw（文件级）
    返回：packets_raw, packets_keep_v2, mesr_raw, mesr_keep_v2, rr
    """
    km_len = int(km_1d.shape[0])

    # 1) raw mesr（保持一致）
    packets_raw, mesr_raw = _mesr_raw_from_aedat4(aedat4_path, km_len=km_len)

    # 2) 读取 raw（一次性读出 x,y 用于构造 denoised stream）
    #    注：这一步是 v2 的核心开销，但你说“先验证口径”，这里不做流式优化。
    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    def on_packet_collect_xy(events_packet, info: PacketInfo):
        if info.raw_end > km_len:
            raise _StopSlicing()
        x, y = _eventstore_to_xy(events_packet)
        x_all.append(x)
        y_all.append(y)

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet_collect_xy,
        max_packets=None,
    )

    if not x_all:
        return 0, 0, 0.0, 0.0, 0.0

    x_raw = np.concatenate(x_all, axis=0)
    y_raw = np.concatenate(y_all, axis=0)

    # raw 数量以“对齐到 keepmask 的有效长度”为准（与原脚本 stop slicing 行为一致）
    n_raw = int(x_raw.size)

    # 3) 构造 denoised stream
    km_use = km_1d[:n_raw].astype(bool, copy=False)
    x_keep = x_raw[km_use]
    y_keep = y_raw[km_use]
    n_keep = int(x_keep.size)

    rr = float(n_keep) / float(n_raw) if n_raw > 0 else 0.0

    # 4) 在 denoised stream 上重新切包计算 mesr_keep_v2
    packets_keep, mesr_keep = _mesr_keep_v2_from_inmemory_denoised(x_keep, y_keep)

    return packets_raw, packets_keep, float(mesr_raw), float(mesr_keep), float(rr)


def main():
    data_root = _resolve(DATA_ROOT_BASE)
    keep_dir = _resolve(KEEP_DIR)
    out_csv = _resolve(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "split", "nd", "scene", "file",
        "packets_raw", "packets_keep_v2",
        "mesr_raw_file", "mesr_keep_v2_file",
        "rr_file",  # N_keep / N_raw
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

            for aedat4_path in files:
                scene = aedat4_path.parent.name
                fname = aedat4_path.name

                km_path = _find_keepmask_npz(aedat4_path, keep_dir, split=split, nd=nd)
                if km_path is None:
                    print(f"  [MISS] keepmask not found: {fname}")
                    continue

                km = _load_keepmask_1d(km_path)

                packets_raw, packets_keep, mesr_raw, mesr_keep, rr = compute_file_mesr_rr_v2(aedat4_path, km)

                rows.append([
                    split, nd, scene, fname,
                    packets_raw, packets_keep,
                    f"{mesr_raw:.6f}", f"{mesr_keep:.6f}",
                    f"{rr:.6f}",
                ])

                sub_files += 1
                sub_sum_raw += mesr_raw
                sub_sum_keep += mesr_keep
                sub_sum_rr += rr

            if sub_files > 0:
                sub_mesr_raw = sub_sum_raw / sub_files
                sub_mesr_keep = sub_sum_keep / sub_files
                sub_rr = sub_sum_rr / sub_files
            else:
                sub_mesr_raw = sub_mesr_keep = sub_rr = 0.0

            rows.append([
                split, nd, "TOTAL_FILES_MEAN", "TOTAL_FILES_MEAN",
                sub_files, sub_files,  # packets 列这里对 TOTAL 没意义，用 files 填充便于读
                f"{sub_mesr_raw:.6f}", f"{sub_mesr_keep:.6f}",
                f"{sub_rr:.6f}",
            ])

            print(f"[SUMMARY] {split}/{nd} files={sub_files} mesr_raw={sub_mesr_raw:.6f} mesr_keep_v2={sub_mesr_keep:.6f} rr={sub_rr:.6f}")

            all_files += sub_files
            all_sum_raw += sub_sum_raw
            all_sum_keep += sub_sum_keep
            all_sum_rr += sub_sum_rr

    if all_files > 0:
        all_mesr_raw = all_sum_raw / all_files
        all_mesr_keep = all_sum_keep / all_files
        all_rr = all_sum_rr / all_files
    else:
        all_mesr_raw = all_mesr_keep = all_rr = 0.0

    rows.append([
        "ALL", "ALL", "TOTAL_ALL_FILES_MEAN", "TOTAL_ALL_FILES_MEAN",
        all_files, all_files,
        f"{all_mesr_raw:.6f}", f"{all_mesr_keep:.6f}",
        f"{all_rr:.6f}",
    ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"[SAVED] {out_csv}")
    print(f"[TOTAL] files={all_files} mesr_raw={all_mesr_raw:.6f} mesr_keep_v2={all_mesr_keep:.6f} rr={all_rr:.6f}")


if __name__ == "__main__":
    main()
