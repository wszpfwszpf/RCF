# tools/step4_dwf_mesr_rr_all8_v2.py
# 日期：2026-01-08
# 中文说明：
# 在 E-MLB 的 8 个子集（day/night × ND00/04/16/64）上统计 DWF 的 ESR 与 RR（v2口径）。
#
# 与 step4_dwf_mesr_rr_all8.py 的差异：
# - raw：不变，raw 流按固定事件数切 packet，packet 级 ESR(v1)，文件 mesr_raw = mean(packet_esr)
# - keep(v2)：先用 keepmask 从整条 raw 流构造 denoised stream（仅保留事件），
#            再在 denoised stream 上按固定事件数切 packet，packet 级 ESR(v1)，文件 mesr_keep_v2 = mean(packet_esr)
# - rr：改为文件级 rr = N_keep / N_raw（不再按 packet 平均）
#
# 输出控制：
# - 不打印任何文件级或中间进度信息
# - 仅输出每个子集的 [SUMMARY]
#
# 前置条件：
# - 已通过 step4_run_dwf_* 生成 keepmask（单列 0/1，与原始事件顺序对齐）

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rcf_fast.packet_slice import run_count_slicer, PacketInfo, _StopSlicing
import rcf_fast.esr_core as esr_core


# =========================
# 用户配置
# =========================
DATA_ROOT_BASE = Path(r"../data/emlb")
KEEP_DIR = Path(r"../data/emlb_dwf_verify/keepmask")   # ← DWF keepmask 目录
OUT_DIR = Path(r"../baselines/step4_dwf_mesr_rr_all8_v2")
OUT_CSV = OUT_DIR / "dwf_mesr_rr_all8_v2_filemean.csv"

SENSOR_W, SENSOR_H = 346, 260
N_PER_PACKET = 30000
KEEP_IS_ONE = True

SPLITS = ["day", "night"]
NDS = ["ND00", "ND04", "ND16", "ND64"]
# =========================


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (Path(__file__).resolve().parent / p).resolve()


def _extract_nd(stem: str) -> Optional[str]:
    m = re.search(r"(ND\d{2})", stem)
    return m.group(1) if m else None


def _collect_aedat4_files(split_dir: Path, nd: str) -> List[Path]:
    return [p for p in split_dir.rglob("*.aedat4") if _extract_nd(p.stem) == nd]


def _find_keepmask_npz(aedat4_path: Path, keep_dir: Path, split: str, nd: str) -> Optional[Path]:
    stem = aedat4_path.stem
    cands = [
        keep_dir / f"{stem}_{split}_{nd.lower()}_keepmask.npz",
        keep_dir / f"{stem}_{nd.lower()}_keepmask.npz",
    ]
    for c in cands:
        if c.exists():
            return c
    hits = sorted(keep_dir.glob(f"{stem}_*{nd.lower()}*keepmask*.npz"))
    return hits[0] if hits else None


def _load_keepmask_1d(npz_path: Path) -> np.ndarray:
    km = np.load(npz_path, allow_pickle=True)["keepmask"]
    if km.ndim == 1 and km.dtype == object:
        km = np.concatenate([np.asarray(x) for x in km], axis=0)
    if km.ndim == 2 and km.shape[1] == 1:
        km = km[:, 0]
    km = km.astype(np.uint8, copy=False)
    if not KEEP_IS_ONE:
        km = (1 - km).astype(np.uint8, copy=False)
    return km


def _eventstore_to_xy(events) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(events, "numpy"):
        arr = events.numpy()
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            return arr["x"].astype(np.int32, copy=False), arr["y"].astype(np.int32, copy=False)

    xs, ys = [], []
    for e in events:
        xs.append(int(e.x()))
        ys.append(int(e.y()))
    return np.asarray(xs, np.int32), np.asarray(ys, np.int32)


def _esr_v1_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    N = x.size
    if N < 2:
        return 0.0
    idx = y.astype(np.int64) * SENSOR_W + x.astype(np.int64)
    cs = np.bincount(idx, minlength=SENSOR_W * SENSOR_H).reshape(SENSOR_H, SENSOR_W)
    return float(esr_core._esr_v1_from_count_surface(cs, N=N, W=SENSOR_W, H=SENSOR_H))


def _mesr_raw_from_aedat4(aedat4_path: Path, km_len: int) -> float:
    """
    raw：raw 流按 N_PER_PACKET 切 packet，packet 级 ESR 均值。
    """
    esr_list: List[float] = []

    def on_packet(events, info: PacketInfo):
        if info.raw_end > km_len:
            raise _StopSlicing()
        x, y = _eventstore_to_xy(events)
        esr_list.append(_esr_v1_from_xy(x, y))

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
    )
    return float(np.mean(esr_list)) if esr_list else 0.0


def _collect_xy_raw_aligned(aedat4_path: Path, km_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 raw 并对齐到 keepmask 可覆盖范围（与 raw mesr 的 stop slicing 行为一致）。
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    def on_packet(events, info: PacketInfo):
        if info.raw_end > km_len:
            raise _StopSlicing()
        x, y = _eventstore_to_xy(events)
        xs.append(x)
        ys.append(y)

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
    )

    if not xs:
        return np.empty((0,), np.int32), np.empty((0,), np.int32)

    x_raw = np.concatenate(xs, axis=0)
    y_raw = np.concatenate(ys, axis=0)
    return x_raw, y_raw


def _mesr_keep_v2_from_denoised_stream(x_keep: np.ndarray, y_keep: np.ndarray) -> float:
    """
    keep(v2)：在 denoised stream 上按 N_PER_PACKET 重新切包，packet 级 ESR 均值。
    """
    N = int(x_keep.size)
    if N <= 0:
        return 0.0

    esr_list: List[float] = []
    for s in range(0, N, N_PER_PACKET):
        e = min(s + N_PER_PACKET, N)
        esr_list.append(_esr_v1_from_xy(x_keep[s:e], y_keep[s:e]))

    return float(np.mean(esr_list)) if esr_list else 0.0


def compute_file_mesr_rr_v2(aedat4_path: Path, km_1d: np.ndarray) -> Tuple[float, float, float]:
    """
    文件级：
      mesr_raw：raw 流切包 ESR 均值
      mesr_keep_v2：denoised stream 切包 ESR 均值
      rr_file：N_keep / N_raw（文件级）
    """
    km_len = int(km_1d.size)

    # raw mesr
    mesr_raw = _mesr_raw_from_aedat4(aedat4_path, km_len=km_len)

    # collect raw xy aligned to km coverage
    x_raw, y_raw = _collect_xy_raw_aligned(aedat4_path, km_len=km_len)
    n_raw = int(x_raw.size)
    if n_raw <= 0:
        return 0.0, 0.0, 0.0

    km_use = km_1d[:n_raw].astype(bool, copy=False)
    x_keep = x_raw[km_use]
    y_keep = y_raw[km_use]
    n_keep = int(x_keep.size)

    rr_file = float(n_keep) / float(n_raw) if n_raw > 0 else 0.0
    mesr_keep_v2 = _mesr_keep_v2_from_denoised_stream(x_keep, y_keep)

    return float(mesr_raw), float(mesr_keep_v2), float(rr_file)


def main():
    data_root = _resolve(DATA_ROOT_BASE)
    keep_dir = _resolve(KEEP_DIR)
    out_csv = _resolve(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "nd", "files", "mesr_raw", "mesr_keep_v2", "rr_file"])

        for split in SPLITS:
            for nd in NDS:
                files = _collect_aedat4_files(data_root / split, nd)

                sum_raw = 0.0
                sum_keep = 0.0
                sum_rr = 0.0
                n_files = 0

                for aed in files:
                    km_path = _find_keepmask_npz(aed, keep_dir, split, nd)
                    if km_path is None:
                        continue
                    km = _load_keepmask_1d(km_path)

                    mr, mk, rr = compute_file_mesr_rr_v2(aed, km)
                    sum_raw += mr
                    sum_keep += mk
                    sum_rr += rr
                    n_files += 1

                if n_files > 0:
                    mesr_raw = sum_raw / n_files
                    mesr_keep_v2 = sum_keep / n_files
                    rr_mean = sum_rr / n_files
                else:
                    mesr_raw = mesr_keep_v2 = rr_mean = 0.0

                writer.writerow([split, nd, n_files,
                                 f"{mesr_raw:.6f}", f"{mesr_keep_v2:.6f}", f"{rr_mean:.6f}"])

                print(f"[SUMMARY] {split}/{nd} files={n_files} "
                      f"mesr_raw={mesr_raw:.6f} mesr_keep_v2={mesr_keep_v2:.6f} rr={rr_mean:.6f}")

    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
