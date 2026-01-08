# tools/step4_knoise_mesr_rr_all8.py
# 日期：2026-01-08
# 中文说明：
# 在 E-MLB 的 8 个子集（day/night × ND00/04/16/64）上统计 KNoise 的 ESR 与 RR。
# 统计口径与 step4_baf_mesr_rr_all8.py 完全一致：
# - 按固定事件数切 packet
# - packet 级 ESR(v1)
# - 文件级 mesr = mean(packet_esr)
# - 子集级 mesr = sum(file_mesr) / 文件数
# - rr 同样按文件平均
#
# 输出控制：
# - 不打印任何文件级 / 中间过程信息
# - 仅输出每个子集的 [SUMMARY]
#
# 前置条件：
# - 已通过 step4_run_knoise_* 生成 keepmask（单列 0/1，与原始事件顺序对齐）

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
KEEP_DIR = Path(r"../data/emlb_knoise_verify/keepmask")   # ← KNoise keepmask 目录
OUT_DIR = Path(r"../baselines/step4_knoise_mesr_rr_all8")
OUT_CSV = OUT_DIR / "knoise_mesr_rr_all8_filemean.csv"

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


def compute_file_mesr_rr(aedat4_path: Path, km_1d: np.ndarray) -> Tuple[float, float, float]:
    km_len = km_1d.size
    esr_raw, esr_keep, rr = [], [], []

    def on_packet(events, info: PacketInfo):
        if info.raw_end > km_len:
            raise _StopSlicing()

        x, y = _eventstore_to_xy(events)
        m = km_1d[info.raw_begin:info.raw_end].astype(bool, copy=False)

        rr.append(m.mean() if m.size > 0 else 0.0)
        esr_raw.append(_esr_v1_from_xy(x, y))
        esr_keep.append(_esr_v1_from_xy(x[m], y[m]))

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
    )

    if not esr_raw:
        return 0.0, 0.0, 0.0

    return float(np.mean(esr_raw)), float(np.mean(esr_keep)), float(np.mean(rr))


def main():
    data_root = _resolve(DATA_ROOT_BASE)
    keep_dir = _resolve(KEEP_DIR)
    out_csv = _resolve(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "nd", "files", "mesr_raw", "mesr_keep", "rr"])

        for split in SPLITS:
            for nd in NDS:
                files = _collect_aedat4_files(data_root / split, nd)

                sum_raw = sum_keep = sum_rr = 0.0
                n_files = 0

                for aed in files:
                    km_path = _find_keepmask_npz(aed, keep_dir, split, nd)
                    if km_path is None:
                        continue
                    km = _load_keepmask_1d(km_path)
                    mr, mk, r = compute_file_mesr_rr(aed, km)

                    sum_raw += mr
                    sum_keep += mk
                    sum_rr += r
                    n_files += 1

                if n_files > 0:
                    mesr_raw = sum_raw / n_files
                    mesr_keep = sum_keep / n_files
                    rr = sum_rr / n_files
                else:
                    mesr_raw = mesr_keep = rr = 0.0

                writer.writerow([split, nd, n_files,
                                 f"{mesr_raw:.6f}", f"{mesr_keep:.6f}", f"{rr:.6f}"])

                print(f"[SUMMARY] {split}/{nd} files={n_files} "
                      f"mesr_raw={mesr_raw:.6f} mesr_keep={mesr_keep:.6f} rr={rr:.6f}")

    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
