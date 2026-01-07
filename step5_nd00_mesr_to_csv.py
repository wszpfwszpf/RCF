from __future__ import annotations

import csv
import re
from pathlib import Path
import numpy as np

# 你已有的两份代码（按你项目实际包名调整 import 路径）
from rcf_fast.packet_slice import run_count_slicer, PacketInfo, _StopSlicing
import rcf_fast.esr_core as esr_core


# =========================
# Config
# =========================
SPLIT = "day"                 # 只跑 day（你已确认）
TARGET_ND = "ND00"            # 配置：目前只生成了 ND00 keepmask
N_PER_PACKET = 30000          # 你已有切分
SENSOR_W = 346
SENSOR_H = 260

EMLB_ROOT = Path("data/emlb")                        # 原始 aedat4 根目录
KM_DIR = Path("data/emlb_rcfv2_verify/keepmask")       # keepmask npz 目录
OUT_DIR = Path("outputs/step5_mesr_csv_v2")             # 输出目录
OUT_CSV = OUT_DIR / f"mesr_v1_{SPLIT}_{TARGET_ND.lower()}_6eta.csv"

# keepmask 语义：1=keep（你已确认）
KEEP_IS_ONE = True

# 全局汇总方式：True=按 packet 数加权（推荐）；False=按文件均值
WEIGHT_BY_PACKETS = False


# =========================
# Helpers
# =========================
def resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()

def extract_nd(stem: str) -> str | None:
    m = re.search(r"(ND\d{2})", stem)
    return m.group(1) if m else None

def derive_km_path(aedat4_path: Path, km_dir: Path) -> Path:
    stem = aedat4_path.stem  # e.g., Architecture-ND00-1
    nd = extract_nd(stem)
    if nd is None:
        raise ValueError(f"Cannot find NDxx in filename: {stem}")
    return km_dir / f"{stem}_{nd.lower()}_keepmask.npz"

def load_km(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "eta_list" not in data or "keepmask" not in data:
        raise KeyError(f"npz missing keys, got {list(data.keys())}")
    eta_list = np.asarray(data["eta_list"], dtype=np.float32)
    km = np.asarray(data["keepmask"])

    # 如果是 object list（按 bin 存），拼接成事件级
    if km.ndim == 1 and km.dtype == object:
        km = np.concatenate([np.asarray(x) for x in km], axis=0)

    return eta_list, km  # km expected (N,6) uint8/0-1

def eventstore_to_xy(events) -> tuple[np.ndarray, np.ndarray]:
    """
    尽量快地从 dv_processing EventStore 拿到 x,y。
    不同 dv 版本 API 不一致，所以做 best-effort。
    """
    for fn in ("numpy", "toNumpy", "asNumpy"):
        if hasattr(events, fn):
            arr = getattr(events, fn)()
            if isinstance(arr, np.ndarray) and arr.dtype.names:
                names = set(arr.dtype.names)
                if "x" in names and "y" in names:
                    return arr["x"].astype(np.int32, copy=False), arr["y"].astype(np.int32, copy=False)

    # fallback：迭代（慢，但稳）
    xs, ys = [], []
    for e in events:
        x = getattr(e, "x", None); y = getattr(e, "y", None)
        x = x() if callable(x) else x
        y = y() if callable(y) else y
        xs.append(int(x)); ys.append(int(y))
    return np.asarray(xs, np.int32), np.asarray(ys, np.int32)

def esr_v1_from_xy(x: np.ndarray, y: np.ndarray) -> float:
    N = int(x.size)
    if N < 2:
        return 0.0
    idx = y.astype(np.int64) * SENSOR_W + x.astype(np.int64)
    counts = np.bincount(idx, minlength=SENSOR_W * SENSOR_H).reshape(SENSOR_H, SENSOR_W)
    return float(esr_core._esr_v1_from_count_surface(counts, N=N, W=SENSOR_W, H=SENSOR_H))


# =========================
# Per-file computation
# =========================
def compute_file_mesr(aedat4_path: Path, km: np.ndarray):
    """
    返回：
      packet_count
      mesr_raw
      mesr_eta(6,)
    """
    if km.ndim != 2 or km.shape[1] != 6:
        raise AssertionError(f"keepmask must be (N,6), got {km.shape}")
    km_len = int(km.shape[0])

    esr_raw_list: list[float] = []
    esr_eta_lists: list[list[float]] = [[] for _ in range(6)]

    def on_packet(events_packet, info: PacketInfo):
        # 只处理 keepmask 覆盖范围内的完整 packet
        if info.raw_end > km_len:
            raise _StopSlicing()

        x, y = eventstore_to_xy(events_packet)

        # raw ESR
        esr_raw_list.append(esr_v1_from_xy(x, y))

        km_slice = km[info.raw_begin:info.raw_end, :]  # (n,6)
        if not KEEP_IS_ONE:
            km_slice = 1 - km_slice

        # 6 路 η：按列过滤事件
        for j in range(6):
            m = km_slice[:, j].astype(bool, copy=False)
            esr_eta_lists[j].append(esr_v1_from_xy(x[m], y[m]))

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
        max_packets=None,
    )

    esr_raw_arr = np.asarray(esr_raw_list, dtype=np.float32)
    esr_eta_arrs = [np.asarray(v, dtype=np.float32) for v in esr_eta_lists]

    packet_count = int(esr_raw_arr.size)
    mesr_raw = float(esr_raw_arr.mean()) if packet_count > 0 else 0.0
    mesr_eta = np.asarray([float(a.mean()) if a.size > 0 else 0.0 for a in esr_eta_arrs], dtype=np.float32)

    return packet_count, mesr_raw, mesr_eta


# =========================
# Batch + CSV
# =========================
def main():
    emlb_root = resolve_path(EMLB_ROOT)
    km_dir = resolve_path(KM_DIR)
    out_csv = resolve_path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    split_dir = emlb_root / SPLIT
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    if not km_dir.exists():
        raise FileNotFoundError(f"KM dir not found: {km_dir}")

    # 只跑 ND00
    aedat4_files = sorted(split_dir.rglob("*.aedat4"))
    aedat4_files = [p for p in aedat4_files if extract_nd(p.stem) == TARGET_ND]

    if len(aedat4_files) == 0:
        raise RuntimeError(f"No {TARGET_ND} files found under {split_dir}")

    # 全局累计（用于“总 MESR”）
    total_packets = 0
    sum_esr_raw = 0.0
    sum_esr_eta = np.zeros((6,), dtype=np.float64)

    # 如果你选择“按文件均值”，则记录文件级 mesr
    file_mesr_raw = []
    file_mesr_eta = []

    # 写 CSV
    header = [
        "split", "nd", "scene", "file",
        "packets",
        "eta0", "eta1", "eta2", "eta3", "eta4", "eta5",
        "mesr_raw",
        "mesr_eta0", "mesr_eta1", "mesr_eta2", "mesr_eta3", "mesr_eta4", "mesr_eta5",
    ]

    # 先拿一份 eta_list 作为列输出（每文件一份更稳，因为 npz 里存了）
    rows = []

    print(f"[RUN] split={SPLIT} nd={TARGET_ND} files={len(aedat4_files)}")

    for i, aedat4_path in enumerate(aedat4_files, 1):
        scene = aedat4_path.parent.name
        km_path = derive_km_path(aedat4_path, km_dir)
        if not km_path.exists():
            print(f"[MISS] km not found: {km_path.name} for {aedat4_path.name}")
            continue

        eta_list, km = load_km(km_path)

        packets, mesr_raw, mesr_eta = compute_file_mesr(aedat4_path, km)

        # 全局统计：按 packet 加权
        total_packets += packets
        sum_esr_raw += mesr_raw * packets
        sum_esr_eta += mesr_eta.astype(np.float64) * packets

        file_mesr_raw.append(mesr_raw)
        file_mesr_eta.append(mesr_eta)

        row = [
            SPLIT, TARGET_ND, scene, aedat4_path.name,
            packets,
            float(eta_list[0]), float(eta_list[1]), float(eta_list[2]),
            float(eta_list[3]), float(eta_list[4]), float(eta_list[5]),
            float(mesr_raw),
            float(mesr_eta[0]), float(mesr_eta[1]), float(mesr_eta[2]),
            float(mesr_eta[3]), float(mesr_eta[4]), float(mesr_eta[5]),
        ]
        rows.append(row)

        if i % 10 == 0:
            print(f"...progress {i}/{len(aedat4_files)} | last={aedat4_path.name} packets={packets}")

    # 计算“6 个总 MESR”
    if WEIGHT_BY_PACKETS:
        total_mesr_raw = (sum_esr_raw / total_packets) if total_packets > 0 else 0.0
        total_mesr_eta = (sum_esr_eta / total_packets) if total_packets > 0 else np.zeros((6,), dtype=np.float64)
        summary_tag = "TOTAL_WEIGHTED_BY_PACKETS"
    else:
        total_mesr_raw = float(np.mean(file_mesr_raw)) if len(file_mesr_raw) > 0 else 0.0
        total_mesr_eta = np.mean(np.stack(file_mesr_eta, axis=0), axis=0) if len(file_mesr_eta) > 0 else np.zeros((6,), dtype=np.float64)
        summary_tag = "TOTAL_MEAN_OVER_FILES"

    # 写入 CSV（最后加一行 summary）
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

        # summary 行：scene/file 用 tag 占位，eta_list 留空 or 用 -1
        w.writerow([
            SPLIT, TARGET_ND, summary_tag, summary_tag,
            total_packets,
            "", "", "", "", "", "",
            float(total_mesr_raw),
            float(total_mesr_eta[0]), float(total_mesr_eta[1]), float(total_mesr_eta[2]),
            float(total_mesr_eta[3]), float(total_mesr_eta[4]), float(total_mesr_eta[5]),
        ])

    print(f"[SAVED] {out_csv}")
    print(f"[SUMMARY] packets={total_packets} mesr_raw={total_mesr_raw:.6f} mesr_eta={np.asarray(total_mesr_eta).round(6)}")


if __name__ == "__main__":
    main()
