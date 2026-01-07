from __future__ import annotations

import csv
import re
from pathlib import Path
import numpy as np

from rcf_fast.packet_slice import run_count_slicer, PacketInfo, _StopSlicing
import rcf_fast.esr_core as esr_core


# =========================
# Config
# =========================
SPLIT = "night"
TARGET_ND = "ND64"  # ND00 / ND04 / ND16 / ND64
N_PER_PACKET = 30000
SENSOR_W = 346
SENSOR_H = 260

EMLB_ROOT = Path("data/emlb")
KM_DIR = Path("data/emlb_rcfv2_verify/keepmask")
OUT_DIR = Path("outputs/step5_comparev2_csv")
OUT_CSV = OUT_DIR / f"compare_mesr_v1_{SPLIT}_{TARGET_ND.lower()}_rcf_vs_random.csv"

KEEP_IS_ONE = True
RNG_SEED = 12345  # 固定种子，保证可复现


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
    stem = aedat4_path.stem
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
    # object list -> concat
    if km.ndim == 1 and km.dtype == object:
        km = np.concatenate([np.asarray(x) for x in km], axis=0)
    return eta_list, km

def eventstore_to_xy(events) -> tuple[np.ndarray, np.ndarray]:
    for fn in ("numpy", "toNumpy", "asNumpy"):
        if hasattr(events, fn):
            arr = getattr(events, fn)()
            if isinstance(arr, np.ndarray) and arr.dtype.names:
                names = set(arr.dtype.names)
                if "x" in names and "y" in names:
                    return arr["x"].astype(np.int32, copy=False), arr["y"].astype(np.int32, copy=False)

    # fallback (slow)
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

def keep_ratio_from_km(km: np.ndarray) -> np.ndarray:
    # km expected (N,6) 0/1
    km01 = km if KEEP_IS_ONE else (1 - km)
    return km01.mean(axis=0).astype(np.float32)


# =========================
# Per-file computation
# =========================
def compute_file_compare(aedat4_path: Path, km: np.ndarray, rng: np.random.Generator):
    """
    返回：
      packets
      mesr_raw
      mesr_rcf(6,)
      mesr_rand(6,)
      keep_ratio(6,)
      mean_keep_per_packet(6,)
    """
    if km.ndim != 2 or km.shape[1] != 6:
        raise AssertionError(f"keepmask must be (N,6), got {km.shape}")
    km_len = int(km.shape[0])

    # per-packet ESR lists
    esr_raw_list: list[float] = []
    esr_rcf_lists: list[list[float]] = [[] for _ in range(6)]
    esr_rand_lists: list[list[float]] = [[] for _ in range(6)]
    keep_counts_sum = np.zeros((6,), dtype=np.int64)
    packet_count = 0

    def on_packet(events_packet, info: PacketInfo):
        nonlocal packet_count
        if info.raw_end > km_len:
            raise _StopSlicing()

        x, y = eventstore_to_xy(events_packet)
        n = int(x.size)

        km_slice = km[info.raw_begin:info.raw_end, :]
        if not KEEP_IS_ONE:
            km_slice = 1 - km_slice
        km_slice = km_slice.astype(bool, copy=False)

        # raw ESR
        esr_raw_list.append(esr_v1_from_xy(x, y))

        # rcf ESR + random same-N ESR
        for j in range(6):
            m = km_slice[:, j]
            n_keep = int(m.sum())
            keep_counts_sum[j] += n_keep

            # rcf
            esr_rcf_lists[j].append(esr_v1_from_xy(x[m], y[m]))

            # random baseline: sample n_keep indices uniformly from [0, n)
            if n_keep <= 1:
                esr_rand_lists[j].append(0.0)
            elif n_keep >= n:
                # same as raw
                esr_rand_lists[j].append(esr_raw_list[-1])
            else:
                idx = rng.choice(n, size=n_keep, replace=False)
                esr_rand_lists[j].append(esr_v1_from_xy(x[idx], y[idx]))

        packet_count += 1

    run_count_slicer(
        aedat4_path=str(aedat4_path),
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
        max_packets=None,
    )

    # aggregate
    packets = packet_count
    esr_raw_arr = np.asarray(esr_raw_list, dtype=np.float32)
    mesr_raw = float(esr_raw_arr.mean()) if packets > 0 else 0.0

    mesr_rcf = np.asarray([float(np.mean(v)) if len(v) > 0 else 0.0 for v in esr_rcf_lists], dtype=np.float32)
    mesr_rand = np.asarray([float(np.mean(v)) if len(v) > 0 else 0.0 for v in esr_rand_lists], dtype=np.float32)

    keep_ratio = keep_ratio_from_km(km)
    mean_keep_per_packet = (keep_counts_sum / max(packets, 1)).astype(np.float32)

    return packets, mesr_raw, mesr_rcf, mesr_rand, keep_ratio, mean_keep_per_packet


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
        raise FileNotFoundError(split_dir)
    if not km_dir.exists():
        raise FileNotFoundError(km_dir)

    aedat4_files = sorted(split_dir.rglob("*.aedat4"))
    aedat4_files = [p for p in aedat4_files if extract_nd(p.stem) == TARGET_ND]
    if len(aedat4_files) == 0:
        raise RuntimeError(f"No {TARGET_ND} files found under {split_dir}")

    rng = np.random.default_rng(RNG_SEED)

    header = [
        "split", "nd", "scene", "file", "packets",
        "eta0","eta1","eta2","eta3","eta4","eta5",
        "keep_ratio0","keep_ratio1","keep_ratio2","keep_ratio3","keep_ratio4","keep_ratio5",
        "mean_keep_pkt0","mean_keep_pkt1","mean_keep_pkt2","mean_keep_pkt3","mean_keep_pkt4","mean_keep_pkt5",
        "mesr_raw",
        "mesr_rcf0","mesr_rcf1","mesr_rcf2","mesr_rcf3","mesr_rcf4","mesr_rcf5",
        "mesr_rand0","mesr_rand1","mesr_rand2","mesr_rand3","mesr_rand4","mesr_rand5",
        "delta0","delta1","delta2","delta3","delta4","delta5",
    ]

    rows = []

    # global summary (file-mean, matching你的 raw 口径)
    file_mesr_raw = []
    file_mesr_rcf = []
    file_mesr_rand = []

    print(f"[RUN] split={SPLIT} nd={TARGET_ND} files={len(aedat4_files)} seed={RNG_SEED}")

    for i, aedat4_path in enumerate(aedat4_files, 1):
        scene = aedat4_path.parent.name
        km_path = derive_km_path(aedat4_path, km_dir)
        if not km_path.exists():
            print(f"[MISS] km not found: {km_path.name} for {aedat4_path.name}")
            continue

        eta_list, km = load_km(km_path)
        packets, mesr_raw, mesr_rcf, mesr_rand, keep_ratio, mean_keep = compute_file_compare(aedat4_path, km, rng)
        delta = (mesr_rcf - mesr_rand).astype(np.float32)

        file_mesr_raw.append(mesr_raw)
        file_mesr_rcf.append(mesr_rcf)
        file_mesr_rand.append(mesr_rand)

        row = [
            SPLIT, TARGET_ND, scene, aedat4_path.name, packets,
            *[float(x) for x in eta_list],
            *[float(x) for x in keep_ratio],
            *[float(x) for x in mean_keep],
            float(mesr_raw),
            *[float(x) for x in mesr_rcf],
            *[float(x) for x in mesr_rand],
            *[float(x) for x in delta],
        ]
        rows.append(row)

        if i % 10 == 0:
            print(f"...progress {i}/{len(aedat4_files)} | last={aedat4_path.name}")

    # summary row: file-mean (与你 raw 统计口径一致)
    if len(file_mesr_raw) > 0:
        total_mesr_raw = float(np.mean(file_mesr_raw))
        total_mesr_rcf = np.mean(np.stack(file_mesr_rcf, axis=0), axis=0)
        total_mesr_rand = np.mean(np.stack(file_mesr_rand, axis=0), axis=0)
        total_delta = total_mesr_rcf - total_mesr_rand
    else:
        total_mesr_raw = 0.0
        total_mesr_rcf = np.zeros((6,), dtype=np.float32)
        total_mesr_rand = np.zeros((6,), dtype=np.float32)
        total_delta = np.zeros((6,), dtype=np.float32)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

        w.writerow([
            SPLIT, TARGET_ND, "TOTAL_MEAN_OVER_FILES", "TOTAL_MEAN_OVER_FILES", "",
            "", "", "", "", "", "",
            "", "", "", "", "", "",
            "", "", "", "", "", "",
            total_mesr_raw,
            *[float(x) for x in total_mesr_rcf],
            *[float(x) for x in total_mesr_rand],
            *[float(x) for x in total_delta],
        ])

    print(f"[SAVED] {out_csv}")
    print(f"[SUMMARY file-mean] mesr_raw={total_mesr_raw:.6f}")
    print(f"[SUMMARY file-mean] mesr_rcf={np.asarray(total_mesr_rcf).round(6)}")
    print(f"[SUMMARY file-mean] mesr_rand={np.asarray(total_mesr_rand).round(6)}")
    print(f"[SUMMARY file-mean] delta(rcf-rand)={np.asarray(total_delta).round(6)}")


if __name__ == "__main__":
    main()
