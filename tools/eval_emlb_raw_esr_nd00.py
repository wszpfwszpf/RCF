# tools/eval_emlb_raw_esr_nd00.py
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rcf_fast.esr_core import (
    SENSOR_W,
    SENSOR_H,
    DEFAULT_N_PACKET,
    compute_mesr_v1_from_aedat4_count_slicing,
)

# =============================================================================
# PyCharm one-click config
# =============================================================================
EMLB_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb"
SUBSET = "dya"  # "day" or "night"
ND_FILTER = "ND00"  # e.g. ND00 / ND16 / ND64

N_PER_PACKET = DEFAULT_N_PACKET
RESOLUTION = (SENSOR_W, SENSOR_H)

# Debug: set 10 for quick check; set 0 for full run
MAX_PACKETS_PER_FILE = 0

# Output directory
OUT_DIR = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb_esr"


@dataclass
class Row:
    scene: str
    file: str
    n_packets: int
    mesr: float
    std: float
    error: str


def _find_nd_files(root: Path, nd: str) -> List[Path]:
    nd_upper = nd.upper()
    files = sorted(root.rglob("*.aedat4"))
    return [p for p in files if nd_upper in p.name.upper()]


def _scene_name_from_path(subset_root: Path, file_path: Path) -> str:
    # subset_root/SceneName/xxx.aedat4
    try:
        rel = file_path.relative_to(subset_root)
        parts = rel.parts
        return parts[0] if len(parts) >= 2 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _is_finite(x: float) -> bool:
    return isinstance(x, (float, int)) and math.isfinite(float(x))


def main() -> None:
    emlb_root = Path(EMLB_ROOT)
    subset_root = emlb_root / SUBSET
    if not subset_root.exists():
        raise FileNotFoundError(f"Subset root not found: {subset_root}")

    nd_tag = ND_FILTER.lower()
    subset_tag = SUBSET.lower()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"raw_mesr_{subset_tag}_{nd_tag}_v1.csv"
    out_scene = out_dir / f"raw_mesr_{subset_tag}_{nd_tag}_v1_scene_summary.csv"

    files = _find_nd_files(subset_root, ND_FILTER)

    print("=" * 110)
    print("[RAW-MESR] E-MLB raw MESR evaluation (ESR V1, count slicing)")
    print(f"[RAW-MESR] Data root  : {subset_root}")
    print(f"[RAW-MESR] ND filter  : {ND_FILTER}")
    print(f"[RAW-MESR] N/packet   : {N_PER_PACKET}")
    print(f"[RAW-MESR] Resolution : {RESOLUTION}")
    if MAX_PACKETS_PER_FILE > 0:
        print(f"[RAW-MESR] Max packets/file (debug): {MAX_PACKETS_PER_FILE}")
    print(f"[RAW-MESR] Files      : {len(files)}")
    print(f"[RAW-MESR] Out CSV    : {out_csv}")
    print(f"[RAW-MESR] Out Scene  : {out_scene}")
    print("=" * 110)

    rows: List[Row] = []

    # For scene summary: aggregate only successful MESR values
    per_scene_vals: Dict[str, List[float]] = {}
    per_scene_packets: Dict[str, int] = {}
    per_scene_failed: Dict[str, int] = {}

    for idx, p in enumerate(files, 1):
        scene = _scene_name_from_path(subset_root, p)
        print(f"[{idx:04d}/{len(files):04d}] {scene:>14s} | {p.name}")

        try:
            stats, _ = compute_mesr_v1_from_aedat4_count_slicing(
                str(p),
                n_per_packet=N_PER_PACKET,
                resolution=RESOLUTION,
                drop_tail=True,
                max_packets=(MAX_PACKETS_PER_FILE if MAX_PACKETS_PER_FILE > 0 else None),
                allow_out_of_order=True,          # critical: skip out-of-order batches
                verbose_out_of_order=True,
            )

            mesr = float(stats.mean_esr)
            std = float(stats.std_esr)
            n_packets = int(stats.n_packets)

            rows.append(Row(scene=scene, file=p.name, n_packets=n_packets, mesr=mesr, std=std, error=""))

            # Aggregate only if valid
            if _is_finite(mesr) and n_packets > 0:
                per_scene_vals.setdefault(scene, []).append(mesr)
                per_scene_packets[scene] = per_scene_packets.get(scene, 0) + n_packets
            else:
                per_scene_failed[scene] = per_scene_failed.get(scene, 0) + 1

        except Exception as e:
            err = str(e)
            print(f"[ERROR] {p.name} | {err}")
            rows.append(Row(scene=scene, file=p.name, n_packets=0, mesr=float("nan"), std=float("nan"), error=err))
            per_scene_failed[scene] = per_scene_failed.get(scene, 0) + 1
            continue

    # Write per-file CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "file", "n_packets", "mesr_raw_v1", "std_raw_v1", "error"])
        for r in rows:
            mesr_str = f"{r.mesr:.6f}" if _is_finite(r.mesr) else "nan"
            std_str = f"{r.std:.6f}" if _is_finite(r.std) else "nan"
            w.writerow([r.scene, r.file, r.n_packets, mesr_str, std_str, r.error])

    # Scene summary CSV
    with out_scene.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "n_files_ok", "n_files_failed", "sum_packets_ok", "mean_mesr_raw_v1", "std_over_files_raw_v1"])

        all_vals: List[float] = []
        for scene in sorted(set(list(per_scene_vals.keys()) + list(per_scene_failed.keys()))):
            vals = per_scene_vals.get(scene, [])
            n_failed = per_scene_failed.get(scene, 0)
            sum_packets = per_scene_packets.get(scene, 0)

            if len(vals) > 0:
                mean_scene = sum(vals) / len(vals)
                m = mean_scene
                var = sum((v - m) ** 2 for v in vals) / len(vals)
                std_scene = var ** 0.5
                all_vals.extend(vals)
            else:
                mean_scene = float("nan")
                std_scene = float("nan")

            w.writerow([scene, len(vals), n_failed, sum_packets,
                        f"{mean_scene:.6f}" if _is_finite(mean_scene) else "nan",
                        f"{std_scene:.6f}" if _is_finite(std_scene) else "nan"])

        # Overall across scenes/files (file-level MESR)
        if len(all_vals) > 0:
            overall_mean = sum(all_vals) / len(all_vals)
            m = overall_mean
            overall_var = sum((v - m) ** 2 for v in all_vals) / len(all_vals)
            overall_std = overall_var ** 0.5
        else:
            overall_mean = float("nan")
            overall_std = float("nan")

        w.writerow([])
        w.writerow(["OVERALL(file-level)", len(all_vals), "", "",
                    f"{overall_mean:.6f}" if _is_finite(overall_mean) else "nan",
                    f"{overall_std:.6f}" if _is_finite(overall_std) else "nan"])

    print("-" * 110)
    print(f"[RAW-MESR] Saved per-file CSV  : {out_csv}")
    print(f"[RAW-MESR] Saved scene summary : {out_scene}")
    print("-" * 110)
    print("[RAW-MESR] Done.")


if __name__ == "__main__":
    main()
