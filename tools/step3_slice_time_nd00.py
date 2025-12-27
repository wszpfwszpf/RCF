# -*- coding: utf-8 -*-
# tools/step3_slice_time_nd00.py
from __future__ import annotations

from pathlib import Path

from rcf_fast.timeinterval_slice import run_time_slicer


# -------------------------
# PyCharm one-click config
# -------------------------
# 改成你的实际根目录即可（你现在是 data/emlb/day）
EMLB_DAY_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\day"

ND_FILTER = "ND00"
DT_MS = 10
MAX_BINS = 50

# 是否尝试用 dv.TimeSurface（如果 binding 暴露了该类）
USE_DV_TIMESURFACE = False


def _find_first_nd_file(root: str, nd: str) -> Path:
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"EMLB root not found: {rootp}")

    # 搜索所有 .aedat4
    files = sorted(rootp.rglob("*.aedat4"))
    if not files:
        raise FileNotFoundError(f"No .aedat4 files found under: {rootp}")

    # 过滤 ND
    nd_upper = nd.upper()
    cand = [p for p in files if nd_upper in p.name.upper()]
    if not cand:
        raise FileNotFoundError(f"No files matching '{nd_upper}' under: {rootp}")
    return cand[0]


def main() -> None:
    path = _find_first_nd_file(EMLB_DAY_ROOT, ND_FILTER)

    print("=" * 100)
    print("[Step3] dv-processing read + strict time slicing")
    print(f"[Step3] File     : {path}")
    print(f"[Step3] ND       : {ND_FILTER}")
    print(f"[Step3] dt_ms    : {DT_MS}")
    print(f"[Step3] max_bins : {MAX_BINS}")
    print(f"[Step3] use_dv_timesurface : {USE_DV_TIMESURFACE}")
    print("=" * 100)

    bin_id = {"k": 0}

    def on_bin(events_10ms, info):
        bin_id["k"] += 1
        print(
            f"[bin {bin_id['k']:04d}] "
            f"n={info.n_events:6d}  dt_us={info.dt_us:6d}  "
            f"t=[{info.t_first}; {info.t_last}]"
        )

    run_time_slicer(
        aedat4_path=path,
        dt_ms=DT_MS,
        on_bin=on_bin,
        max_bins=MAX_BINS,
        use_dv_timesurface=USE_DV_TIMESURFACE,
    )

    print("=" * 100)
    print("[Step3] Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
