import os
import glob
import csv
from tqdm import tqdm
from matplotlib import rcParams

from beam.utils.io.psee_loader import PSEELoader
from beam.utils.io.scatter_plot import plot_event_2d

rcParams['font.family'] = 'Times New Roman'


def load_ranges(csv_path: str):
    """
    读取 dv_clip_2s_ranges_by_bandrate.csv
    需要字段：file_stem, ok, t_start_us, t_end_us
    """
    mp = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stem = r["file_stem"].strip()
            ok = r["ok"].strip().upper()
            if ok != "OK":
                continue
            t_start_us = int(float(r["t_start_us"]))
            t_end_us = int(float(r["t_end_us"]))
            mp[stem] = (t_start_us, t_end_us)
    return mp


def main():
    DV_DIR = os.path.join("data", "DV")
    CSV_PATH = os.path.join("data", "DVcliped_2s_vis", "dv_clip_2s_ranges_by_bandrate.csv")
    OUT_DIR = os.path.join("data", "DVcliped_2s_vis")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    ranges = load_ranges(CSV_PATH)

    dat_files = sorted(glob.glob(os.path.join(DV_DIR, "*.dat")))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in: {DV_DIR}")

    print("=" * 100)
    print("[STEP9] DV visualization using CSV clip ranges (official demo logic, save-only)")
    print(f"[IN ] DV_DIR : {os.path.abspath(DV_DIR)}")
    print(f"[IN ] CSV    : {os.path.abspath(CSV_PATH)}")
    print(f"[OUT] OUTDIR : {os.path.abspath(OUT_DIR)}")
    print(f"[INFO] Files : {len(dat_files)}")
    print("=" * 100)

    for i, file_path in enumerate(dat_files, 1):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        if stem not in ranges:
            print(f"[{i:03d}/{len(dat_files):03d}] [SKIP] {stem} (no range in CSV or not OK)")
            continue

        t_start_us, t_end_us = ranges[stem]
        event_time = float(t_end_us)  # demo里用 load_delta_t(event_time)

        print(f"[{i:03d}/{len(dat_files):03d}] {stem} | clip_us=[{t_start_us},{t_end_us}]")

        # ---- 官方 demo 逻辑（保持结构一致） ----
        video = PSEELoader(file_path)
        events = video.load_delta_t(event_time)
        events['t'].sort()

        Oral_event_samples_clip = [
            event for event in tqdm(events, desc='Time interception', leave=False)
            if t_start_us <= event['t'] <= t_end_us
        ]

        # demo里这行注释写的是“x-range ROI”，代码实际也确实是按 event[1]=x 做筛选
        Oral_event_samples_clip_y = [
            event for event in tqdm(Oral_event_samples_clip, leave=False)
            if 600 <= event[1] <= 670
        ]

        out_png = os.path.join(OUT_DIR, f"{stem}_clip2s_scatter.png")
        plot_event_2d(Oral_event_samples_clip_y, y_aix=50, save_path=out_png, show=False)
        # ---------------------------------------

        print(f"  saved: {out_png}")

    print("-" * 100)
    print("[DONE] All figures saved.")
    print("-" * 100)


if __name__ == "__main__":
    main()
