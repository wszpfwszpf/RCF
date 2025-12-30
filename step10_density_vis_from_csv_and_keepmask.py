# stepX_scatter_from_csv_ranges_pycharm.py
# ------------------------------------------------------------
# PyCharm one-click script:
# - read clip windows from CSV (microseconds)
# - loop all DV .dat in data/DV
# - clip by [t_start_us, t_end_us]
# - filter x in [600,670]
# - scatter plot (vectorized per polarity) and save (no show)
# - title_tag interface reserved for raw / eta0.20 / ...
# ------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Hyper-parameters (EDIT HERE)
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))  # 脚本与 data 同级时OK；否则改成你的工程根目录
DV_DIR = os.path.join(PROJECT_ROOT, "data", "DV")          # .dat 输入目录
# CSV_PATH = os.path.join(PROJECT_ROOT, "dv_clip_2s_ranges_by_bandrate.csv")  # 你的csv（已上传那份）
CSV_PATH = r'data/DVcliped_2s_vis/dv_clip_2s_ranges_by_bandrate.csv'
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "DVcliped_2s_visv2")             # 输出图目录

# clip window from csv
USE_ONLY_OK_ROWS = True  # csv里如果有 ok 列且为 OK，则只用这些行；没有 ok 列会自动忽略此配置

# scatter configuration (尽量贴近官方 demo)
Y_AIX = 50               # 官方demo默认用 y=50 那条线；你要换就改这里（注意：y取值必须在0~719）
X_MIN, X_MAX = 600, 670  # 你指定的 x 范围
TITLE_TAG = "raw"        # 标题后缀接口：raw / eta0.20 / ...

# figure saving
FIGSIZE = (5, 5)
DPI = 200
POINT_SIZE = 6           # s=6 保持你之前的视觉密度
SAVE_BBOX_TIGHT = True

# performance tweak (不改变纹理，只减少一些渲染开销)
ANTIALIASED = False
RASTERIZED = True        # png 也会有一定收益
# ============================================================


# ---- adjust this import if your project structure differs ----
# 如果你的 psee_loader.py 在别的目录，改成对应 import 即可
from beam.utils.io.psee_loader import PSEELoader


def load_events_clip_us(dat_path: str, t_start_us: int, t_end_us: int):
    """
    Load events in [t_start_us, t_end_us] (microseconds) from .dat using PSEELoader.
    Strategy:
      - load_delta_t(t_end_us) loads events from file start to t_end_us
      - then filter by t >= t_start_us
    """
    video = PSEELoader(dat_path)
    events = video.load_delta_t(int(t_end_us))
    t_all = events["t"].astype(np.int64)
    return events[t_all >= int(t_start_us)]


def plot_scatter_vectorized(t_us, x_plot, p, save_path, title_tag):
    """
    Vectorized scatter (per polarity once). No show, save only.
    """
    if t_us.size == 0:
        return False

    # time shift to 0
    t_s = (t_us - t_us[0]).astype(np.float64) * 1e-6

    # polarity to {0,1}
    if p.min() < 0:
        p01 = (p > 0).astype(np.int8)
    else:
        p01 = p.astype(np.int8)

    idx_pos = (p01 == 1)
    idx_neg = ~idx_pos

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # two scatters max (key speed-up)
    if idx_pos.any():
        # ax.scatter(
        #     t_s[idx_pos], x_plot[idx_pos],
        #     c="red", s=POINT_SIZE, linewidths=0,
        #     antialiased=ANTIALIASED, rasterized=RASTERIZED
        # )
        # ax.scatter(t_s[idx_pos], x_plot[idx_pos],
        #            c="red", s=36, marker='o', alpha=1.0)

        color = 'red'
        plt.scatter(t_s[idx_pos], x_plot[idx_pos], color=color)

    if idx_neg.any():
        # ax.scatter(
        #     t_s[idx_neg], x_plot[idx_neg],
        #     c="blue", s=POINT_SIZE, linewidths=0,
        #     antialiased=ANTIALIASED, rasterized=RASTERIZED
        # )
        color = 'blue'
        plt.scatter(t_s[idx_pos], x_plot[idx_pos], color=color)

    ax.set_xlabel("Time (s)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y-axis", fontsize=14, fontweight="bold")
    ax.set_title(f"Scatter visualization of dynamic visual data ({title_tag})",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(left=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if SAVE_BBOX_TIGHT:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    else:
        fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- read csv --------
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # expect file_stem, t_start_us, t_end_us
    required = {"file_stem", "t_start_us", "t_end_us"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {required}, but got {list(df.columns)}")

    if USE_ONLY_OK_ROWS and ("ok" in df.columns):
        df = df[df["ok"].astype(str).str.upper().eq("OK")].copy()

    clip_map = {}
    for _, r in df.iterrows():
        stem = str(r["file_stem"])
        clip_map[stem] = (int(r["t_start_us"]), int(r["t_end_us"]))

    # -------- scan dat files --------
    dat_files = sorted(glob.glob(os.path.join(DV_DIR, "*.dat")))
    if len(dat_files) == 0:
        raise FileNotFoundError(f"No .dat files in: {DV_DIR}")

    print("====================================================================================================")
    print("[SCATTER] DV clip scatter (vectorized, PyCharm one-click)")
    print(f"[IN ] DV_DIR   : {DV_DIR}")
    print(f"[CSV] CSV_PATH : {CSV_PATH} | entries={len(clip_map)}")
    print(f"[OUT] OUT_DIR  : {OUT_DIR}")
    print(f"[CFG] y_aix    : {Y_AIX}")
    print(f"[CFG] x_range  : [{X_MIN},{X_MAX}]")
    print(f"[CFG] title    : {TITLE_TAG}")
    print("====================================================================================================")

    n_ok, n_skip = 0, 0

    for idx, dat_path in enumerate(dat_files, 1):
        stem = os.path.splitext(os.path.basename(dat_path))[0]
        if stem not in clip_map:
            n_skip += 1
            print(f"[{idx:03d}/{len(dat_files):03d}] {stem} | skip: no clip window in CSV")
            continue

        t_start_us, t_end_us = clip_map[stem]
        print(f"[{idx:03d}/{len(dat_files):03d}] {stem} | clip=({t_start_us},{t_end_us}) us")

        # load events within time window
        events = load_events_clip_us(dat_path, t_start_us, t_end_us)
        if events.size == 0:
            n_skip += 1
            print("  -> empty after time clip, skip")
            continue

        # prefilter x range to cut plotting cost
        x_all = events["x"].astype(np.int32)
        m = (x_all >= X_MIN) & (x_all <= X_MAX)
        events = events[m]
        if events.size == 0:
            n_skip += 1
            print("  -> empty after x-range filter, skip")
            continue

        # official demo plots events on a single y line: y == Y_AIX
        y_all = events["y"].astype(np.int32)
        m = (y_all == Y_AIX)
        events_line = events[m]
        if events_line.size == 0:
            n_skip += 1
            print("  -> empty after y_aix filter, skip")
            continue

        t_us = events_line["t"].astype(np.int64)
        x_plot = events_line["x"].astype(np.int32)  # 注意：纵轴画的是 x（官方demo就是画 t vs x）
        p = events_line["p"].astype(np.int8)

        save_path = os.path.join(
            OUT_DIR,
            f"{stem}_clip_{t_start_us}-{t_end_us}us_x{X_MIN}-{X_MAX}_y{Y_AIX}_{TITLE_TAG}.png"
        )

        ok = plot_scatter_vectorized(t_us, x_plot, p, save_path, TITLE_TAG)
        if ok:
            n_ok += 1
            print(f"  saved: {os.path.basename(save_path)} | N={t_us.size:,}")

    print("----------------------------------------------------------------------------------------------------")
    print(f"[DONE] saved={n_ok} | skipped={n_skip}")
    print("----------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
