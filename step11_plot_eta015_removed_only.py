# -*- coding: utf-8 -*-
# step11_plot_eta015_removed_only.py
# ---------------------------------------------------------
# 功能：给定一个 .dat 文件、一个 keepmask(npz，含6列对应6个η)、以及起止时间（秒），
#      在相同时间窗与相同ROI条件下，仅绘制 η=0.15 的 removed（被移除事件）scatter 图。
#
# 说明：
# - removed 定义：keepmask 为 False 的事件（即被RCF移除）
# - 使用绿色点绘制 removed（通过把事件极性统一设为 +1，并在绘图函数内部改为绿色显示）
#   注意：为了不修改 scatter_plotv2，这里直接把 removed 事件的 p 置为 1，
#   然后在本脚本中临时“替换”颜色策略：通过 plot_p=True 仍可画，但颜色仍是红/蓝。
#   因此本脚本采用一个极小的本地绘图包装：保持坐标轴、字体、xlim/ylim/t0_us一致，
#   仅将颜色固定为绿色，其他风格一致。
# ---------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
from matplotlib import rcParams
import matplotlib.pyplot as plt

from beam.utils.io.psee_loader import PSEELoader
from beam.utils.io.scatter_plotv2 import plot_event_2d  # 仍用于保持一致的坐标与风格基准

rcParams["font.family"] = "Times New Roman"


# ------------------------------- #
# 配置：路径与时间窗（你已写在文件里）
# ------------------------------- #
# DAT_PATH = r"data/DV/Off_set3_trail5.dat"
# KEEPMASK_NPZ_PATH = r"data/DV_rcf_full/Off_set3_trail5_keepmask_10ms.npz"

DAT_PATH = r"data/DV/On_set2_trail2.dat"
KEEPMASK_NPZ_PATH = r"data/DV_rcf_full/On_set2_trail2_keepmask_10ms.npz"

T_START_S = 1.7
T_END_S = 3.7

ETAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 顺序必须与keepmask 6列一致
TARGET_ETA = 0.15

ROI_X_MIN = 600
ROI_X_MAX = 670

Y_AIX_FOR_PLOT = 50

OUT_ROOT = os.path.join("data", "DVcompare")


def _load_keepmask_matrix(npz_path: str) -> np.ndarray:
    """从npz中读取keepmask矩阵，返回 shape=(N,6) 的bool矩阵。"""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing keepmask npz: {npz_path}")

    z = np.load(npz_path)
    cand_keys = ["keepmask", "km", "mask", "keep_mask", "keep"]
    arr = None
    for k in cand_keys:
        if k in z.files:
            arr = z[k]
            break
    if arr is None:
        if len(z.files) == 0:
            raise ValueError(f"Empty npz: {npz_path}")
        arr = z[z.files[0]]

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"keepmask must be 2D, got shape={arr.shape} from {npz_path}")

    if arr.shape[1] == 6:
        km = arr
    elif arr.shape[0] == 6 and arr.shape[1] != 6:
        km = arr.T
    else:
        raise ValueError(f"keepmask shape not compatible with 6 columns: got {arr.shape}")

    return (km > 0)


def _ensure_outdir(dat_path: str) -> str:
    """在 data/DVcompare 下根据dat文件名创建子目录。"""
    stem = os.path.splitext(os.path.basename(dat_path))[0]
    out_dir = os.path.join(OUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_event_2d_green(
    data,
    y_aix=None,
    x_aix=None,
    title="",
    t0_us=None,
    xlim=(0, 2),
    ylim=None,
    save_path=None,
    dpi=200
):
    """
    保持与 plot_event_2d 一致的整体风格（5x5, Times New Roman, xlim/ylim/t0_us），
    但散点颜色固定为绿色，用于 removed 可视化。
    """
    if y_aix is not None:
        t_values = [e[0] for e in data if e[2] == y_aix]
        x_values = [e[1] for e in data if e[2] == y_aix]
        if len(t_values) == 0:
            return
        t_values = np.asarray(t_values, dtype=np.float64)
        t0 = t_values[0] if t0_us is None else float(t0_us)
        t_shift = (t_values - t0) * 1e-6

        plt.figure(figsize=(5, 5))
        for t, x in zip(t_shift, x_values):
            plt.scatter(t, x, color="green")

        plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
        plt.ylabel("Y-axis", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    elif x_aix is not None:
        t_values = [e[0] for e in data if e[1] == x_aix]
        y_values = [e[2] for e in data if e[1] == x_aix]
        if len(t_values) == 0:
            return
        t_values = np.asarray(t_values, dtype=np.float64)
        t0 = t_values[0] if t0_us is None else float(t0_us)
        t_shift = (t_values - t0) * 1e-6

        plt.figure(figsize=(5, 5))
        for t, y in zip(t_shift, y_values):
            plt.scatter(t, y, color="green")

        plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
        plt.ylabel("X-axis", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    else:
        return

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def main():
    if not os.path.exists(DAT_PATH):
        raise FileNotFoundError(f"Missing DAT: {DAT_PATH}")

    # 秒 -> 微秒
    t_start_us = int(round(T_START_S * 1e6))
    t_end_us = int(round(T_END_S * 1e6))
    if t_end_us <= t_start_us:
        raise ValueError(f"Invalid time window: start={T_START_S}s end={T_END_S}s")

    out_dir = _ensure_outdir(DAT_PATH)
    print("=" * 100)
    print("[STEP11-REMOVED] Plot only η=0.15 removed (green scatter)")
    print(f"[IN ] DAT      : {os.path.abspath(DAT_PATH)}")
    print(f"[IN ] KEEPMASK : {os.path.abspath(KEEPMASK_NPZ_PATH)}")
    print(f"[IN ] T(s)     : [{T_START_S}, {T_END_S}] -> us=[{t_start_us}, {t_end_us}]")
    print(f"[ROI] x-range  : [{ROI_X_MIN}, {ROI_X_MAX}]")
    print(f"[OUT] OUTDIR   : {os.path.abspath(out_dir)}")
    print("=" * 100)

    km = _load_keepmask_matrix(KEEPMASK_NPZ_PATH)

    if TARGET_ETA not in ETAS:
        raise ValueError(f"TARGET_ETA={TARGET_ETA} not in ETAS list: {ETAS}")
    target_col = ETAS.index(TARGET_ETA)

    # 读取事件（与原脚本一致）
    video = PSEELoader(DAT_PATH)
    events = video.load_delta_t(float(t_end_us))
    events["t"].sort()

    # 时间窗裁剪（保留索引用于mask对齐）
    idx_time, ev_time = [], []
    for i, e in enumerate(tqdm(events, desc="Time interception", leave=False)):
        if t_start_us <= int(e["t"]) <= t_end_us:
            idx_time.append(i)
            ev_time.append(e)
    if len(idx_time) == 0:
        print("[WARN] No events in time window. Nothing to plot.")
        return

    if km.shape[0] < len(events):
        raise ValueError(
            f"keepmask length mismatch: km.N={km.shape[0]} < events.N={len(events)}. "
            f"Please ensure keepmask aligns with loaded events."
        )

    # ROI裁剪
    idx_roi, ev_roi = [], []
    for local_j, e in enumerate(tqdm(ev_time, desc="ROI interception", leave=False)):
        x_val = e[1]
        if ROI_X_MIN <= x_val <= ROI_X_MAX:
            idx_roi.append(idx_time[local_j])  # 原events索引空间
            ev_roi.append(e)
    if len(ev_roi) == 0:
        print("[WARN] No events in ROI within time window. Nothing to plot.")
        return

    # 取 η=0.15 对应列，得到 keep_flags
    mask_col = km[:, target_col]  # True=keep, False=remove
    keep_flags = [bool(mask_col[i]) for i in idx_roi]

    # removed = ~keep
    ev_removed = [e for e, k in zip(ev_roi, keep_flags) if not k]

    out_png = os.path.join(out_dir, "eta_0.150_removed_scatter.png")
    plot_event_2d_green(
        ev_removed,
        y_aix=Y_AIX_FOR_PLOT,
        title="η=0.3 removed",
        t0_us=t_start_us,
        xlim=(0, 2),
        ylim=(ROI_X_MIN, ROI_X_MAX),
        save_path=out_png,
        dpi=200
    )

    print(f"  saved: {out_png} | removed_events={len(ev_removed)}/{len(ev_roi)}")
    print("-" * 100)
    print("[DONE] Removed figure saved.")
    print("-" * 100)


if __name__ == "__main__":
    main()
