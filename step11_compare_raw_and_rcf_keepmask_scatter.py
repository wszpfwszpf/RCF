# -*- coding: utf-8 -*-
# step11_compare_raw_and_rcf_keepmask_scatter.py
# ---------------------------------------------------------
# 功能：给定一个 .dat 文件、一个 keepmask(npz，含6列对应6个η)、以及起止时间（秒），
#      在相同时间窗与相同ROI条件下，分别对 raw 与 6个η下的RCF结果绘制一致的scatter对比图，
#      共生成7张图，并保存到 data/DVcompare/<文件名>/ 目录中。
#
# 说明：
# - 可视化使用 scatter 原样绘制（不量化、不去重）
# - ROI 固定为 x∈[600,670]
# - 时间窗由外部给定（秒），内部转换为微秒后进行裁剪
# - 绘图函数使用：from beam.utils.io.scatter_plotv2 import plot_event_2d
# ---------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
from matplotlib import rcParams

from beam.utils.io.psee_loader import PSEELoader
from beam.utils.io.scatter_plotv2 import plot_event_2d

rcParams["font.family"] = "Times New Roman"


# ------------------------------- #
# 你只需要改这里的配置（不写命令行）
# ------------------------------- #
# DAT_PATH = os.path.join("data", "DV", "xxx.dat")   # TODO: 改成你的dat路径
# DAT_PATH = r'data/DV/Off_set3_trail5.dat'
# KEEPMASK_NPZ_PATH = os.path.join("data", "DV", "xxx_km.npz")  #
# KEEPMASK_NPZ_PATH = 'data/DV_rcf_full/Off_set3_trail5_keepmask_10ms.npz'

DAT_PATH = r"data/DV/On_set2_trail2.dat"
KEEPMASK_NPZ_PATH = r"data/DV_rcf_full/On_set2_trail2_keepmask_10ms.npz"

# 你会提供的起止时间（秒）
T_START_S = 1.7   # TODO: 改成你的起始时间（秒）
T_END_S = 3.7    # TODO: 改成你的结束时间（秒）

# 6个η值（仅用于标题与文件名；顺序要与keepmask的6列一致）
ETAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# ROI（按你现在的约定：x∈[600,670]）
ROI_X_MIN = 600
ROI_X_MAX = 670

# 绘图切片参数：保持与你step9逻辑一致（按固定y_aix画 t-x）
Y_AIX_FOR_PLOT = 50

# 输出根目录
OUT_ROOT = os.path.join("data", "DVcompare")


def _load_keepmask_matrix(npz_path: str) -> np.ndarray:
    """
    从npz中读取keepmask矩阵，兼容不同key命名与不同shape:
    - 期望最终返回 shape = (N, 6)，N为事件数，6列对应6个η
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing keepmask npz: {npz_path}")

    z = np.load(npz_path)

    # 尝试常见key
    cand_keys = ["keepmask", "km", "mask", "keep_mask", "keep"]
    arr = None
    for k in cand_keys:
        if k in z.files:
            arr = z[k]
            break

    # 如果没有匹配key，就取第一个数组
    if arr is None:
        if len(z.files) == 0:
            raise ValueError(f"Empty npz: {npz_path}")
        arr = z[z.files[0]]

    arr = np.asarray(arr)

    # 兼容shape
    # (N,6) 或 (6,N) 或 (N,)（不符合需求）
    if arr.ndim != 2:
        raise ValueError(f"keepmask must be 2D, got shape={arr.shape} from {npz_path}")

    if arr.shape[1] == 6:
        km = arr
    elif arr.shape[0] == 6 and arr.shape[1] != 6:
        km = arr.T
    else:
        raise ValueError(f"keepmask shape not compatible with 6 columns: got {arr.shape}")

    # 转成bool，兼容 {0,1} / {-1,1} 等
    km = (km > 0)

    return km


def _ensure_outdir(dat_path: str) -> str:
    """
    在 data/DVcompare 下根据dat文件名创建子目录
    """
    stem = os.path.splitext(os.path.basename(dat_path))[0]
    out_dir = os.path.join(OUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    if not os.path.exists(DAT_PATH):
        raise FileNotFoundError(f"Missing DAT: {DAT_PATH}")

    # 秒 -> 微秒（与事件时间戳单位一致）
    t_start_us = int(round(T_START_S * 1e6))
    t_end_us = int(round(T_END_S * 1e6))
    if t_end_us <= t_start_us:
        raise ValueError(f"Invalid time window: start={T_START_S}s end={T_END_S}s")

    out_dir = _ensure_outdir(DAT_PATH)
    print("=" * 100)
    print("[STEP11] Compare RAW vs RCF(keepmask 6 etas) scatter plots (save-only)")
    print(f"[IN ] DAT      : {os.path.abspath(DAT_PATH)}")
    print(f"[IN ] KEEPMASK : {os.path.abspath(KEEPMASK_NPZ_PATH)}")
    print(f"[IN ] T(s)     : [{T_START_S}, {T_END_S}]  ->  us=[{t_start_us}, {t_end_us}]")
    print(f"[ROI] x-range  : [{ROI_X_MIN}, {ROI_X_MAX}]")
    print(f"[OUT] OUTDIR   : {os.path.abspath(out_dir)}")
    print("=" * 100)

    # 读取keepmask矩阵（N,6）
    km = _load_keepmask_matrix(KEEPMASK_NPZ_PATH)

    # 读取事件（保持与你step9一致：load_delta_t 用 t_end_us）
    video = PSEELoader(DAT_PATH)
    events = video.load_delta_t(float(t_end_us))
    events["t"].sort()

    # 基于时间窗裁剪，并记录索引，保证keepmask对齐
    idx_time = []
    ev_time = []

    # 这里 events 是结构化数组，逐条迭代会比较慢，但与step9逻辑一致（你要求不改其他内容）
    for i, e in enumerate(tqdm(events, desc="Time interception", leave=False)):
        if t_start_us <= int(e["t"]) <= t_end_us:
            idx_time.append(i)
            ev_time.append(e)

    if len(idx_time) == 0:
        print("[WARN] No events in the given time window. Nothing to plot.")
        return

    # keepmask长度检查
    if km.shape[0] < len(events):
        # 保守处理：如果keepmask比events短，直接报错，避免错误对齐
        raise ValueError(
            f"keepmask length mismatch: km.N={km.shape[0]} < events.N={len(events)}. "
            f"Please ensure keepmask aligns with loaded events."
        )

    # 再按ROI裁剪（按你现有代码：event[1] 表示 x）
    idx_roi = []
    ev_roi = []
    for local_j, e in enumerate(tqdm(ev_time, desc="ROI interception", leave=False)):
        # e 是结构体，但你之前用 event[1] 访问x，这里保持一致
        # 结构体字段通常为 ['t','x','y','p']，但仍保留event[1]访问方式
        x_val = e[1]
        if ROI_X_MIN <= x_val <= ROI_X_MAX:
            idx_roi.append(idx_time[local_j])  # 回到原events索引空间
            ev_roi.append(e)

    if len(ev_roi) == 0:
        print("[WARN] No events in ROI within the given time window. Nothing to plot.")
        return

    # 1) RAW 图
    out_raw = os.path.join(out_dir, "raw_scatter.png")
    plot_event_2d(
        ev_roi,
        y_aix=Y_AIX_FOR_PLOT,
        title="Raw",
        ylim = (ROI_X_MIN, ROI_X_MAX),
        save_path=out_raw,

        show=False,
        t0_us=t_start_us
    )
    print(f"  saved: {out_raw}")

    # 2) 6 个 η 图
    if len(ETAS) != 6:
        raise ValueError(f"ETAS must have length 6, got {len(ETAS)}")

    for col, eta in enumerate(ETAS):
        mask_col = km[:, col]  # shape (N,)

        # 在 ROI+time 对应的原events索引上取mask
        keep_flags = [bool(mask_col[i]) for i in idx_roi]

        # 过滤事件
        ev_keep = [e for e, k in zip(ev_roi, keep_flags) if k]

        out_png = os.path.join(out_dir, f"eta_{eta:.3f}_scatter.png")
        plot_event_2d(
            ev_keep,
            y_aix=Y_AIX_FOR_PLOT,
            title=f"η={eta*2}",
            ylim=(ROI_X_MIN, ROI_X_MAX),
            save_path=out_png,
            show=False,
            t0_us=t_start_us
        )
        print(f"  saved: {out_png} | kept_events={len(ev_keep)}/{len(ev_roi)}")

    print("-" * 100)
    print("[DONE] All figures saved.")
    print("-" * 100)


if __name__ == "__main__":
    main()
