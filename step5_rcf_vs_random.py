# -*- coding: utf-8 -*-
"""
批量生成 RCF vs Random 的 mESR 主展示图，并拼成 2×4 大图。

输入：一个目录下的多个 csv（每个 csv 包含 eta0..eta5, mesr_rcf0..5, mesr_rand0..5,
     delta0..5, keep_ratio0..5 等列）
输出：
1) 每个 csv -> 一张主展示图（上：mESR曲线对比；下：ΔmESR柱状并标注保留率），保存到同目录
2) 把所有生成的主展示图按 2×4 拼成一张大图，保存到同目录

关键细节（论文版）：
- 图幅更窄：适合单栏排版
- 文字不被边框遮挡：抬高 ylim + text offset + clip_on=False
- 标题更论文化：用“Mean ESR Improvement …”而不是只写 η
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# 你只需要改这里
# -----------------------------
CSV_DIR = r"outputs/step5_compare_csv"  # 改成你的csv文件夹
OUT_SUFFIX = "_main_mesr.png"
BIG_FIG_NAME = "main_mesr_2x4.png"

# 拼图布局
N_COLS = 4
N_ROWS = 2

# 输出分辨率
DPI = 300


def parse_title_from_filename(filename: str) -> str:
    """
    从 csv 文件名中解析论文用标题，例如：
    compare_mesr_v1_night_nd04_rcf_vs_random.csv -> Night ND04
    """
    name = filename.lower()

    # day / night
    if "night" in name:
        scene = "Night"
    elif "day" in name:
        scene = "Day"
    else:
        scene = "Scene"

    # nd level
    nd = "ND"
    for k in ["nd00", "nd04", "nd64", "nd16"]:
        if k in name:
            nd = k.upper()
            break

    return f"{scene} {nd}"


def _find_total_row(df: pd.DataFrame) -> pd.Series | None:
    """找到 TOTAL_MEAN_OVER_FILES 行（如果存在）。"""
    if "file" not in df.columns:
        return None
    hits = df[df["file"].astype(str) == "TOTAL_MEAN_OVER_FILES"]
    if len(hits) == 0:
        return None
    return hits.iloc[0]


def _safe_get_columns(df: pd.DataFrame, prefix: str, n: int = 6):
    cols = []
    for i in range(n):
        c = f"{prefix}{i}"
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
        cols.append(c)
    return cols


def _extract_summary(df: pd.DataFrame):
    """
    从一个csv提取：
    - eta: (6,)
    - retention mean: (6,)  来自所有序列行均值 keep_ratio{i}（符合“先算每个序列，再取平均”）
    - mesr_rcf, mesr_rand, delta: (6,) 来自 TOTAL 行或序列均值
    """
    eta_cols = _safe_get_columns(df, "eta", 6)
    keep_cols = _safe_get_columns(df, "keep_ratio", 6)
    rcf_cols = _safe_get_columns(df, "mesr_rcf", 6)
    rnd_cols = _safe_get_columns(df, "mesr_rand", 6)
    dlt_cols = _safe_get_columns(df, "delta", 6)

    # eta 取第一行即可（一般每行一致）
    eta = df.iloc[0][eta_cols].astype(float).to_numpy()

    # retention：对“非TOTAL行”求均值
    if "file" in df.columns:
        data_rows = df[df["file"].astype(str) != "TOTAL_MEAN_OVER_FILES"]
    else:
        data_rows = df
    retention = data_rows[keep_cols].astype(float).mean(axis=0).to_numpy()

    total = _find_total_row(df)
    if total is not None:
        mesr_rcf = total[rcf_cols].astype(float).to_numpy()
        mesr_rnd = total[rnd_cols].astype(float).to_numpy()
        delta = total[dlt_cols].astype(float).to_numpy()
    else:
        mesr_rcf = data_rows[rcf_cols].astype(float).mean(axis=0).to_numpy()
        mesr_rnd = data_rows[rnd_cols].astype(float).mean(axis=0).to_numpy()
        delta = data_rows[dlt_cols].astype(float).mean(axis=0).to_numpy()

    return eta, retention, mesr_rcf, mesr_rnd, delta


def _paperize_axes(ax):
    """统一轻量论文风格：细网格 + 合理边框。"""
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)



def plot_main_figure(eta, retention, mesr_rcf, mesr_rnd, delta, title, out_path):
    """
    论文版主展示图：
    上：mESR(RCF) vs mESR(Random deletion)
    下：ΔmESR 柱状 + r(保留率)标注
    """
    eta = np.asarray(eta, dtype=float)
    retention = np.asarray(retention, dtype=float)
    mesr_rcf = np.asarray(mesr_rcf, dtype=float)
    mesr_rnd = np.asarray(mesr_rnd, dtype=float)
    delta = np.asarray(delta, dtype=float)

    # 更窄一些（继续收缩宽度）
    fig = plt.figure(figsize=(5.2, 4.4))  # 之前6.2，现在更窄
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.10)  # hspace减小，整体更紧凑

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    # -----------------------------
    # 上图：曲线对比
    # -----------------------------
    ax1.plot(eta, mesr_rcf, marker="o", linewidth=1.6, label="RCF")
    ax1.plot(eta, mesr_rnd, marker="o", linewidth=1.6, label="Random deletion")
    ax1.set_ylabel("mESR")
    ax1.legend(loc="lower left", frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)

    # 关键：上子图不要 x-label，也不要 x tick label（避免重复 & 避免碰撞）
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", labelbottom=False)

    # -----------------------------
    # 下图：ΔmESR 柱状
    # -----------------------------
    # 柱宽更紧凑：按相邻 eta 的最小间距来确定宽度
    if len(eta) >= 2:
        min_step = float(np.min(np.diff(np.sort(eta))))
        bar_w = 0.55 * min_step
    else:
        bar_w = 0.03

    ax2.bar(eta, delta, width=bar_w)
    ax2.set_xlabel(r"Threshold $\eta$")
    ax2.set_ylabel(r"$\Delta$mESR")
    ax2.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    # 关键：收紧 x 轴左右留白，让柱子看起来不“松”
    # 让左右边界只比最左/最右多一点点
    if len(eta) >= 2:
        pad = 0.8 * bar_w
    else:
        pad = 0.02
    ax2.set_xlim(float(eta.min() - pad), float(eta.max() + pad))

    # -----------------------------
    # 防止标注被边框压住：抬高ylim + 偏移 + 不裁剪
    # -----------------------------
    y_max = float(np.nanmax(delta))
    y_min = float(np.nanmin(delta))
    lower = min(0.0, y_min * 1.10)
    upper = y_max * 1.28 + 1e-9
    ax2.set_ylim(lower, upper)

    y_range = upper - lower
    offset = 0.04 * y_range

    for x, y, r in zip(eta, delta, retention):
        ax2.text(
            x, y + offset,
            f"r={int(round(r * 100))}%",
            ha="center", va="bottom",
            fontsize=9,
            clip_on=False
        )

    # 总标题只显示条件标签（你要的 Night ND04）
    fig.suptitle(title, y=0.965, fontsize=11)

    # tight_layout 预留标题空间；不要 bbox_inches="tight" 过度裁切
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300)  # 不用 bbox_inches="tight"，减少“裁切压迫”
    plt.close(fig)


def stitch_images_grid(image_paths, out_path, n_rows=2, n_cols=4, pad=12, bg=(255, 255, 255)):
    """把若干张图拼成固定行列的大图（论文版留白更舒服）。"""
    if len(image_paths) == 0:
        raise ValueError("No images to stitch.")

    # 只取前 n_rows*n_cols 张
    image_paths = image_paths[: n_rows * n_cols]
    imgs = [Image.open(p).convert("RGB") for p in image_paths]

    # 统一缩放到同一尺寸（避免某张图尺寸略不同导致拼接不齐）
    target_w = min(im.width for im in imgs)
    target_h = min(im.height for im in imgs)
    imgs = [im.resize((target_w, target_h), resample=Image.Resampling.LANCZOS) for im in imgs]

    canvas_w = n_cols * target_w + (n_cols - 1) * pad
    canvas_h = n_rows * target_h + (n_rows - 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    for idx, im in enumerate(imgs):
        r = idx // n_cols
        c = idx % n_cols
        x = c * (target_w + pad)
        y = r * (target_h + pad)
        canvas.paste(im, (x, y))

    canvas.save(out_path)


def main():
    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No csv files found in: {CSV_DIR}")

    out_imgs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        eta, retention, mesr_rcf, mesr_rnd, delta = _extract_summary(df)

        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(CSV_DIR, f"{base}{OUT_SUFFIX}")

        # title：默认用文件名（你如果有固定命名规则，可在这里做映射更论文式）
        # title = base
        title = parse_title_from_filename(base)

        plot_main_figure(eta, retention, mesr_rcf, mesr_rnd, delta, title, out_path)
        out_imgs.append(out_path)
        print(f"[OK] saved: {out_path}")

    # 拼成 2×4 大图：按文件名排序后的前 8 张
    big_out = os.path.join(CSV_DIR, BIG_FIG_NAME)
    stitch_images_grid(out_imgs, big_out, n_rows=N_ROWS, n_cols=N_COLS)
    print(f"[OK] stitched: {big_out}")


if __name__ == "__main__":
    main()
