# scatter_plotv2.py
# ---------------------------------------------------------
# 本文件用于对事件流数据进行二维 scatter 可视化。
# 支持按固定 x 或 y 位置切片，将事件的时间轴与空间轴进行散点展示。
#
# 主要用于科研论文中的定性对比展示（如 raw 与不同 η 下的 RCF 结果）。
#
# 本版本特性（为论文对比而设计）：
# 1) 支持统一的时间零点 t0_us（微秒），避免去噪后整体左移
# 2) 支持固定时间轴范围 xlim（例如 0–2 秒）
# 3) 支持固定纵轴范围 ylim（例如 ROI 的空间范围）
# ---------------------------------------------------------

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def plot_event_2d(
    data,
    y_aix=None,
    x_aix=None,
    plot_p=True,
    title="Scatter visualization of dynamic visual data",
    t0_us=None,              # 统一时间零点（微秒）
    xlim=(0, 2),             # 时间轴范围（秒）
    ylim=None,               # 纵轴范围（像素或坐标）
    save_path=None,
    show=False,
    dpi=200
):
    """
    对事件流数据进行二维 scatter 可视化。

    参数说明：
    data      : 事件数据，事件格式为 (t, x, y, p)，t 单位为微秒
    y_aix     : 固定 y 轴位置（若不为 None，则绘制 t-x 图）
    x_aix     : 固定 x 轴位置（若不为 None，则绘制 t-y 图）
    plot_p    : 是否区分事件极性（True：红/蓝；False：黑色）
    title     : 图像标题（如 Raw / η=0.3）
    t0_us     : 时间零点（微秒），推荐传入裁剪窗口起点
    xlim      : x 轴显示范围（秒）
    ylim      : y 轴显示范围（空间坐标）
    save_path : 图像保存路径
    show      : 是否显示图像
    dpi       : 保存图像分辨率
    """

    # -------------------------------------------------
    # 分支 1：固定 y，绘制 (t, x)
    # -------------------------------------------------
    if y_aix is not None:
        t_values = [e[0] for e in data if e[2] == y_aix]
        x_values = [e[1] for e in data if e[2] == y_aix]

        if len(t_values) == 0:
            return

        t_values = np.asarray(t_values, dtype=np.float64)
        t0 = t_values[0] if t0_us is None else float(t0_us)
        t_shift = (t_values - t0) * 1e-6  # us → s

        plt.figure(figsize=(5, 5))

        if plot_p:
            p_values = [e[3] for e in data if e[2] == y_aix]
            for t, x, p in tqdm(
                zip(t_shift, x_values, p_values),
                total=len(t_shift),
                desc="event scatters plot"
            ):
                plt.scatter(t, x, color="red" if p == 1 else "blue")
        else:
            for t, x in tqdm(
                zip(t_shift, x_values),
                total=len(t_shift),
                desc="event scatters plot"
            ):
                plt.scatter(t, x, color="black")

        plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
        plt.ylabel("Y-axis", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    # -------------------------------------------------
    # 分支 2：固定 x，绘制 (t, y)
    # -------------------------------------------------
    elif x_aix is not None:
        t_values = [e[0] for e in data if e[1] == x_aix]
        y_values = [e[2] for e in data if e[1] == x_aix]

        if len(t_values) == 0:
            return

        t_values = np.asarray(t_values, dtype=np.float64)
        t0 = t_values[0] if t0_us is None else float(t0_us)
        t_shift = (t_values - t0) * 1e-6  # us → s

        plt.figure(figsize=(5, 5))

        if plot_p:
            p_values = [e[3] for e in data if e[1] == x_aix]
            for t, y, p in tqdm(
                zip(t_shift, y_values, p_values),
                total=len(t_shift),
                desc="event scatters plot"
            ):
                plt.scatter(t, y, color="red" if p == 1 else "blue")
        else:
            for t, y in tqdm(
                zip(t_shift, y_values),
                total=len(t_shift),
                desc="event scatters plot"
            ):
                plt.scatter(t, y, color="black")

        plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
        plt.ylabel("X-axis", fontsize=14, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")

        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

    else:
        return

    # -------------------------------------------------
    # 保存 / 显示
    # -------------------------------------------------
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")
