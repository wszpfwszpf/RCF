#
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# def plot_event_2d(data, y_aix=None, x_aix=None, plot_p=True):
#     if y_aix is not None:
#         t_values = [event[0] for event in data if event[2] == y_aix]
#         x_values = [event[1] for event in data if event[2] == y_aix]
#         t_values_shifted = (t_values - t_values[0])*1e-6
#         if plot_p:
#             p_values = [event[3] for event in data if event[2] == y_aix]
#             # 绘制所有散点
#             plt.figure(figsize=(5, 5))
#             for t, x, p in tqdm(zip(t_values_shifted, x_values, p_values), total=len(t_values), desc='event scatters plot'):
#                 color = 'red' if p == 1 else 'blue'
#                 plt.scatter(t, x, color=color)
#
#             plt.xlabel('Time (s)',fontsize=14, fontweight='bold')
#             plt.ylabel('Y-axis',fontsize=14, fontweight='bold')
#             plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
#             plt.xlim(left=0)
#             plt.show()
#         else:
#             plt.figure(figsize=(5, 5))
#             for t, x in tqdm(zip(t_values_shifted, x_values), total=len(t_values),
#                                 desc='event scatters plot'):
#                 color = 'black'
#                 plt.scatter(t, x, color=color)
#
#             plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
#             plt.ylabel('Y-axis', fontsize=14, fontweight='bold')
#             plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
#             plt.xlim(left=0)
#             plt.show()
#
#     elif x_aix is not None:
#         t_values = [event[0] for event in data if event[1] == x_aix]
#         y_values = [event[2] for event in data if event[1] == x_aix]
#
#
#         if plot_p:
#             p_values = [event[3] for event in data if event[1] == x_aix]
#
#             plt.figure(figsize=(5, 5))
#             for t, y, p in tqdm(zip(t_values, y_values, p_values), total=len(t_values), desc='event scatters plot'):
#                 color = 'red' if p == 1 else 'blue'
#                 plt.scatter(t, y, color=color)
#
#             plt.xlabel('Time (s)',fontsize=14, fontweight='bold')
#             plt.ylabel('X-axis',fontsize=14, fontweight='bold')
#             plt.title("Scatter visualization of dynamic visual data",fontsize=14, fontweight='bold')
#             plt.xlim(left=0)
#             plt.show()
#         else:
#             plt.figure(figsize=(5, 5))
#             for t, y in tqdm(zip(t_values, y_values), total=len(t_values), desc='event scatters plot'):
#                 color = 'black'
#                 plt.scatter(t, y, color=color)
#
#             plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
#             plt.ylabel('X-axis', fontsize=14, fontweight='bold')
#             plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
#             plt.xlim(left=0)
#             plt.show()
#
#
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_event_2d(data, y_aix=None, x_aix=None, plot_p=True, save_path=None, show=False, dpi=200):
    if y_aix is not None:
        t_values = [event[0] for event in data if event[2] == y_aix]
        x_values = [event[1] for event in data if event[2] == y_aix]
        if len(t_values) == 0:
            # 没有可画的数据：直接返回
            return
        t_values_shifted = (t_values - t_values[0]) * 1e-6

        if plot_p:
            p_values = [event[3] for event in data if event[2] == y_aix]
            plt.figure(figsize=(5, 5))
            for t, x, p in tqdm(zip(t_values_shifted, x_values, p_values), total=len(t_values), desc='event scatters plot'):
                color = 'red' if p == 1 else 'blue'
                plt.scatter(t, x, color=color)

            plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
            plt.ylabel('Y-axis', fontsize=14, fontweight='bold')
            plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
            plt.xlim(left=0)
        else:
            plt.figure(figsize=(5, 5))
            for t, x in tqdm(zip(t_values_shifted, x_values), total=len(t_values), desc='event scatters plot'):
                plt.scatter(t, x, color='black')

            plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
            plt.ylabel('Y-axis', fontsize=14, fontweight='bold')
            plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
            plt.xlim(left=0)

    elif x_aix is not None:
        t_values = [event[0] for event in data if event[1] == x_aix]
        y_values = [event[2] for event in data if event[1] == x_aix]
        if len(t_values) == 0:
            return

        if plot_p:
            p_values = [event[3] for event in data if event[1] == x_aix]
            plt.figure(figsize=(5, 5))
            for t, y, p in tqdm(zip(t_values, y_values, p_values), total=len(t_values), desc='event scatters plot'):
                color = 'red' if p == 1 else 'blue'
                plt.scatter(t, y, color=color)

            plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
            plt.ylabel('X-axis', fontsize=14, fontweight='bold')
            plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
            plt.xlim(left=0)
        else:
            plt.figure(figsize=(5, 5))
            for t, y in tqdm(zip(t_values, y_values), total=len(t_values), desc='event scatters plot'):
                plt.scatter(t, y, color='black')

            plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
            plt.ylabel('X-axis', fontsize=14, fontweight='bold')
            plt.title("Scatter visualization of dynamic visual data", fontsize=14, fontweight='bold')
            plt.xlim(left=0)

    # ---- 新增：保存/展示控制（替换原来的 plt.show()）----
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

