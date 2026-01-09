import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) OUTPUT
# =========================
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "mesr_emlb_all_methods.png")

# =========================
# 1) DATA (edit here)
# =========================
# Fixed order of 8 bars per method:
# [Day ND00, Day ND04, Day ND16, Day ND64,
#  Night ND00, Night ND04, Night ND16, Night ND64]
CATS = [
    "D-ND00", "D-ND04", "D-ND16", "D-ND64",
    "N-ND00", "N-ND04", "N-ND16", "N-ND64"
]

MESR = {
    "BAF":         [0.882045, 0.884204, 0.881529, 0.874798, 0.972110, 0.973332, 0.938549, 0.921432],
    "DWF":         [0.887864, 0.886862, 0.883861, 0.881764, 1.015741, 1.001919, 0.951321, 0.939014],
    "KNoise":      [0.874379, 0.894061, 0.923542, 0.977268, 1.039130, 1.190603, 1.428267, 1.826458],
    "RCF (η=0.3)": [0.886143, 0.889723, 0.919422, 0.926440, 0.958082, 0.980243, 1.142748, 1.474106],
}

methods = list(MESR.keys())
assert all(len(MESR[m]) == len(CATS) for m in methods)

# =========================
# 2) PLOT CONFIG (KEY PART)
# =========================
bar_w = 0.005      # 再缩小一半（原 0.028 → 0.014）
group_gap = 0.015  # 方法之间进一步压缩

# 8 fixed colors for 8 subsets
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(8)]

# =========================
# 3) COMPUTE X POSITIONS
# =========================
n_cats = len(CATS)
group_w = n_cats * bar_w

x0 = 0.0
group_centers = []
xs = {}

for m in methods:
    xs[m] = x0 + np.arange(n_cats) * bar_w
    group_centers.append(x0 + group_w / 2.0)
    x0 += group_w + group_gap

# =========================
# 4) DRAW
# =========================
plt.figure(figsize=(6, 3.2))  # 整体宽度再次减小

for m in methods:
    y = np.array(MESR[m], dtype=float)
    for ci in range(n_cats):
        plt.bar(
            xs[m][ci],
            y[ci],
            width=bar_w,
            color=colors[ci],
            edgecolor="black",
            linewidth=0.35
        )

plt.xticks(group_centers, methods)
plt.ylabel("MESR (v2, keep)")
plt.title("E-MLB MESR Comparison")

plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"[SAVED] {OUT_PATH}")
