import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config (edit here)
# =========================
CSV_DIR = r"data/emlb_rcf_verify/daycsv10ms"   # folder containing 8 csv files
OUT_DIR = r"outputs/efficiency"     # where to save summary csv + figure (can be same as CSV_DIR)
TIME_COL = "time_ms"
import os
import re
import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib as mpl


# =========================
# Global font (Times New Roman)
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False


# =========================
# Config (edit here)
# =========================
# CSV_DIR = r"C:\path\to\your\csv_folder"   # folder containing 8 csv files
# OUT_DIR = r"C:\path\to\output_folder"     # where to save summary csv + figure
# TIME_COL = "time_ms"

REMOVE_MAX = True                 # remove one maximum per file
ERROR_BAR = "std"                 # "std" or "sem"
ND_ORDER = ["ND00", "ND04", "ND16", "ND64"]

FIG_W, FIG_H = 9.0, 4.8
DPI = 200


# =========================
# Helpers
# =========================
def parse_scene_from_filename(fname: str):
    base = os.path.basename(fname)
    low = base.lower()

    scene = None
    if "day" in low:
        scene = "Day"
    if "night" in low:
        scene = "Night" if scene is None else scene

    m = re.search(r"nd(\d{2})", low)
    nd = None
    if m:
        nd = f"ND{m.group(1)}"

    if scene is None or nd is None:
        raise ValueError(f"Cannot parse scene/nd from filename: {base}")

    return scene, nd


def compute_trimmed_stats(time_ms: np.ndarray, remove_max: bool = True):
    t = np.asarray(time_ms, dtype=np.float64)
    t = t[~np.isnan(t)]

    if t.size == 0:
        raise ValueError("Empty time array after NaN removal.")

    if remove_max and t.size >= 2:
        t = np.delete(t, np.argmax(t))

    mean = float(np.mean(t))
    std = float(np.std(t, ddof=1)) if t.size >= 2 else 0.0
    sem = float(std / np.sqrt(t.size)) if t.size >= 2 else 0.0
    n = int(t.size)
    return mean, std, sem, n


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if len(csv_paths) == 0:
        raise RuntimeError(f"No CSV files found under: {CSV_DIR}")

    rows = []
    for path in csv_paths:
        scene, nd = parse_scene_from_filename(path)

        df = pd.read_csv(path)
        if TIME_COL not in df.columns:
            raise KeyError(f"Missing column '{TIME_COL}' in: {os.path.basename(path)}")

        mean, std, sem, n = compute_trimmed_stats(df[TIME_COL].values, REMOVE_MAX)
        rows.append({
            "file": os.path.basename(path),
            "scene": scene,
            "nd": nd,
            "n_used": n,
            "mean_ms": mean,
            "std_ms": std,
            "sem_ms": sem,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "runtime_summary.csv"), index=False)

    overall_mean = float(summary["mean_ms"].mean())

    agg = summary.groupby(["scene", "nd"], as_index=False).agg(
        mean_ms=("mean_ms", "mean"),
        std_ms=("std_ms", "mean"),
        sem_ms=("sem_ms", "mean"),
    )

    def get_val(scene, nd, col):
        hit = agg[(agg["scene"] == scene) & (agg["nd"] == nd)]
        return float(hit.iloc[0][col]) if len(hit) else np.nan

    day_means = [get_val("Day", nd, "mean_ms") for nd in ND_ORDER]
    night_means = [get_val("Night", nd, "mean_ms") for nd in ND_ORDER]

    if ERROR_BAR.lower() == "sem":
        day_err = [get_val("Day", nd, "sem_ms") for nd in ND_ORDER]
        night_err = [get_val("Night", nd, "sem_ms") for nd in ND_ORDER]
        err_label = "SEM"
    else:
        day_err = [get_val("Day", nd, "std_ms") for nd in ND_ORDER]
        night_err = [get_val("Night", nd, "std_ms") for nd in ND_ORDER]
        err_label = "STD"

    x = np.arange(len(ND_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.bar(x - width/2, day_means, width, yerr=day_err, capsize=4, label="Day")
    ax.bar(x + width/2, night_means, width, yerr=night_err, capsize=4, label="Night")

    ax.set_xticks(x)
    ax.set_xticklabels(ND_ORDER)
    ax.set_ylabel("Avg. processing time per bin (ms)")
    ax.set_title(
        f"RCF Runtime (bin = 10ms,remove max, error={err_label}) | "
        f"Overall mean over files = {overall_mean:.3f} ms"
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "runtime_bar.png"))
    plt.close(fig)

    print(f"[OK] Overall mean over files: {overall_mean:.6f} ms")


if __name__ == "__main__":
    main()
