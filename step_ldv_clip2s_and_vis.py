import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams["font.family"] = "Times New Roman"

def robust_mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (MAD)."""
    med = np.median(x)
    return np.median(np.abs(x - med))

def find_vibration_start(t: np.ndarray,
                         v: np.ndarray,
                         baseline_s: float = 0.3,
                         k: float = 6.0,
                         min_consecutive: int = 3) -> tuple[float, float, float]:
    """
    Return (t0, thr, sigma_robust).
    - baseline_s: use first baseline_s seconds for baseline stats (median & MAD).
    - k: threshold factor.
    - min_consecutive: require N consecutive samples above threshold to reduce false triggers.
    """
    assert len(t) == len(v) and len(t) > 10

    # Baseline region
    t0_base = t[0]
    mask_base = t <= (t0_base + baseline_s)
    if mask_base.sum() < 10:
        # fallback: first 10% samples
        n = max(10, int(0.1 * len(t)))
        v_base = v[:n]
    else:
        v_base = v[mask_base]

    med = np.median(v_base)
    mad = robust_mad(v_base)
    sigma = 1.4826 * mad if mad > 0 else np.std(v_base)  # fallback if MAD=0
    thr = med + k * sigma

    # Detect first run of consecutive exceedances
    exceed = np.abs(v - med) > (k * sigma)
    if exceed.any():
        idx = np.where(exceed)[0]
        # enforce consecutive run
        for i in idx:
            j = i + min_consecutive
            if j <= len(exceed) and exceed[i:j].all():
                return float(t[i]), float(thr), float(sigma)

    # Fallback: no detection -> choose time where |v-med| is maximal
    imax = int(np.argmax(np.abs(v - med)))
    return float(t[imax]), float(thr), float(sigma)

def clip_window(t: np.ndarray, v: np.ndarray, t_start: float, win_s: float = 2.0):
    """
    Clip [t_start, t_start+win_s]. If not enough tail, shift start left.
    Return (t_sel, v_sel, t_start_adj, t_end_adj).
    """
    t_min, t_max = float(t[0]), float(t[-1])
    t_end = t_start + win_s

    if t_end > t_max:
        # shift left to fit
        t_start = max(t_min, t_max - win_s)
        t_end = t_start + win_s

    # If start is too early (rare), clamp
    if t_start < t_min:
        t_start = t_min
        t_end = min(t_max, t_start + win_s)

    mask = (t >= t_start) & (t <= t_end)
    t_sel = t[mask]
    v_sel = v[mask]
    return t_sel, v_sel, float(t_start), float(t_end)

def load_ldv_txt(txt_path: str):
    """
    Your LDV format: 5 header rows then two columns: time(s), velocity(m/s).
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        data = np.loadtxt(f, skiprows=5)
    t = data[:, 0].astype(np.float64)
    v = data[:, 1].astype(np.float64)
    return t, v

def main():
    LDV_DIR = os.path.join("data", "LDV")
    OUT_DIR = os.path.join("data", "LDVvis")
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "ldv_clip_2s_ranges.csv")

    # Parameters (can be tuned)
    WIN_S = 2.0
    BASELINE_S = 0.3
    K = 6.0
    MIN_CONSEC = 3

    files = sorted(glob.glob(os.path.join(LDV_DIR, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt found in {LDV_DIR}")

    rows = []
    for i, fp in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(fp))[0]
        t, v = load_ldv_txt(fp)

        # detect vibration start
        t0, thr, sigma = find_vibration_start(
            t, v, baseline_s=BASELINE_S, k=K, min_consecutive=MIN_CONSEC
        )

        # clip 2 seconds
        t_sel, v_sel, t_s, t_e = clip_window(t, v, t0, win_s=WIN_S)

        # plot (relative time)
        t_rel = t_sel - t_s
        plt.figure(figsize=(6, 4))
        plt.plot(t_rel, v_sel, linewidth=1.2)
        plt.title(f"{name} | LDV 2s clip", fontsize=12, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
        plt.ylabel("Velocity (m/s)", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.25)

        out_png = os.path.join(OUT_DIR, f"{name}_ldv_2s.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        # record
        rows.append([
            name,
            f"{t_s:.6f}", f"{t_e:.6f}",
            f"{WIN_S:.3f}",
            f"{t0:.6f}",
            f"{BASELINE_S:.3f}",
            f"{K:.2f}",
            f"{sigma:.6e}",
        ])

        print(f"[{i:03d}/{len(files):03d}] {name} | clip [{t_s:.3f}, {t_e:.3f}] -> {out_png}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file_stem",
            "t_start_abs_s", "t_end_abs_s",
            "win_s",
            "detected_onset_t_abs_s",
            "baseline_s",
            "k",
            "sigma_robust"
        ])
        w.writerows(rows)

    print("-" * 80)
    print(f"[DONE] Saved plots to: {OUT_DIR}")
    print(f"[DONE] Saved CSV to : {csv_path}")
    print("-" * 80)

if __name__ == "__main__":
    main()
