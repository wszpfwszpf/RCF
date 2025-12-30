# step6_align_raw_with_ldv_nolag_fixedsign.py
# ------------------------------------------------------------
# NO lag, FIXED sign for all recordings.
# If results are poor under fixed sign, the observable is invalid.
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

ROI_NPZ_DIR = r"data\DV_roi_npz"
LDV_DIR     = r"data\LDV"
OUT_DIR     = r"data\ALIGN_raw_ldv_fixedsign"
os.makedirs(OUT_DIR, exist_ok=True)

ROI_CX, ROI_CY = 600, 200
ROI_W, ROI_H = 256, 224
ROI_X0 = ROI_CX - ROI_W // 2  # 472
ROI_Y0 = ROI_CY - ROI_H // 2  # 88

MAX_T_S = 6.0
WIN_S = 0.010
HALF_WIN_S = WIN_S * 0.5

# >>> FIXED SIGN HERE <<<
FIXED_SIGN = -1   # set to +1 or -1, keep constant for all files

PLOT_EXAMPLES = 3
plt.rcParams["font.family"] = "Times New Roman"


def list_npz_files(root: str):
    return [f for f in sorted(os.listdir(root)) if f.lower().endswith(".npz")]


def load_ldv_txt(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        data = np.loadtxt(f, skiprows=5)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def load_roi_events(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    t_us = d["t"].astype(np.int64)
    x = d["x"].astype(np.int32)
    y = d["y"].astype(np.int32)

    if t_us.size > 1 and not np.all(t_us[1:] >= t_us[:-1]):
        idx = np.argsort(t_us, kind="mergesort")
        t_us, x, y = t_us[idx], x[idx], y[idx]

    t0 = t_us[0] if t_us.size > 0 else 0
    t_s = (t_us - t0).astype(np.float64) * 1e-6

    m = (t_s >= 0.0) & (t_s <= MAX_T_S)
    t_s = t_s[m]
    x = x[m] - ROI_X0
    y = y[m] - ROI_Y0
    return t_s, x, y


def build_s_of_t(ldv_t: np.ndarray, ev_t: np.ndarray, ev_x: np.ndarray):
    s = np.empty_like(ldv_t, dtype=np.float64)
    n = ev_t.shape[0]
    j0 = 0
    last = np.nan

    for k, tk in enumerate(ldv_t):
        left = tk - HALF_WIN_S
        right = tk + HALF_WIN_S

        while j0 < n and ev_t[j0] < left:
            j0 += 1
        j1 = j0
        while j1 < n and ev_t[j1] <= right:
            j1 += 1

        if j1 > j0:
            last = float(np.median(ev_x[j0:j1]))
            s[k] = last
        else:
            s[k] = last if not np.isnan(last) else 0.0

    return s


def diff_velocity(s: np.ndarray, t: np.ndarray):
    ve = np.zeros_like(s, dtype=np.float64)
    dt = np.diff(t)
    ds = np.diff(s)
    ve[1:] = np.divide(ds, dt, out=np.zeros_like(ds), where=dt > 0)
    ve[0] = ve[1] if ve.size > 1 else 0.0
    return ve


def zscore(x: np.ndarray):
    mu = np.mean(x)
    sd = np.std(x) + 1e-12
    return (x - mu) / sd


def corr_no_lag(x: np.ndarray, y: np.ndarray):
    xx = zscore(x.astype(np.float64))
    yy = zscore(y.astype(np.float64))
    if xx.size < 32:
        return 0.0
    return float(np.mean(xx * yy))


def linear_fit_nrmse(x: np.ndarray, y: np.ndarray):
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a * x + b
    rmse = np.sqrt(np.mean((yhat - y) ** 2))
    nrmse = rmse / (np.std(y) + 1e-12)
    return float(a), float(b), float(nrmse), yhat


def main():
    files = list_npz_files(ROI_NPZ_DIR)
    results = []
    plotted = 0

    for fn in files:
        stem = os.path.splitext(fn)[0]
        npz_path = os.path.join(ROI_NPZ_DIR, fn)
        txt_path = os.path.join(LDV_DIR, stem + ".txt")
        if not os.path.isfile(txt_path):
            continue

        ldv_t, ldv_v = load_ldv_txt(txt_path)
        m = (ldv_t >= 0.0) & (ldv_t <= MAX_T_S)
        ldv_t, ldv_v = ldv_t[m], ldv_v[m]

        ev_t, ev_x, _ = load_roi_events(npz_path)

        s = build_s_of_t(ldv_t, ev_t, ev_x)
        ve = diff_velocity(s, ldv_t)

        # >>> FIXED SIGN applied here <<<
        ve_use = float(FIXED_SIGN) * ve

        r = corr_no_lag(ve_use, ldv_v)
        a, b, nrmse, ve_scaled = linear_fit_nrmse(ve_use, ldv_v)

        results.append((stem, FIXED_SIGN, r, nrmse, a, b, ve_use.size))
        print(f"[OK] {stem} | fixed_sign={FIXED_SIGN} | r={r:.4f} | NRMSE={nrmse:.4f}")

        if plotted < PLOT_EXAMPLES:
            plotted += 1
            plt.figure(figsize=(8, 3))
            plt.plot(ldv_t, ldv_v, linewidth=1.0, label="LDV v(t) [m/s]")
            plt.plot(ldv_t, ve_scaled, linewidth=1.0, label="Event proxy (scaled) [m/s]")
            plt.title(f"{stem} | fixed_sign={FIXED_SIGN:+d}, r={r:.3f}, NRMSE={nrmse:.3f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity")
            plt.legend()
            out_png = os.path.join(OUT_DIR, f"{stem}_raw_fixedsign.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()

    out_csv = os.path.join(OUT_DIR, "raw_ldv_metrics_fixedsign.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("file,fixed_sign,r,nrmse,a,b,n_samples\n")
        for row in results:
            f.write(",".join(map(str, row)) + "\n")

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved plots (first {PLOT_EXAMPLES} files): {OUT_DIR}")


if __name__ == "__main__":
    main()
