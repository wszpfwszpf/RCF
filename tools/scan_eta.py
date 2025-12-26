# Scan eta on pre-scored RCF outputs (PyCharm one-click runnable)
# Input : data/rcf-scored/*.npz containing t,x,y,p,score1,score2,score
# Output: data/eta-scan/summary.csv (+ optional filtered npz / optional frame images)

import numpy as np
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
IN_DIR_REL = Path("data/rcf-scored")
OUT_DIR_REL = Path("data/eta-scan")
RECURSIVE = True
OVERWRITE = False
VERBOSE = True

SENSOR_WIDTH  = 346
SENSOR_HEIGHT = 260

# eta grid (hard sweep)
ETA_LIST = [float(v) for v in np.linspace(0.05, 0.30, 6)]
# scan eta from 0.05 to 0.30 with step 0.05


# Optional outputs
SAVE_FILTERED_NPZ = False      # save filtered events per eta
SAVE_FRAMES = True            # save 33ms event-frame images per eta (slow)
FRAME_WIN_US = 33000           # 33ms
MAX_FRAMES_PER_FILE = 6        # limit frames per file to reduce output volume

# Frame rendering config (only used if SAVE_FRAMES=True)
RESOLUTION = (SENSOR_WIDTH, SENSOR_HEIGHT)              # None -> infer from x/y max + 1; or set e.g. (346, 260)
POINT_SIZE = 1                 # scatter point size
DPI = 150

# Polarity mapping:
# p==1 => red, p==0 => blue  (如果你的p是{-1,+1}，下面函数也兼容)
P_POS_COLOR = "red"
P_NEG_COLOR = "blue"
REMOVED_COLOR = "green"

# -------------------------
# Helpers
# -------------------------
def project_root() -> Path:
    # tools/scan_eta.py -> project root = parent of tools
    here = Path(__file__).resolve()
    return here.parent.parent

def list_npz_files(in_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(in_dir.glob(pattern))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    out_csv.write_text("\n".join(lines), encoding="utf-8")

def save_filtered_npz(out_path: Path, t, x, y, p, score) -> None:
    np.savez_compressed(
        str(out_path),
        t=t.astype(np.int64),
        x=x.astype(np.int16),
        y=y.astype(np.int16),
        p=p.astype(np.int8),
        score=score.astype(np.float32),
    )



def render_triptych_frames_33ms(
    t_raw, x_raw, y_raw, p_raw,
    t_keep, x_keep, y_keep, p_keep,
    t_rem, x_rem, y_rem, p_rem,
    out_dir: Path, base_name: str, eta: float, resolution=None
):
    """
    Save up to MAX_FRAMES_PER_FILE triptych figures (33ms):
      (1) Raw (red/blue by polarity)
      (2) Kept/Denoised (red/blue by polarity)
      (3) Removed (green)
    """
    import matplotlib.pyplot as plt

    if len(t_raw) == 0:
        return

    # infer resolution
    if resolution is None:
        W = int(x_raw.max()) + 1
        H = int(y_raw.max()) + 1
    else:
        W, H = resolution

    start_t = int(t_raw[0])
    end_t = int(t_raw[-1])
    win = int(FRAME_WIN_US)

    def split_pol(x, y, p):
        # support p in {0,1} or {-1,+1}
        p = p.astype(np.int32)
        pos = (p == 1)
        neg = (p == 0) | (p == -1)
        return (x[pos], y[pos]), (x[neg], y[neg])

    frame_idx = 0
    while start_t + frame_idx * win < end_t and frame_idx < MAX_FRAMES_PER_FILE:
        w0 = start_t + frame_idx * win
        w1 = w0 + win

        # slice raw/keep/rem for this window
        r0 = np.searchsorted(t_raw, w0, side="left")
        r1 = np.searchsorted(t_raw, w1, side="left")
        k0 = np.searchsorted(t_keep, w0, side="left") if len(t_keep) else 0
        k1 = np.searchsorted(t_keep, w1, side="left") if len(t_keep) else 0
        m0 = np.searchsorted(t_rem, w0, side="left") if len(t_rem) else 0
        m1 = np.searchsorted(t_rem, w1, side="left") if len(t_rem) else 0

        # if raw window empty, skip
        if r1 <= r0:
            frame_idx += 1
            continue

        xr, yr, pr = x_raw[r0:r1], y_raw[r0:r1], p_raw[r0:r1]
        xk, yk, pk = x_keep[k0:k1], y_keep[k0:k1], p_keep[k0:k1] if k1 > k0 else (np.array([]), np.array([]), np.array([], dtype=np.int8))
        xm, ym, pm = x_rem[m0:m1], y_rem[m0:m1], p_rem[m0:m1] if m1 > m0 else (np.array([]), np.array([]), np.array([], dtype=np.int8))

        # split polarity for raw & keep
        (xr_pos, yr_pos), (xr_neg, yr_neg) = split_pol(xr, yr, pr)
        (xk_pos, yk_pos), (xk_neg, yk_neg) = split_pol(xk, yk, pk) if len(xk) else ((np.array([]), np.array([])), (np.array([]), np.array([])))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI)
        fig.patch.set_facecolor("white")

        # ---- Raw ----
        ax = axes[0]
        ax.scatter(xr_pos, yr_pos, s=POINT_SIZE, c=P_POS_COLOR)
        ax.scatter(xr_neg, yr_neg, s=POINT_SIZE, c=P_NEG_COLOR)
        ax.set_title(f"Raw | N={len(xr)}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal", adjustable="box")

        # ---- Denoised/Kept ----
        ax = axes[1]
        ax.scatter(xk_pos, yk_pos, s=POINT_SIZE, c=P_POS_COLOR)
        ax.scatter(xk_neg, yk_neg, s=POINT_SIZE, c=P_NEG_COLOR)
        ax.set_title(f"Denoised | N={len(xk)}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal", adjustable="box")

        # ---- Removed ----
        ax = axes[2]
        if len(xm):
            ax.scatter(xm, ym, s=POINT_SIZE, c=REMOVED_COLOR)
        ax.set_title(f"Removed | N={len(xm)}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal", adjustable="box")

        # global title
        fig.suptitle(f"{base_name} | eta={eta:.2f} | [{w0},{w1}) us", fontsize=10)
        plt.tight_layout()

        out_png = out_dir / f"{base_name}__eta{eta:.2f}__frame{frame_idx:03d}.png"
        plt.savefig(out_png, facecolor="white")
        plt.close(fig)

        frame_idx += 1


def main():
    root = project_root()
    in_dir = (root / IN_DIR_REL).resolve()
    out_dir = (root / OUT_DIR_REL).resolve()
    ensure_dir(out_dir)

    if VERBOSE:
        print(f"[INFO] Project root: {root}")
        print(f"[INFO] Input dir   : {in_dir}")
        print(f"[INFO] Output dir  : {out_dir}")
        print(f"[INFO] ETA_LIST    : {ETA_LIST}")
        print(f"[INFO] SAVE_FILTERED_NPZ={SAVE_FILTERED_NPZ}, SAVE_FRAMES={SAVE_FRAMES}")

    files = list_npz_files(in_dir, recursive=RECURSIVE)
    if not files:
        print(f"[ERROR] No .npz files found in: {in_dir}")
        return

    rows = []
    ok, fail = 0, 0

    # optional subfolders
    if SAVE_FILTERED_NPZ:
        ensure_dir(out_dir / "filtered-npz")
    if SAVE_FRAMES:
        ensure_dir(out_dir / "frames")

    for f in files:
        try:
            data = np.load(f)
            t = data["t"].astype(np.int64)
            x = data["x"].astype(np.int32)
            y = data["y"].astype(np.int32)
            p = data["p"].astype(np.int8) if "p" in data.files else np.zeros_like(t, dtype=np.int8)
            score = data["score"].astype(np.float32)

            N = len(t)
            if N == 0:
                if VERBOSE:
                    print(f"[WARN] empty: {f.name}")
                continue

            # sweep
            for eta in ETA_LIST:
                mask = score >= float(eta)
                kept = int(mask.sum())
                retention = kept / float(N)

                rows.append({
                    "file": f.name,
                    "eta": f"{eta:.2f}",
                    "N_total": N,
                    "N_keep": kept,
                    "retention": f"{retention:.6f}",
                })

                # optional: save filtered npz
                if SAVE_FILTERED_NPZ:
                    out_npz = out_dir / "filtered-npz" / f"{f.stem}__eta{eta:.2f}.npz"
                    if out_npz.exists() and not OVERWRITE:
                        pass
                    else:
                        save_filtered_npz(out_npz, t[mask], x[mask], y[mask], p[mask], score[mask])

                # optional: frames
                if SAVE_FRAMES:
                    frame_dir = out_dir / "frames" / f.stem
                    ensure_dir(frame_dir)

                    # raw
                    t_raw, x_raw, y_raw, p_raw = t, x, y, p

                    # kept
                    t_keep, x_keep, y_keep, p_keep = t[mask], x[mask], y[mask], p[mask]

                    # removed
                    inv = ~mask
                    t_rem, x_rem, y_rem, p_rem = t[inv], x[inv], y[inv], p[inv]

                    render_triptych_frames_33ms(
                        t_raw, x_raw, y_raw, p_raw,
                        t_keep, x_keep, y_keep, p_keep,
                        t_rem, x_rem, y_rem, p_rem,
                        out_dir=frame_dir,
                        base_name=f.stem,
                        eta=float(eta),
                        resolution=RESOLUTION
                    )

                # if SAVE_FRAMES:
                #     frame_dir = out_dir / "frames" / f.stem
                #     ensure_dir(frame_dir)
                #     # If you don't overwrite, skip if some frames already exist
                #     # (simple check: one expected name)
                #     probe = frame_dir / f"{f.stem}__eta{eta:.2f}__frame000.png"
                #     if probe.exists() and not OVERWRITE:
                #         pass
                #     else:
                #         render_frames_33ms(
                #             t=t[mask], x=x[mask], y=y[mask], p=p[mask],
                #             out_dir=frame_dir,
                #             base_name=f.stem,
                #             eta=float(eta),
                #             resolution=RESOLUTION
                #         )

            ok += 1
            if VERBOSE:
                print(f"[OK] scanned: {f.name} | N={N}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {f.name}: {repr(e)}")

    # write summary CSV
    out_csv = out_dir / "summary.csv"
    write_csv(rows, out_csv)

    print(f"\n[SUMMARY] files_ok={ok}, files_fail={fail}, total_files={len(files)}")
    print(f"[SUMMARY] saved: {out_csv}")

if __name__ == "__main__":
    main()

# Run in PyCharm: just click Run on this file.

