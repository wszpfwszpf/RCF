# beam/render_dv_roi_pack.py
# -*- coding: utf-8 -*-
"""
For each .dat in <proj_root>/data/DV:
  - use ONLY first 6 seconds of events (from the first event timestamp)
  - render 33ms accumulated frames:
      (A) ROI-only frame images -> <proj_root>/data/DVvis_roi/<stem>/frame_XXXXXX.png
      (B) Full-frame images with ROI red box -> <proj_root>/data/DVvis_roi_marked/<stem>/frame_XXXXXX.png
  - save ROI events (t,x,y,p) -> <proj_root>/data/DV_roi_npz/<stem>.npz

Run from beam/:
  python beam/render_dv_roi_pack.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.io.psee_loader import PSEELoader


# -----------------------------
# Fixed ROI (your final decision)
# center=(600,200), size=(256,224)
# -----------------------------
ROI_CX, ROI_CY = 600, 200
ROI_W, ROI_H = 256, 224

X0 = int(ROI_CX - ROI_W // 2)  # 472
X1 = int(X0 + ROI_W)           # 728 (exclusive)
Y0 = int(ROI_CY - ROI_H // 2)  # 88
Y1 = int(Y0 + ROI_H)           # 312 (exclusive)

# -----------------------------
# Rendering / time settings
# -----------------------------
WIN_US = 33_000          # 33ms
CLIP_US = 6_000_000      # only first 6 seconds

# Full sensor resolution (you said 1280x720)
FULL_W, FULL_H = 1280, 720


def find_project_root(start: Path) -> Path:
    """Find repo root by searching upwards for both 'data' and 'beam' folders."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").is_dir() and (cur / "beam").is_dir():
            return cur
        cur = cur.parent
    return start.resolve().parent


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def field_pick(dtype_names, *cands):
    for c in cands:
        if c in dtype_names:
            return c
    return None


def render_events_to_rgb(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Render events to an RGB image on white background:
      p>0 red, p<=0 blue
    """
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    if x.size == 0:
        return img

    xx = x.astype(np.int32)
    yy = y.astype(np.int32)
    pp = p.astype(np.int16)

    valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
    if not np.any(valid):
        return img

    xx, yy, pp = xx[valid], yy[valid], pp[valid]

    pos = pp > 0
    neg = ~pos

    # red for pos, blue for neg
    img[yy[pos], xx[pos]] = np.array([255, 0, 0], dtype=np.uint8)
    img[yy[neg], xx[neg]] = np.array([0, 0, 255], dtype=np.uint8)
    return img


def draw_rect_inplace(img: np.ndarray, x0, y0, x1, y1, thickness=2):
    """Draw a red rectangle (in-place) on an RGB image."""
    H, W = img.shape[:2]
    x0 = int(max(0, min(W - 1, x0)))
    x1 = int(max(0, min(W, x1)))
    y0 = int(max(0, min(H - 1, y0)))
    y1 = int(max(0, min(H, y1)))
    if x1 <= x0 or y1 <= y0:
        return

    t = int(max(1, thickness))
    red = np.array([255, 0, 0], dtype=np.uint8)

    # top
    img[y0:min(H, y0 + t), x0:x1] = red
    # bottom
    img[max(0, y1 - t):y1, x0:x1] = red
    # left
    img[y0:y1, x0:min(W, x0 + t)] = red
    # right
    img[y0:y1, max(0, x1 - t):x1] = red


def save_rgb_png(img: np.ndarray, out_path: Path, dpi: int = 160):
    """Save an RGB image without axes."""
    ensure_dir(out_path.parent)
    H, W = img.shape[:2]
    plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_first_6s_events(dat_path: Path, chunk_us: int = 500_000) -> np.ndarray:
    """
    Load events from .dat but only keep first 6 seconds relative to first event timestamp.
    Uses repeated load_delta_t to avoid load_n_events(-1) edge cases.
    """
    video = PSEELoader(str(dat_path))

    chunks = []
    first_t = None
    last_t = None

    while True:
        ev = video.load_delta_t(chunk_us)
        if ev is None or len(ev) == 0:
            break

        if "t" not in ev.dtype.names:
            raise RuntimeError(f"Unexpected event dtype fields: {ev.dtype.names}")

        # establish first timestamp
        if first_t is None:
            first_t = int(ev["t"][0])
        # stop condition: keep only t <= first_t + CLIP_US
        t_limit = first_t + CLIP_US

        # filter this chunk
        m = ev["t"].astype(np.int64) <= t_limit
        if np.any(m):
            chunks.append(ev[m])

        # if this chunk already goes beyond limit, we can stop
        if int(ev["t"][-1]) >= t_limit:
            break

        # guard against non-progress
        if last_t is not None and int(ev["t"][-1]) <= last_t:
            break
        last_t = int(ev["t"][-1])

    if len(chunks) == 0:
        return np.zeros((0,), dtype=[("t", "i8")])

    ev_all = np.concatenate(chunks, axis=0)
    # sort by t
    order = np.argsort(ev_all["t"], kind="mergesort")
    return ev_all[order]


def iter_windows_by_time(t: np.ndarray, win_us: int):
    """Yield (frame_idx, t0, t1, slice) for events in [t0, t1). t must be sorted."""
    if t.size == 0:
        return
    t0 = int(t[0])
    t_end = int(t[-1])
    start = 0
    frame_idx = 0
    while t0 < t_end:
        t1 = t0 + win_us
        end = int(np.searchsorted(t, t1, side="left"))
        yield frame_idx, t0, t1, slice(start, end)
        frame_idx += 1
        t0 = t1
        start = end


def process_one(dat_path: Path, out_roi_root: Path, out_mark_root: Path, out_npz_root: Path) -> dict:
    stem = dat_path.stem

    ev = load_first_6s_events(dat_path)
    if ev.size == 0:
        # still create empty npz for consistency
        ensure_dir(out_npz_root)
        np.savez_compressed(out_npz_root / f"{stem}.npz",
                            t=np.array([], dtype=np.int64),
                            x=np.array([], dtype=np.int16),
                            y=np.array([], dtype=np.int16),
                            p=np.array([], dtype=np.int8))
        return {
            "file": stem,
            "frames": 0,
            "events_total": 0,
            "events_in_roi": 0,
            "roi_ratio": 0.0,
        }

    names = ev.dtype.names
    fx = field_pick(names, "x", "X")
    fy = field_pick(names, "y", "Y")
    fp = field_pick(names, "p", "polarity", "P")
    if fx is None or fy is None or fp is None:
        raise RuntimeError(f"Missing x/y/p fields in {stem}: {names}")

    t = ev["t"].astype(np.int64)
    x = ev[fx].astype(np.int32)
    y = ev[fy].astype(np.int32)
    p = ev[fp].astype(np.int16)

    # ROI mask on the 1280x720 coordinate system
    in_roi = (x >= X0) & (x < X1) & (y >= Y0) & (y < Y1)

    events_total = int(x.size)
    events_in_roi_total = int(in_roi.sum())
    roi_ratio = events_in_roi_total / max(1, events_total)

    # save ROI events as npz (original timestamps, not normalized)
    ensure_dir(out_npz_root)
    np.savez_compressed(
        out_npz_root / f"{stem}.npz",
        t=t[in_roi].astype(np.int64),
        x=x[in_roi].astype(np.int32),
        y=y[in_roi].astype(np.int32),
        p=p[in_roi].astype(np.int16),
        roi=np.array([X0, Y0, X1, Y1], dtype=np.int32),
        full_res=np.array([FULL_W, FULL_H], dtype=np.int32),
    )

    # render frames
    out_roi_dir = out_roi_root / stem
    out_mark_dir = out_mark_root / stem
    ensure_dir(out_roi_dir)
    ensure_dir(out_mark_dir)

    frame_count = 0
    for frame_idx, t0, t1, sl in iter_windows_by_time(t, WIN_US):
        if sl.start == sl.stop:
            # empty frame
            xw = np.empty((0,), dtype=np.int32)
            yw = np.empty((0,), dtype=np.int32)
            pw = np.empty((0,), dtype=np.int16)
            xw_roi = xw
            yw_roi = yw
            pw_roi = pw
        else:
            xw = x[sl]
            yw = y[sl]
            pw = p[sl]
            mroi = in_roi[sl]
            xw_roi = xw[mroi]
            yw_roi = yw[mroi]
            pw_roi = pw[mroi]

        # (A) ROI-only image (crop coordinates into ROI-local)
        if xw_roi.size == 0:
            roi_img = np.ones((ROI_H, ROI_W, 3), dtype=np.uint8) * 255
        else:
            roi_local_x = (xw_roi - X0).astype(np.int32)
            roi_local_y = (yw_roi - Y0).astype(np.int32)
            roi_img = render_events_to_rgb(roi_local_x, roi_local_y, pw_roi, ROI_W, ROI_H)

        save_rgb_png(roi_img, out_roi_dir / f"frame_{frame_idx:06d}.png")

        # (B) Full image with ROI red box
        full_img = render_events_to_rgb(xw, yw, pw, FULL_W, FULL_H)
        draw_rect_inplace(full_img, X0, Y0, X1, Y1, thickness=2)
        save_rgb_png(full_img, out_mark_dir / f"frame_{frame_idx:06d}.png")

        frame_count += 1

    return {
        "file": stem,
        "frames": frame_count,
        "events_total": events_total,
        "events_in_roi": events_in_roi_total,
        "roi_ratio": roi_ratio,
        "roi_x0": X0, "roi_y0": Y0, "roi_x1": X1, "roi_y1": Y1,
        "roi_w": ROI_W, "roi_h": ROI_H,
        "full_w": FULL_W, "full_h": FULL_H,
        "clip_us": CLIP_US,
        "win_us": WIN_US,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    proj_root = find_project_root(script_dir)

    dv_root = proj_root / "data" / "DV"
    out_roi_root = proj_root / "data" / "DVvis_roi"
    out_mark_root = proj_root / "data" / "DVvis_roi_marked"
    out_npz_root = proj_root / "data" / "DV_roi_npz"

    print("=" * 80)
    print(f"[INFO] script    : {Path(__file__).resolve()}")
    print(f"[INFO] proj_root : {proj_root}")
    print(f"[INFO] DV root   : {dv_root}")
    print(f"[INFO] OUT roi   : {out_roi_root}")
    print(f"[INFO] OUT marked: {out_mark_root}")
    print(f"[INFO] OUT npz   : {out_npz_root}")
    print(f"[INFO] ROI box   : x[{X0},{X1}) y[{Y0},{Y1})  size=({ROI_W},{ROI_H})  center=({ROI_CX},{ROI_CY})")
    print(f"[INFO] FULL res  : ({FULL_W},{FULL_H})")
    print(f"[INFO] CLIP      : first {CLIP_US/1e6:.1f}s, WIN={WIN_US/1000:.1f}ms")
    print("=" * 80)

    if not dv_root.is_dir():
        raise FileNotFoundError(f"DV folder not found: {dv_root}")

    dat_files = sorted(dv_root.glob("*.dat"))
    print(f"[SCAN] {dv_root}  dat_files={len(dat_files)}")
    if len(dat_files) == 0:
        print("[WARN] No .dat files found.")
        return

    ensure_dir(out_roi_root)
    ensure_dir(out_mark_root)
    ensure_dir(out_npz_root)

    summary_path = out_roi_root / "summary.csv"
    rows = []

    for i, p in enumerate(dat_files, 1):
        print(f"[{i:3d}/{len(dat_files)}] {p.stem}")
        s = process_one(p, out_roi_root, out_mark_root, out_npz_root)
        rows.append(s)

    # write summary
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=" * 80)
    print(f"[DONE] files={len(rows)}  summary={summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
