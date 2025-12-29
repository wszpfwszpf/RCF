# -*- coding: utf-8 -*-
"""
auto_roi.py
-----------
Robust automatic ROI selection for XTUDV-like DV .dat files.

- Input : <proj_root>/data/DV/*.dat
- Output: <proj_root>/outputs/beam_roi/
    - roi_summary.csv
    - <sample_name>/roi.json
    - <sample_name>/roi_overlay.png
    - <sample_name>/heatmap.png

Run from anywhere:
    python beam/auto_roi.py
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Your project loader
from utils.io.psee_loader import PSEELoader


# -----------------------------
# Config
# -----------------------------

@dataclass
class RoiConfig:
    # Sensor resolution (you said 1280x720)
    W: int = 1280
    H: int = 720

    # Read events in chunks (microseconds)
    chunk_us: int = 200_000  # 0.2s per chunk, safe and not too slow

    # Build heatmap at a coarse grid to be robust to noise and speed up
    bin_px: int = 4  # 4px bin => heatmap size ~ (320,180)

    # Use only a time segment (optional). Set None to use full stream.
    # If your files have "quiet->impulse->decay", you can keep full stream first for stability.
    use_time_window: bool = False
    t0_us: int = 0
    t1_us: int = 0  # ignored if use_time_window=False

    # Remove very "static strong edges" area (e.g., right vertical bar).
    # We mask columns with extremely high activity.
    enable_column_mask: bool = True
    column_mask_topk_frac: float = 0.01  # top 1% columns are candidates
    column_mask_ratio_thr: float = 6.0   # if a column sum is > thr * median, mask it

    # ROI selection on heatmap:
    # Keep pixels above quantile and take largest connected component bbox.
    active_quantile: float = 0.995  # 99.5% quantile threshold (tighter -> smaller ROI)
    min_component_area_bins: int = 200  # min area in binned cells

    # Expand ROI
    margin_px: int = 40

    # Enforce ROI minimum size (avoid too small ROI)
    min_roi_w: int = 160
    min_roi_h: int = 120

    # Enforce ROI maximum size (optional)
    max_roi_w: int = 900
    max_roi_h: int = 600

    # Visualization: number of frames to draw overlay from a short excerpt
    overlay_win_us: int = 33_000  # 33ms
    overlay_n_frames: int = 6

    # Polarity colors (for overlay)
    pos_color = (1.0, 0.0, 0.0)  # red
    neg_color = (0.0, 0.0, 1.0)  # blue


# -----------------------------
# Utilities
# -----------------------------

def _proj_root_from_beam() -> Path:
    """beam/auto_roi.py -> proj_root"""
    return Path(__file__).resolve().parents[1]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _basename_no_suffix(p: Path) -> str:
    return p.name.rsplit(".", 1)[0]

def _robust_median(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.median(x))

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x0,y0,x1,y1) in mask coordinate (x1,y1 exclusive)."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    x0 = int(xs.min()); x1 = int(xs.max()) + 1
    y0 = int(ys.min()); y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


# -----------------------------
# Robust event loading
# -----------------------------

def load_all_events_by_dt(video: PSEELoader, chunk_us: int) -> Dict[str, np.ndarray]:
    """
    Robustly load all events by repeatedly calling load_delta_t(chunk_us).

    Returns dict with keys: t,x,y,p (numpy arrays)
    - 't' expected in microseconds (int) per your DV loader usage.
    """
    ts, xs, ys, ps = [], [], [], []
    n_total = 0

    while True:
        ev = video.load_delta_t(chunk_us)
        if ev is None or len(ev) == 0:
            break
        ts.append(ev["t"].astype(np.int64, copy=False))
        xs.append(ev["x"].astype(np.int32, copy=False))
        ys.append(ev["y"].astype(np.int32, copy=False))
        ps.append(ev["p"].astype(np.int8, copy=False))
        n_total += len(ev)

        # safety guard against corrupted files
        if n_total > 300_000_000:
            raise RuntimeError("Too many events loaded; possible corrupted file or infinite loop.")

    if n_total == 0:
        return {
            "t": np.array([], dtype=np.int64),
            "x": np.array([], dtype=np.int32),
            "y": np.array([], dtype=np.int32),
            "p": np.array([], dtype=np.int8),
        }

    return {
        "t": np.concatenate(ts),
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "p": np.concatenate(ps),
    }


def time_filter(ev: Dict[str, np.ndarray], t0_us: int, t1_us: int) -> Dict[str, np.ndarray]:
    if ev["t"].size == 0:
        return ev
    m = (ev["t"] >= t0_us) & (ev["t"] <= t1_us)
    return {k: v[m] for k, v in ev.items()}


# -----------------------------
# Heatmap + ROI
# -----------------------------

def build_activity_heatmap(ev: Dict[str, np.ndarray], W: int, H: int, bin_px: int) -> np.ndarray:
    """
    Build activity heatmap (counts) on binned grid.
    heatmap shape: (Hb, Wb) with Hb = ceil(H/bin_px), Wb = ceil(W/bin_px)
    """
    x = ev["x"]; y = ev["y"]
    if x.size == 0:
        Hb = int(math.ceil(H / bin_px))
        Wb = int(math.ceil(W / bin_px))
        return np.zeros((Hb, Wb), dtype=np.int64)

    # clamp to sensor range (defensive)
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    xb = (x // bin_px).astype(np.int32)
    yb = (y // bin_px).astype(np.int32)

    Hb = int(math.ceil(H / bin_px))
    Wb = int(math.ceil(W / bin_px))

    heat = np.zeros((Hb, Wb), dtype=np.int64)
    np.add.at(heat, (yb, xb), 1)
    return heat


def mask_strong_columns(heat: np.ndarray, cfg: RoiConfig) -> np.ndarray:
    """
    Mask (set to 0) extremely strong columns to reduce domination of static vertical edges.
    Strategy:
      - compute column sums
      - find columns whose sum is > cfg.column_mask_ratio_thr * median
      - additionally restrict to top-k fraction by sum
    """
    heat2 = heat.copy()
    col_sum = heat2.sum(axis=0)
    med = _robust_median(col_sum)
    if med <= 0:
        return heat2

    # top-k threshold
    k = max(1, int(cfg.column_mask_topk_frac * col_sum.size))
    topk_thr = np.partition(col_sum, -k)[-k]

    bad = (col_sum >= topk_thr) & (col_sum > cfg.column_mask_ratio_thr * med)
    if np.any(bad):
        heat2[:, bad] = 0
    return heat2


def largest_connected_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Find largest 4-connected component in a binary mask.
    Return component mask. If none meets min_area, return empty mask.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    best_area = 0
    best_pixels = None

    # simple BFS/stack, no external deps
    for yy in range(H):
        for xx in range(W):
            if not mask[yy, xx] or visited[yy, xx]:
                continue
            stack = [(yy, xx)]
            visited[yy, xx] = True
            pixels = [(yy, xx)]
            while stack:
                y0, x0 = stack.pop()
                for ny, nx in ((y0 - 1, x0), (y0 + 1, x0), (y0, x0 - 1), (y0, x0 + 1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        pixels.append((ny, nx))
            area = len(pixels)
            if area > best_area:
                best_area = area
                best_pixels = pixels

    if best_pixels is None or best_area < min_area:
        return np.zeros_like(mask, dtype=bool)

    comp = np.zeros_like(mask, dtype=bool)
    ys, xs = zip(*best_pixels)
    comp[np.array(ys), np.array(xs)] = True
    return comp


def estimate_roi_from_events(ev: Dict[str, np.ndarray], cfg: RoiConfig) -> Dict[str, int]:
    """
    Core ROI estimator:
      1) activity heatmap (binned)
      2) optional strong-column masking
      3) threshold by quantile
      4) largest connected component
      5) bbox -> pixel coords
      6) expand margin and apply min/max constraints
    """
    heat = build_activity_heatmap(ev, cfg.W, cfg.H, cfg.bin_px)
    heat_for_roi = heat

    if cfg.enable_column_mask:
        heat_for_roi = mask_strong_columns(heat_for_roi, cfg)

    # quantile threshold
    flat = heat_for_roi.flatten()
    if flat.size == 0:
        thr = 0
    else:
        # ignore zeros to avoid thr=0 when sparse
        nz = flat[flat > 0]
        if nz.size == 0:
            thr = 0
        else:
            thr = float(np.quantile(nz, cfg.active_quantile))

    active = heat_for_roi >= thr
    if thr <= 0:
        # fallback: use non-zero area
        active = heat_for_roi > 0

    comp = largest_connected_component(active, cfg.min_component_area_bins)
    if not comp.any():
        # fallback: take bbox of all active pixels (still may be empty)
        comp = active

    xb0, yb0, xb1, yb1 = _bbox_from_mask(comp)

    # Convert binned bbox to pixel bbox
    x0 = xb0 * cfg.bin_px
    x1 = xb1 * cfg.bin_px
    y0 = yb0 * cfg.bin_px
    y1 = yb1 * cfg.bin_px

    # Expand margin
    x0 -= cfg.margin_px
    y0 -= cfg.margin_px
    x1 += cfg.margin_px
    y1 += cfg.margin_px

    # Clamp to sensor
    x0 = _clamp(int(x0), 0, cfg.W - 1)
    y0 = _clamp(int(y0), 0, cfg.H - 1)
    x1 = _clamp(int(x1), x0 + 1, cfg.W)
    y1 = _clamp(int(y1), y0 + 1, cfg.H)

    # Enforce min size around center
    roi_w = x1 - x0
    roi_h = y1 - y0
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    target_w = int(np.clip(cfg.min_roi_w, 1, cfg.W))
    target_h = int(np.clip(cfg.min_roi_h, 1, cfg.H))

    if roi_w < target_w:
        half = target_w // 2
        x0 = _clamp(cx - half, 0, cfg.W - target_w)
        x1 = x0 + target_w
    if roi_h < target_h:
        half = target_h // 2
        y0 = _clamp(cy - half, 0, cfg.H - target_h)
        y1 = y0 + target_h

    # Enforce max size around center
    if cfg.max_roi_w > 0 and (x1 - x0) > cfg.max_roi_w:
        half = cfg.max_roi_w // 2
        x0 = _clamp(cx - half, 0, cfg.W - cfg.max_roi_w)
        x1 = x0 + cfg.max_roi_w
    if cfg.max_roi_h > 0 and (y1 - y0) > cfg.max_roi_h:
        half = cfg.max_roi_h // 2
        y0 = _clamp(cy - half, 0, cfg.H - cfg.max_roi_h)
        y1 = y0 + cfg.max_roi_h

    return {
        "roi_x0": int(x0),
        "roi_y0": int(y0),
        "roi_x1": int(x1),
        "roi_y1": int(y1),
    }


# -----------------------------
# Visualization helpers
# -----------------------------

def render_event_frame(ev: Dict[str, np.ndarray], W: int, H: int,
                       x0: int, y0: int, x1: int, y1: int,
                       pos_rgb=(1.0, 0.0, 0.0), neg_rgb=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Render a simple RGB frame (white background) for events inside ROI.
    """
    img = np.ones((y1 - y0, x1 - x0, 3), dtype=np.float32)

    if ev["t"].size == 0:
        return img

    x = ev["x"]; y = ev["y"]; p = ev["p"]
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    if not np.any(m):
        return img

    x = x[m] - x0
    y = y[m] - y0
    p = p[m]

    # polarity convention in your loader might be {0,1} or {-1,+1}
    # We'll treat p>0 as positive.
    pos = p > 0
    neg = ~pos

    img[y[pos], x[pos], :] = pos_rgb
    img[y[neg], x[neg], :] = neg_rgb
    return img


def save_heatmap_png(heat: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.imshow(heat, aspect="auto")
    plt.title("Activity heatmap (binned)")
    plt.xlabel("x bins")
    plt.ylabel("y bins")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_roi_overlay(ev: Dict[str, np.ndarray], cfg: RoiConfig,
                     roi: Dict[str, int], out_png: Path) -> None:
    """
    Save a montage of a few frames (33ms accumulation each) inside ROI.
    """
    x0, y0, x1, y1 = roi["roi_x0"], roi["roi_y0"], roi["roi_x1"], roi["roi_y1"]
    if ev["t"].size == 0:
        # save blank
        img = np.ones((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.float32)
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.title("EMPTY events")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        return

    tmin = int(ev["t"].min())
    tmax = int(ev["t"].max())

    # pick frames starting near the beginning, but skip a small offset to avoid initial empty area
    start = tmin + int(0.05 * (tmax - tmin))
    win = int(cfg.overlay_win_us)

    n = int(cfg.overlay_n_frames)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(cols * 4, rows * 3))
    for i in range(n):
        t0 = start + i * win
        t1 = t0 + win
        m = (ev["t"] >= t0) & (ev["t"] < t1)
        sub = {k: v[m] for k, v in ev.items()}
        img = render_event_frame(
            sub, cfg.W, cfg.H, x0, y0, x1, y1,
            pos_rgb=cfg.pos_color, neg_rgb=cfg.neg_color
        )
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"{(t0 - tmin)/1e6:.3f}s~{(t1 - tmin)/1e6:.3f}s")
        ax.axis("off")

    plt.suptitle(f"ROI overlay | x[{x0},{x1}) y[{y0},{y1})", y=0.98)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Main pipeline per file
# -----------------------------

def process_one_dat(dat_path: Path, out_root: Path, cfg: RoiConfig) -> Dict[str, object]:
    name = _basename_no_suffix(dat_path)
    sample_dir = out_root / name
    _ensure_dir(sample_dir)

    # Load events robustly
    video = PSEELoader(str(dat_path))
    ev = load_all_events_by_dt(video, cfg.chunk_us)

    if ev["t"].size == 0:
        # Still dump a minimal record
        roi = {"roi_x0": 0, "roi_y0": 0, "roi_x1": 0, "roi_y1": 0}
        rec = {
            "name": name,
            "dat": str(dat_path),
            "n_events_total": 0,
            "t_start_us": 0,
            "t_end_us": 0,
            **roi,
            "roi_w": 0,
            "roi_h": 0,
            "note": "EMPTY",
        }
        with open(sample_dir / "roi.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    # Optional time window
    if cfg.use_time_window:
        ev_used = time_filter(ev, cfg.t0_us, cfg.t1_us)
        if ev_used["t"].size == 0:
            ev_used = ev
    else:
        ev_used = ev

    roi = estimate_roi_from_events(ev_used, cfg)

    # Save artifacts
    heat = build_activity_heatmap(ev_used, cfg.W, cfg.H, cfg.bin_px)
    if cfg.enable_column_mask:
        heat2 = mask_strong_columns(heat, cfg)
    else:
        heat2 = heat
    save_heatmap_png(heat2, sample_dir / "heatmap.png")
    save_roi_overlay(ev_used, cfg, roi, sample_dir / "roi_overlay.png")

    t_start = int(ev["t"].min())
    t_end = int(ev["t"].max())
    roi_w = int(roi["roi_x1"] - roi["roi_x0"])
    roi_h = int(roi["roi_y1"] - roi["roi_y0"])

    # Count events inside ROI (quality check)
    m_roi = (
        (ev["x"] >= roi["roi_x0"]) & (ev["x"] < roi["roi_x1"]) &
        (ev["y"] >= roi["roi_y0"]) & (ev["y"] < roi["roi_y1"])
    )
    n_roi = int(np.count_nonzero(m_roi))

    rec = {
        "name": name,
        "dat": str(dat_path),
        "n_events_total": int(ev["t"].size),
        "n_events_in_roi": n_roi,
        "t_start_us": t_start,
        "t_end_us": t_end,
        **roi,
        "roi_w": roi_w,
        "roi_h": roi_h,
        "note": "",
    }

    with open(sample_dir / "roi.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    return rec


def write_csv(records: List[Dict[str, object]], out_csv: Path) -> None:
    # minimal CSV writer (no pandas dependency)
    if not records:
        return
    keys = list(records[0].keys())
    lines = [",".join(keys)]
    for r in records:
        row = []
        for k in keys:
            v = r.get(k, "")
            s = str(v).replace(",", ";")
            row.append(s)
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def main():
    cfg = RoiConfig()

    proj_root = _proj_root_from_beam()
    dv_root = proj_root / "data" / "DV"
    out_root = proj_root / "outputs" / "beam_roi"

    print("=" * 80)
    print(f"[INFO] script : {Path(__file__).resolve()}")
    print(f"[INFO] proj   : {proj_root}")
    print(f"[INFO] DV root: {dv_root}")
    print(f"[INFO] OUT    : {out_root}")
    print("=" * 80)

    if not dv_root.exists():
        raise FileNotFoundError(f"DV root not found: {dv_root}")

    _ensure_dir(out_root)
    dat_files = sorted(dv_root.glob("*.dat"))

    print(f"[SCAN] {dv_root}  dat_files={len(dat_files)}")
    if len(dat_files) == 0:
        return

    records = []
    for i, p in enumerate(dat_files, 1):
        name = _basename_no_suffix(p)
        print(f"[{i:3d}/{len(dat_files)}] {name}")
        try:
            rec = process_one_dat(p, out_root, cfg)
            records.append(rec)
        except Exception as e:
            print(f"[ERROR] {name}: {repr(e)}")
            records.append({
                "name": name,
                "dat": str(p),
                "n_events_total": 0,
                "n_events_in_roi": 0,
                "t_start_us": 0,
                "t_end_us": 0,
                "roi_x0": 0, "roi_y0": 0, "roi_x1": 0, "roi_y1": 0,
                "roi_w": 0, "roi_h": 0,
                "note": f"ERROR: {type(e).__name__}",
            })

    out_csv = out_root / "roi_summary.csv"
    write_csv(records, out_csv)
    print("=" * 80)
    print(f"[DONE] processed={len(records)}")
    print(f"[OUT ] {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
