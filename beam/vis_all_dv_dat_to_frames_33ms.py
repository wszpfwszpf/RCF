# beam/vis_all_dv_dat_to_frames_33ms.py
# -*- coding: utf-8 -*-
"""
Visualize all .dat files under data/DV by accumulating events into frames every 33ms.

Input : <proj_root>/data/DV/*.dat
Output: <proj_root>/data/DVvis/<dat_stem>/frame_000000.png ...

Notes:
- This script is placed under beam/, not project root.
- We locate proj_root by going up one level from this script: beam/ -> proj_root.
- DV .dat is read via official PSEELoader (beam/utils/io/psee_loader.py).
- Frame background is white; polarity: ON(red) / OFF(blue) by default.
"""

from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# --- Use the same loader you already fixed (beam/utils/io/psee_loader.py) ---
from beam.utils.io.psee_loader import PSEELoader


# ----------------------------
# Config
# ----------------------------
WIN_US = 33_000  # 33ms in microseconds

# If you want fixed resolution, set (W,H). If None, infer from events per-file.
# FORCE_RESOLUTION: Optional[Tuple[int, int]] = None
FORCE_RESOLUTION = (1280, 720)

# Max frames to export per file (None = all)
# MAX_FRAMES_PER_FILE: Optional[int] = None
MAX_FRAMES_PER_FILE: Optional[int] = 150

# If True, recursively scan data/DV/**.dat
RECURSIVE = False

# Visualization style
POINT_ALPHA = 1.0  # scatter alpha; kept but we render as image, not scatter
DPI = 150


# ----------------------------
# Path helpers
# ----------------------------
def get_proj_root() -> Path:
    # beam/vis_xxx.py -> parents[1] is project root
    return Path(__file__).resolve().parents[1]


def list_dat_files(dv_dir: Path) -> list[Path]:
    if RECURSIVE:
        return sorted([p for p in dv_dir.rglob("*.dat") if p.is_file()])
    else:
        return sorted([p for p in dv_dir.glob("*.dat") if p.is_file()])


# ----------------------------
# Event accumulation -> frame
# ----------------------------
def infer_resolution_from_events(ev: np.ndarray) -> Tuple[int, int]:
    # ev['x'], ev['y'] are 0-based
    x = ev["x"].astype(np.int32)
    y = ev["y"].astype(np.int32)
    W = int(x.max()) + 1 if x.size > 0 else 1
    H = int(y.max()) + 1 if y.size > 0 else 1
    return W, H


def accumulate_to_rgb_frame(ev: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Create white background RGB image.
    ON events (p>0) -> red
    OFF events (p<=0) -> blue

    We render as an image to be fast and consistent for video making.
    """
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    if ev is None or len(ev) == 0:
        return img

    x = ev["x"].astype(np.int32)
    y = ev["y"].astype(np.int32)
    p = ev["p"]

    # normalize polarity to boolean
    pos = p > 0
    neg = ~pos

    # clamp just in case
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    # red for pos
    img[y[pos], x[pos], 0] = 255
    img[y[pos], x[pos], 1] = 0
    img[y[pos], x[pos], 2] = 0

    # blue for neg
    img[y[neg], x[neg], 0] = 0
    img[y[neg], x[neg], 1] = 0
    img[y[neg], x[neg], 2] = 255

    return img


def suggest_crop_box_from_heatmap(heat: np.ndarray, margin: int = 2) -> Optional[Tuple[int, int, int, int]]:
    """
    heat: (H,W) counts
    return (x0,y0,x1,y1) inclusive bounds of non-zero region with margin
    """
    ys, xs = np.nonzero(heat > 0)
    if xs.size == 0:
        return None
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, heat.shape[1] - 1)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin, heat.shape[0] - 1)
    return x0, y0, x1, y1


# ----------------------------
# Main per-file processing
# ----------------------------
def process_one_dat(dat_path: Path, out_root: Path) -> Dict:
    stem = dat_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    video = PSEELoader(str(dat_path))

    # We'll stream in 33ms windows using load_delta_t(WIN_US).
    # Note: PSEELoader internally advances cursor each call.
    n_frames = 0
    n_events_total = 0

    inferred_W, inferred_H = None, None

    # For crop suggestion: accumulate a global heatmap over first N frames to avoid huge memory.
    heat_acc = None
    HEAT_FRAMES = 50  # only use early frames for coarse ROI suggestion

    while True:
        ev = video.load_delta_t(WIN_US)
        if ev is None or len(ev) == 0:
            break

        if FORCE_RESOLUTION is None:
            if inferred_W is None:
                inferred_W, inferred_H = infer_resolution_from_events(ev)
        else:
            inferred_W, inferred_H = FORCE_RESOLUTION

        W, H = inferred_W, inferred_H

        # heatmap accumulate for ROI suggestion
        if heat_acc is None:
            heat_acc = np.zeros((H, W), dtype=np.int32)
        if n_frames < HEAT_FRAMES:
            x = np.clip(ev["x"].astype(np.int32), 0, W - 1)
            y = np.clip(ev["y"].astype(np.int32), 0, H - 1)
            np.add.at(heat_acc, (y, x), 1)

        # render frame
        frame = accumulate_to_rgb_frame(ev, W, H)

        # save png
        out_png = out_dir / f"frame_{n_frames:06d}.png"
        plt.imsave(str(out_png), frame)

        n_events_total += int(len(ev))
        n_frames += 1

        if MAX_FRAMES_PER_FILE is not None and n_frames >= MAX_FRAMES_PER_FILE:
            break

    # suggest crop box
    crop_box = suggest_crop_box_from_heatmap(heat_acc) if heat_acc is not None else None

    meta = {
        "dat": str(dat_path),
        "out_dir": str(out_dir),
        "frames": n_frames,
        "events_total": n_events_total,
        "W": inferred_W if inferred_W is not None else "",
        "H": inferred_H if inferred_H is not None else "",
        "crop_x0": crop_box[0] if crop_box else "",
        "crop_y0": crop_box[1] if crop_box else "",
        "crop_x1": crop_box[2] if crop_box else "",
        "crop_y1": crop_box[3] if crop_box else "",
    }
    return meta


def main():
    proj_root = get_proj_root()
    dv_dir = proj_root / "data" / "DV"
    out_root = proj_root / "data" / "DVvis"
    out_root.mkdir(parents=True, exist_ok=True)

    dat_files = list_dat_files(dv_dir)
    print("=" * 80)
    print("[INFO] script   :", Path(__file__).resolve())
    print("[INFO] proj_root:", proj_root)
    print("[INFO] DV dir   :", dv_dir)
    print("[INFO] OUT dir  :", out_root)
    print("[INFO] dat_files:", len(dat_files))
    print("[INFO] WIN_US   :", WIN_US)
    print("[INFO] FORCE_RES:", FORCE_RESOLUTION)
    print("[INFO] MAX_FRAMES_PER_FILE:", MAX_FRAMES_PER_FILE)
    print("=" * 80)

    if len(dat_files) == 0:
        raise RuntimeError("No .dat files found under data/DV")

    metas = []
    for i, dat_path in enumerate(dat_files, 1):
        print(f"[{i:02d}/{len(dat_files)}] Processing {dat_path.name}")
        meta = process_one_dat(dat_path, out_root)
        print(f"    -> frames={meta['frames']}  events_total={meta['events_total']}  W,H=({meta['W']},{meta['H']})")
        if meta["crop_x0"] != "":
            print(f"    -> crop_suggest: x[{meta['crop_x0']},{meta['crop_x1']}], y[{meta['crop_y0']},{meta['crop_y1']}]")
        metas.append(meta)

    # write summary csv
    summary_csv = out_root / "DVvis_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["dat", "out_dir", "frames", "events_total", "W", "H",
                "crop_x0", "crop_y0", "crop_x1", "crop_y1"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in metas:
            w.writerow({k: m.get(k, "") for k in cols})

    print("=" * 80)
    print("[DONE] Saved frames into:", out_root)
    print("[DONE] Summary CSV      :", summary_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
