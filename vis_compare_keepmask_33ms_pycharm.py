import os
import glob
from pathlib import Path

import numpy as np

# ============================================================
# 0) 配置区（PyCharm 一键运行只改这里）
# ============================================================

# 输入 AEDAT4 文件（单个文件）
AEDAT4_PATH = r"data/emlb/day/Car_Day/Car_Day-ND64-3.aedat4"

# 四种方法 keepmask 根目录（支持递归搜索）
KEEP_DWF_ROOT    = r"data/emlb_dwf_verify/keepmask"
KEEP_TS_ROOT     = r"data/emlb_ts_verify/keepmask"
KEEP_YNOISE_ROOT = r"data/emlb_ynoise_verify/keepmask"
KEEP_RCF_ROOT    = r"data/emlb_rcf_verify/keepmask"

# 输出目录（会在里面创建 raw/dwf/ts/ynoise/rcf/pinjie）
OUT_DIR = r"baselines/vis_compare2"
# RCF 选择的 eta
RCF_ETA = 0.1
RCF_ETA_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 如你存盘顺序不同，就改这里

# 传感器分辨率（E-MLB 常见为 DAVIS346）
SENSOR_W = 346
SENSOR_H = 260

# 可视化帧设置：33ms × 30帧 ≈ 1秒
DT_US = 33_000
N_FRAMES = 60




# ============================================================
# 1) AEDAT4 读取（dv-processing bindings，兼容无 xs()/ys()）
# ============================================================
def read_aedat4_events(aedat4_path: str):
    """
    Return events as numpy arrays:
      t_us (int64), x (int32), y (int32), p (int8 in {-1,+1})
    Compatible with dv_processing versions where EventStore has no xs()/ys().
    """
    aedat4_path = str(aedat4_path)

    try:
        import dv_processing as dv  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import dv_processing. Please install dv-processing python bindings in this env."
        ) from e

    reader = dv.io.MonoCameraRecording(aedat4_path)
    if hasattr(reader, "isEventStreamAvailable") and (not reader.isEventStreamAvailable()):
        raise RuntimeError(f"No event stream available in: {aedat4_path}")

    t_all, x_all, y_all, p_all = [], [], [], []

    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            continue

        # Fast path: events.numpy()/toNumpy()/asNumpy() -> structured array
        arr = None
        for fn in ("numpy", "toNumpy", "asNumpy"):
            if hasattr(events, fn):
                try:
                    arr = getattr(events, fn)()
                    break
                except Exception:
                    arr = None

        if isinstance(arr, np.ndarray) and arr.dtype.names:
            names = set(arr.dtype.names)
            if "x" in names and "y" in names and ("t" in names or "timestamp" in names):
                x = arr["x"].astype(np.int32, copy=False)
                y = arr["y"].astype(np.int32, copy=False)
                t = (arr["t"] if "t" in names else arr["timestamp"]).astype(np.int64, copy=False)

                if "p" in names:
                    pol = arr["p"]
                elif "polarity" in names:
                    pol = arr["polarity"]
                elif "pol" in names:
                    pol = arr["pol"]
                else:
                    pol = np.ones_like(x, dtype=np.int8)

                pol = np.asarray(pol)
                if pol.dtype != np.int8:
                    pol = pol.astype(np.int8, copy=False)
                if pol.min() >= 0:
                    pol = np.where(pol > 0, 1, -1).astype(np.int8, copy=False)

                t_all.append(t)
                x_all.append(x)
                y_all.append(y)
                p_all.append(pol)
                continue

        # Fallback: iterate events (robust but slower)
        x_list, y_list, t_list, p_list = [], [], [], []
        for e in events:
            ex = getattr(e, "x", None); ex = ex() if callable(ex) else ex
            ey = getattr(e, "y", None); ey = ey() if callable(ey) else ey

            et = getattr(e, "timestamp", None); et = et() if callable(et) else et
            if et is None:
                et = getattr(e, "t", None); et = et() if callable(et) else et

            ep = getattr(e, "polarity", None); ep = ep() if callable(ep) else ep
            if ep is None:
                ep = getattr(e, "p", None); ep = ep() if callable(ep) else ep
            if ep is None:
                ep = 1

            if ex is None or ey is None or et is None:
                continue

            x_list.append(int(ex))
            y_list.append(int(ey))
            t_list.append(int(et))

            ep = int(ep)
            if ep in (0, 1):
                ep = 1 if ep == 1 else -1
            elif ep not in (-1, 1):
                ep = 1 if ep > 0 else -1
            p_list.append(ep)

        if len(t_list) > 0:
            t_all.append(np.asarray(t_list, dtype=np.int64))
            x_all.append(np.asarray(x_list, dtype=np.int32))
            y_all.append(np.asarray(y_list, dtype=np.int32))
            p_all.append(np.asarray(p_list, dtype=np.int8))

    if not t_all:
        return (np.empty((0,), np.int64),
                np.empty((0,), np.int32),
                np.empty((0,), np.int32),
                np.empty((0,), np.int8))

    t = np.concatenate(t_all)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    return t, x, y, p


# ============================================================
# 2) keepmask 查找与加载
# ============================================================
def find_keepmask_file(root_dir: str, aedat4_path: str):
    """
    Search keepmask file under root_dir that matches AEDAT4 stem.
    Accepts .npy / .npz / .csv.
    """
    key = Path(aedat4_path).stem
    root_dir = str(root_dir)

    patterns = [
        f"**/*{key}*keepmask*.npy",
        f"**/*{key}*keepmask*.npz",
        f"**/*{key}*keepmask*.csv",
        f"**/*{key}*mask*.npy",
        f"**/*{key}*mask*.npz",
        f"**/*{key}*mask*.csv",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))

    if not candidates:
        candidates = glob.glob(os.path.join(root_dir, "**/*keepmask*.npy"), recursive=True) + \
                     glob.glob(os.path.join(root_dir, "**/*keepmask*.npz"), recursive=True) + \
                     glob.glob(os.path.join(root_dir, "**/*keepmask*.csv"), recursive=True)

    if not candidates:
        raise FileNotFoundError(f"No keepmask found under: {root_dir} for key='{key}'")

    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


def load_keepmask(path: str) -> np.ndarray:
    """
    Load keepmask:
      - .npy: array
      - .npz: key 'keepmask' preferred, else first array
      - .csv: numeric columns (1D or 2D)
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        return np.asarray(np.load(path))

    if ext == ".npz":
        data = np.load(path)
        if "keepmask" in data.files:
            return np.asarray(data["keepmask"])
        return np.asarray(data[data.files[0]])

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if "keepmask" in df.columns:
            return df["keepmask"].to_numpy()
        if "keep" in df.columns:
            return df["keep"].to_numpy()
        num = df.select_dtypes(include=["number"]).to_numpy()
        return num

    raise ValueError(f"Unsupported keepmask format: {path}")


def pick_rcf_eta_column(km: np.ndarray, eta: float, eta_list):
    """
    RCF keepmask can be shape (N, 6). Select column corresponding to eta.
    """
    km = np.asarray(km)
    if km.ndim == 1:
        return km.astype(np.uint8)
    if km.ndim != 2:
        raise ValueError(f"Unexpected RCF keepmask shape: {km.shape}")

    idx = int(np.argmin(np.abs(np.asarray(eta_list, dtype=np.float64) - float(eta))))
    if idx >= km.shape[1]:
        raise ValueError(f"RCF keepmask columns={km.shape[1]} but need idx={idx} for eta={eta}")
    return km[:, idx].astype(np.uint8)


# ============================================================
# 3) 渲染与保存
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def render_event_frame_png(x, y, p, W, H, out_path):
    """
    White background RGB:
      p>0 -> red
      p<0 -> blue
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    if x.size > 0:
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        p = p.astype(np.int8)

        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x = x[valid]; y = y[valid]; p = p[valid]

        pos = (p > 0)
        neg = ~pos

        # red
        img[y[pos], x[pos], 0] = 255
        img[y[pos], x[pos], 1] = 0
        img[y[pos], x[pos], 2] = 0
        # blue
        img[y[neg], x[neg], 0] = 0
        img[y[neg], x[neg], 1] = 0
        img[y[neg], x[neg], 2] = 255

    from imageio.v2 import imwrite
    imwrite(out_path, img)



def stitch_1x5_with_label(
    paths_in_order,
    labels,
    out_path,
    border_px=3,
    label_height=36,
    font_size=22
):
    """
    Stitch 5 images into 1x5 with:
      - black border around each sub-image
      - label text on top (Times New Roman)
    Order: raw, ts, ynoise, dwf, rcf
    """
    from PIL import Image, ImageDraw, ImageFont

    assert len(paths_in_order) == 5
    assert len(labels) == 5

    imgs = [Image.open(p).convert("RGB") for p in paths_in_order]

    # --- load Times New Roman ---
    font_path_candidates = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
    ]
    font = None
    for fp in font_path_candidates:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
    if font is None:
        font = ImageFont.load_default()
        print("[WARN] Times New Roman not found, using default font.")

    W, H = imgs[0].size
    H_total = H + label_height + 2 * border_px
    W_total = 5 * (W + 2 * border_px)

    canvas = Image.new("RGB", (W_total, H_total), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x0 = i * (W + 2 * border_px)
        y0 = 0

        # --- draw border ---
        draw.rectangle(
            [
                x0,
                y0,
                x0 + W + 2 * border_px - 1,
                H_total - 1,
            ],
            outline=(0, 0, 0),
            width=border_px,
        )

        # --- draw label ---
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x0 + (W + 2 * border_px - text_w) // 2
        text_y = (label_height - font_size) // 2

        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

        # --- paste image ---
        canvas.paste(
            img,
            (x0 + border_px, label_height + border_px)
        )

    canvas.save(out_path)

# ============================================================
# 4) 主流程（自动裁剪到最小长度 + pinjie）
# ============================================================
def run():
    # Output folders
    dir_raw = os.path.join(OUT_DIR, "raw")
    dir_dwf = os.path.join(OUT_DIR, "dwf")
    dir_ts = os.path.join(OUT_DIR, "ts")
    dir_yn = os.path.join(OUT_DIR, "ynoise")
    dir_rcf = os.path.join(OUT_DIR, "rcf")
    dir_pinjie = os.path.join(OUT_DIR, "pinjie")
    for d in [dir_raw, dir_dwf, dir_ts, dir_yn, dir_rcf, dir_pinjie]:
        ensure_dir(d)

    # Load events
    t_us, x, y, p = read_aedat4_events(AEDAT4_PATH)
    if t_us.size == 0:
        raise RuntimeError("Empty event stream from AEDAT4.")

    # Normalize time to start at 0 (microseconds)
    t_us = t_us - t_us.min()

    # Locate keepmasks by AEDAT4 stem
    km_dwf_path = find_keepmask_file(KEEP_DWF_ROOT, AEDAT4_PATH)
    km_ts_path  = find_keepmask_file(KEEP_TS_ROOT, AEDAT4_PATH)
    km_yn_path  = find_keepmask_file(KEEP_YNOISE_ROOT, AEDAT4_PATH)
    km_rcf_path = find_keepmask_file(KEEP_RCF_ROOT, AEDAT4_PATH)

    km_dwf = load_keepmask(km_dwf_path).reshape(-1)
    km_ts  = load_keepmask(km_ts_path).reshape(-1)
    km_yn  = load_keepmask(km_yn_path).reshape(-1)

    km_rcf_raw = load_keepmask(km_rcf_path)
    km_rcf = pick_rcf_eta_column(km_rcf_raw, eta=RCF_ETA, eta_list=RCF_ETA_LIST).reshape(-1)

    # ---------- Auto-crop to min length ----------
    N_min = min(t_us.size, km_dwf.size, km_ts.size, km_yn.size, km_rcf.size)
    if N_min <= 0:
        raise RuntimeError("After aligning, N_min <= 0. Check inputs.")

    if N_min != t_us.size:
        print(f"[WARN] events length {t_us.size} -> {N_min} (cropped)")
    if N_min != km_dwf.size:
        print(f"[WARN] DWF keepmask length {km_dwf.size} -> {N_min} (cropped)")
    if N_min != km_ts.size:
        print(f"[WARN] TS keepmask length {km_ts.size} -> {N_min} (cropped)")
    if N_min != km_yn.size:
        print(f"[WARN] YNoise keepmask length {km_yn.size} -> {N_min} (cropped)")
    if N_min != km_rcf.size:
        print(f"[WARN] RCF keepmask length {km_rcf.size} -> {N_min} (cropped)")

    t_us = t_us[:N_min]
    x = x[:N_min]
    y = y[:N_min]
    p = p[:N_min]

    km_dwf = km_dwf[:N_min]
    km_ts  = km_ts[:N_min]
    km_yn  = km_yn[:N_min]
    km_rcf = km_rcf[:N_min]
    # --------------------------------------------

    keep_dwf = (km_dwf.astype(np.int32) > 0)
    keep_ts  = (km_ts.astype(np.int32) > 0)
    keep_yn  = (km_yn.astype(np.int32) > 0)
    keep_rcf = (km_rcf.astype(np.int32) > 0)

    # Render frames
    for i in range(N_FRAMES):
        t0 = i * DT_US
        t1 = (i + 1) * DT_US

        idx = (t_us >= t0) & (t_us < t1)

        # raw
        path_raw = os.path.join(dir_raw, f"frame_{i:03d}.png")
        render_event_frame_png(x[idx], y[idx], p[idx], SENSOR_W, SENSOR_H, path_raw)

        # ts
        idx_ts = idx & keep_ts
        path_ts = os.path.join(dir_ts, f"frame_{i:03d}.png")
        render_event_frame_png(x[idx_ts], y[idx_ts], p[idx_ts], SENSOR_W, SENSOR_H, path_ts)

        # ynoise
        idx_yn = idx & keep_yn
        path_yn = os.path.join(dir_yn, f"frame_{i:03d}.png")
        render_event_frame_png(x[idx_yn], y[idx_yn], p[idx_yn], SENSOR_W, SENSOR_H, path_yn)

        # dwf
        idx_dwf = idx & keep_dwf
        path_dwf = os.path.join(dir_dwf, f"frame_{i:03d}.png")
        render_event_frame_png(x[idx_dwf], y[idx_dwf], p[idx_dwf], SENSOR_W, SENSOR_H, path_dwf)

        # rcf
        idx_rcf = idx & keep_rcf
        path_rcf = os.path.join(dir_rcf, f"frame_{i:03d}.png")
        render_event_frame_png(x[idx_rcf], y[idx_rcf], p[idx_rcf], SENSOR_W, SENSOR_H, path_rcf)

        # stitch 1x5: raw, ts, ynoise, dwf, rcf
        out_stitch = os.path.join(dir_pinjie, f"frame_{i:03d}.png")
        # stitch_1x5([path_raw, path_ts, path_yn, path_dwf, path_rcf], out_stitch)
        stitch_1x5_with_label(
            paths_in_order=[
                path_raw,
                path_ts,
                path_yn,
                path_dwf,
                path_rcf,
            ],
            labels=[
                "Raw",
                "TS",
                "YNoise",
                "DWF",
                "RCF",
            ],
            out_path=out_stitch,
        )

    print("\n[OK] Visualization finished.")
    print("AEDAT4 :", AEDAT4_PATH)
    print("Keepmask matched:")
    print("  DWF   :", km_dwf_path)
    print("  TS    :", km_ts_path)
    print("  YNoise:", km_yn_path)
    print("  RCF   :", km_rcf_path)
    print("Output :", OUT_DIR)
    print("RCF eta:", RCF_ETA, " (column from", RCF_ETA_LIST, ")")
    print("Aligned length:", N_min)


if __name__ == "__main__":
    run()
