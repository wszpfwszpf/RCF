import os
import glob
import csv
import numpy as np
from tqdm import tqdm

from beam.utils.io.psee_loader import PSEELoader


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return np.median(np.abs(x - med))


def floor_to_step(x: float, step: float) -> float:
    # 向下凑整：2.17 -> 2.1（step=0.1）
    return np.floor(x / step) * step


def compute_band_rate_trigger(
    dat_path: str,
    max_time_s: float = 6.0,
    x_min: int = 600,
    x_max: int = 670,
    bin_ms: float = 2.0,
    baseline_s: float = 0.5,
    k: float = 8.0,
    consec_bins: int = 10,
):
    """
    返回：
      t_onset_us: 触发时刻（us）
      stats: dict(用于记录CSV)
    """
    max_time_us = int(round(max_time_s * 1e6))
    bin_us = int(round(bin_ms * 1000))

    video = PSEELoader(dat_path)
    events = video.load_delta_t(max_time_us)  # 从起点加载前max_time_us

    # 统一按时间排序
    events["t"].sort()

    # 选窄带事件（与官方可视化保持一致：只筛 x，不筛 y）
    x = events["x"] if "x" in events.dtype.names else events["p"]  # 保险兜底（一般不会走到）
    # 你的结构数组里通常是 ('t','x','y','p')，这里用字段名最稳
    x = events["x"]
    t_us = events["t"].astype(np.int64)

    band_mask = (x_min <= x) & (x <= x_max)
    t_band = t_us[band_mask]

    if t_band.size < 1000:
        # 事件太少时，触发不可靠
        return None, {
            "status": "too_few_events",
            "n_band": int(t_band.size),
            "bin_ms": bin_ms,
        }

    # bin计数：把 t_band 映射到 bin index
    t0 = int(t_us[0])
    t1 = int(min(t_us[-1], max_time_us))
    n_bins = max(1, (t1 - t0) // bin_us + 1)

    idx = ((t_band - t0) // bin_us).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)

    # 基线统计（前baseline_s）
    base_bins = int(round((baseline_s * 1e6) / bin_us))
    base_bins = max(10, min(base_bins, len(counts)))
    base = counts[:base_bins]

    med = float(np.median(base))
    m = float(mad(base))
    sigma = float(1.4826 * m) if m > 0 else float(np.std(base))
    thr = med + k * sigma

    # 连续超阈触发
    exceed = counts > thr
    t_onset_us = None
    for i in range(0, len(exceed) - consec_bins + 1):
        if exceed[i:i + consec_bins].all():
            t_onset_us = t0 + i * bin_us
            break

    # 兜底：若没触发，用最大峰所在bin
    if t_onset_us is None:
        i_peak = int(np.argmax(counts))
        t_onset_us = t0 + i_peak * bin_us
        status = "fallback_peak"
    else:
        status = "ok"

    stats = {
        "status": status,
        "t0_us": int(t0),
        "t1_us": int(t1),
        "n_total": int(t_us.size),
        "n_band": int(t_band.size),
        "bin_ms": float(bin_ms),
        "baseline_s": float(baseline_s),
        "k": float(k),
        "consec_bins": int(consec_bins),
        "med": med,
        "sigma": sigma,
        "thr": thr,
    }
    return int(t_onset_us), stats


def choose_clip_range(
    t_onset_us: int,
    t_min_us: int,
    t_max_us: int,
    clip_s: float = 2.0,
    pre_s: float = 0.2,
    round_step_s: float = 0.1,
):
    """
    - 先取 start = onset - pre
    - 再向下凑整到 0.1s（x.x）
    - end = start + 2s
    - 做边界裁剪，不够就往前挪
    """
    clip_us = int(round(clip_s * 1e6))
    pre_us = int(round(pre_s * 1e6))

    start_us = max(t_min_us, t_onset_us - pre_us)
    start_s = start_us * 1e-6

    # 凑整到 x.x（向下取整更安全：保证 start 不会跑到 onset 之后）
    start_s_round = floor_to_step(start_s, round_step_s)
    start_us = int(round(start_s_round * 1e6))

    end_us = start_us + clip_us

    # 若超出末尾，则向前挪
    if end_us > t_max_us:
        end_us = t_max_us
        start_us = max(t_min_us, end_us - clip_us)

        # 再次凑整（仍保持 x.x 形式），但要保证不越界
        start_s = start_us * 1e-6
        start_s_round = floor_to_step(start_s, round_step_s)
        start_us = int(round(start_s_round * 1e6))
        end_us = min(t_max_us, start_us + clip_us)

    return start_us, end_us


def main():
    DV_DIR = os.path.join("data", "DV")
    OUT_DIR = os.path.join("data", "DVcliped_2s_vis")
    os.makedirs(OUT_DIR, exist_ok=True)

    out_csv = os.path.join(OUT_DIR, "dv_clip_2s_ranges_by_bandrate.csv")

    # 你要的“窄带”
    X_MIN, X_MAX = 600, 670

    # 参数（默认偏稳）
    MAX_TIME_S = 6.0
    BIN_MS = 2.0
    BASELINE_S = 0.5
    K = 8.0
    CONSEC_BINS = 10

    # 截取窗口 & 凑整
    CLIP_S = 2.0
    PRE_S = 0.2
    ROUND_STEP_S = 0.1  # 2.1, 2.2, x.x

    dat_files = sorted(glob.glob(os.path.join(DV_DIR, "*.dat")))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {DV_DIR}")

    print("=" * 100)
    print("[STEP8-A] Find 2s clip ranges by band event-rate trigger (x in [600,670])")
    print(f"[IN ] DV_DIR      : {os.path.abspath(DV_DIR)}")
    print(f"[OUT] OUT_CSV     : {os.path.abspath(out_csv)}")
    print(f"[CFG] MAX_TIME_S  : {MAX_TIME_S}s")
    print(f"[CFG] BAND_X      : [{X_MIN}, {X_MAX}]")
    print(f"[CFG] BIN_MS      : {BIN_MS}ms | BASELINE_S={BASELINE_S}s | K={K} | CONSEC={CONSEC_BINS}")
    print(f"[CFG] CLIP_S      : {CLIP_S}s | PRE_S={PRE_S}s | ROUND_STEP={ROUND_STEP_S}s")
    print("=" * 100)

    rows = []
    for i, dat_path in enumerate(dat_files, 1):
        stem = os.path.splitext(os.path.basename(dat_path))[0]

        t_onset_us, stats = compute_band_rate_trigger(
            dat_path,
            max_time_s=MAX_TIME_S,
            x_min=X_MIN, x_max=X_MAX,
            bin_ms=BIN_MS,
            baseline_s=BASELINE_S,
            k=K,
            consec_bins=CONSEC_BINS,
        )

        if t_onset_us is None:
            print(f"[{i:03d}/{len(dat_files):03d}] {stem} | FAIL: {stats.get('status')} n_band={stats.get('n_band')}")
            rows.append([
                stem, "FAIL", "", "", "", "",
                stats.get("status", ""),
                stats.get("n_band", 0),
                "", "", "", "", "", "", "", ""
            ])
            continue

        t_min_us = int(stats["t0_us"])
        t_max_us = int(stats["t1_us"])

        t_start_us, t_end_us = choose_clip_range(
            t_onset_us=t_onset_us,
            t_min_us=t_min_us,
            t_max_us=t_max_us,
            clip_s=CLIP_S,
            pre_s=PRE_S,
            round_step_s=ROUND_STEP_S,
        )

        # 打印（用秒展示更直观）
        print(
            f"[{i:03d}/{len(dat_files):03d}] {stem} | onset={t_onset_us*1e-6:.3f}s "
            f"| clip=[{t_start_us*1e-6:.1f},{t_end_us*1e-6:.1f}]s | status={stats['status']}"
        )

        rows.append([
            stem, "OK",
            f"{t_onset_us}", f"{t_onset_us*1e-6:.6f}",
            f"{t_start_us}", f"{t_start_us*1e-6:.6f}",
            f"{t_end_us}", f"{t_end_us*1e-6:.6f}",
            stats["status"],
            stats["n_total"], stats["n_band"],
            stats["bin_ms"], stats["baseline_s"], stats["k"], stats["consec_bins"],
            stats["med"], stats["sigma"], stats["thr"],
        ])

    # 写CSV
    header = [
        "file_stem", "ok",
        "t_onset_us", "t_onset_s",
        "t_start_us", "t_start_s",
        "t_end_us", "t_end_s",
        "trigger_status",
        "n_total_events", "n_band_events",
        "bin_ms", "baseline_s", "k", "consec_bins",
        "baseline_median", "baseline_sigma_robust", "threshold",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("-" * 100)
    print(f"[DONE] Saved clip ranges CSV: {os.path.abspath(out_csv)}")
    print("-" * 100)


if __name__ == "__main__":
    main()
