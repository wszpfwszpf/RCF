import re
import numpy as np
from pathlib import Path
import dv_processing as dv


# -----------------------------
# Path resolve: 以脚本目录为基准解析相对路径
# -----------------------------
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    base = Path(__file__).resolve().parent
    return (base / p).resolve()


# -----------------------------
# Load keepmask from NPZ: key='keepmask'
# 支持 object list（按bin存）拼接
# -----------------------------
def load_keepmask(npz_path: Path, key: str = "keepmask") -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"KM npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {list(data.keys())}")

    km = np.asarray(data[key])

    # 若按 bin 存为 object array：拼起来
    if km.ndim == 1 and km.dtype == object:
        parts = [np.asarray(x) for x in km]
        km = np.concatenate(parts, axis=0)

    return km


# -----------------------------
# Iterate event batches using MonoCameraRecording (与你代码一致)
# -----------------------------
def iter_event_batches(aedat4_path: Path):
    reader = dv.io.MonoCameraRecording(str(aedat4_path))
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            continue
        yield events


# -----------------------------
# Count full-bin events by fixed time window (microseconds)
# 丢弃尾部不足 1bin 的时间段
# 起点以第一个事件 t0 对齐（与你之前验证一致）
# -----------------------------
def count_events_fullbins_by_time(aedat4_path: Path, bin_us: int):
    t0 = None
    t_last = None
    total = 0

    for events in iter_event_batches(aedat4_path):
        if events.size() == 0:
            continue
        if t0 is None:
            t0 = int(events.getLowestTime())
        t_last = int(events.getHighestTime())
        total += events.size()

    if t0 is None:
        return 0, 0, 0, None  # empty

    duration = t_last - t0
    n_fullbins = duration // bin_us
    last_full_end = t0 + n_fullbins * bin_us  # [t0, last_full_end) 覆盖完整 bins

    # 第二遍：统计落在完整 bins 时间范围内的事件数
    n_full = 0
    for events in iter_event_batches(aedat4_path):
        if events.size() == 0:
            continue
        sliced = events.sliceTime(t0, last_full_end)  # [t0, last_full_end)
        n_full += sliced.size()

    return n_full, total, n_fullbins, last_full_end


# -----------------------------
# Derive km filename from aedat4 filename:
#   Architecture-ND00-1.aedat4 -> Architecture-ND00-1_nd00_keepmask.npz
#   关键：_ndxx_keepmask 中 ndxx 小写
# -----------------------------
def derive_km_path(aedat4_path: Path, km_dir: Path) -> Path:
    stem = aedat4_path.stem  # e.g., Architecture-ND00-1
    m = re.search(r"(ND\d{2})", stem)
    if not m:
        raise ValueError(f"Cannot find 'NDxx' in filename stem: {stem}")
    nd = m.group(1).lower()  # nd00
    km_name = f"{stem}_{nd}_keepmask.npz"
    return km_dir / km_name


def km_event_count(km: np.ndarray) -> int:
    # keepmask 可能是 (N,) 或 (N,6)
    if km.ndim == 1:
        return int(km.shape[0])
    if km.ndim == 2:
        return int(km.shape[0])
    raise AssertionError(f"Unsupported keepmask shape: {km.shape}, ndim={km.ndim}")


# -----------------------------
# Batch check
# -----------------------------
def check_all(split: str = "day",
              emlb_root: str | Path = "data/emlb",
              km_dir: str | Path = "data/emlb_rcf_verify/keepmask",
              bin_us: int = 10000,
              verbose_fail_only: bool = True):
    """
    split: 'day' 或 'night'
    """
    emlb_root = resolve_path(emlb_root)
    km_dir = resolve_path(km_dir)

    split_dir = emlb_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    if not km_dir.exists():
        raise FileNotFoundError(f"Keepmask dir not found: {km_dir}")

    aedat4_files = sorted(split_dir.rglob("*.aedat4"))
    if len(aedat4_files) == 0:
        raise RuntimeError(f"No .aedat4 files found under: {split_dir}")

    print("=" * 90)
    print(f"[SPLIT] {split}")
    print(f"[AEDAT4 ROOT] {split_dir}")
    print(f"[KM DIR] {km_dir}")
    print(f"[BIN] bin_us={bin_us}")
    print(f"[FILES] aedat4 count = {len(aedat4_files)}")
    print("=" * 90)

    ok = 0
    fail = 0
    missing_km = 0

    results = []  # 用于最终汇总（可导出csv/npz）

    for idx, aedat4_path in enumerate(aedat4_files, 1):
        rel = aedat4_path.relative_to(emlb_root)
        try:
            km_path = derive_km_path(aedat4_path, km_dir)
            if not km_path.exists():
                missing_km += 1
                fail += 1
                msg = f"[MISS] {rel}  -> km not found: {km_path.name}"
                if not verbose_fail_only:
                    print(msg)
                else:
                    print(msg)
                results.append({
                    "aedat4": str(rel),
                    "km": str(km_path.name),
                    "status": "MISSING_KM",
                    "total_events": None,
                    "fullbins_events": None,
                    "km_events": None,
                    "fullbins": None,
                })
                continue

            # 统计完整 bins 事件数（与你已验证一致）
            n_full, n_total, n_bins, last_end = count_events_fullbins_by_time(aedat4_path, bin_us)

            km = load_keepmask(km_path)
            n_km = km_event_count(km)

            if n_full == n_km:
                ok += 1
                if not verbose_fail_only:
                    print(f"[PASS] {rel} | total={n_total} full={n_full} bins={n_bins} km={n_km}")
                status = "PASS"
            else:
                fail += 1
                print(f"[FAIL] {rel} | total={n_total} full={n_full} bins={n_bins} km={n_km} | km={km_path.name}")
                status = "MISMATCH"

            results.append({
                "aedat4": str(rel),
                "km": str(km_path.name),
                "status": status,
                "total_events": int(n_total),
                "fullbins_events": int(n_full),
                "km_events": int(n_km),
                "fullbins": int(n_bins),
            })

        except Exception as e:
            fail += 1
            print(f"[ERR ] {rel} | {type(e).__name__}: {e}")
            results.append({
                "aedat4": str(rel),
                "km": None,
                "status": f"ERROR:{type(e).__name__}",
                "total_events": None,
                "fullbins_events": None,
                "km_events": None,
                "fullbins": None,
            })

        # 简单进度
        if idx % 10 == 0:
            print(f"...progress {idx}/{len(aedat4_files)} | ok={ok} fail={fail} missing_km={missing_km}")

    print("=" * 90)
    print(f"[DONE] split={split} | ok={ok} fail={fail} missing_km={missing_km} | total={len(aedat4_files)}")
    print("=" * 90)

    # 如果你想把结果落盘（便于论文/日志），取消注释即可：
    # out = resolve_path(f"km_align_report_{split}_bin{bin_us}us.npz")
    # np.savez_compressed(out, results=np.array(results, dtype=object))
    # print(f"[SAVED] {out}")

    return results


if __name__ == "__main__":
    # 你要兼容 night：把 split 改成 'night' 再跑一次即可
    # 注意：确保 data/emlb/night 和对应 keepmask 已准备好（命名规则一致）
    check_all(
        split="day",
        emlb_root="data/emlb",
        km_dir="data/emlb_rcf_verify/keepmask",
        bin_us=10000,
        verbose_fail_only=True
    )
