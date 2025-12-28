import re
import numpy as np
from pathlib import Path
import dv_processing as dv


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    base = Path(__file__).resolve().parent
    return (base / p).resolve()


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


def iter_event_batches(aedat4_path: Path):
    reader = dv.io.MonoCameraRecording(str(aedat4_path))
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            continue
        yield events


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
        return 0, 0, 0, None

    duration = t_last - t0
    n_fullbins = duration // bin_us
    last_full_end = t0 + n_fullbins * bin_us  # [t0, last_full_end)

    n_full = 0
    for events in iter_event_batches(aedat4_path):
        if events.size() == 0:
            continue
        sliced = events.sliceTime(t0, last_full_end)
        n_full += sliced.size()

    return n_full, total, n_fullbins, last_full_end


def km_event_count(km: np.ndarray) -> int:
    if km.ndim == 1:
        return int(km.shape[0])
    if km.ndim == 2:
        return int(km.shape[0])
    raise AssertionError(f"Unsupported keepmask shape: {km.shape}, ndim={km.ndim}")


def extract_nd_from_stem(stem: str) -> str | None:
    m = re.search(r"(ND\d{2})", stem)
    return m.group(1) if m else None


def derive_km_path(aedat4_path: Path, km_dir: Path, target_nd: str) -> Path:
    """
    Architecture-ND00-1.aedat4 -> Architecture-ND00-1_nd00_keepmask.npz
    target_nd: 'ND00'/'ND04'/'ND16'...
    """
    stem = aedat4_path.stem
    nd = extract_nd_from_stem(stem)
    if nd is None:
        raise ValueError(f"Cannot find NDxx in filename stem: {stem}")

    if nd.upper() != target_nd.upper():
        # 理论上不会走到这里（因为我们在文件筛选阶段就过滤掉了）
        raise ValueError(f"File ND mismatch: {stem} has {nd}, target is {target_nd}")

    km_name = f"{stem}_{target_nd.lower()}_keepmask.npz"
    return km_dir / km_name


def check_all(split: str = "day",
              target_nd: str = "ND00",
              emlb_root: str | Path = "data/emlb",
              km_dir: str | Path = "data/emlb_rcf_verify/keepmask",
              bin_us: int = 10000,
              verbose_fail_only: bool = True):
    """
    split: 'day' 或 'night'
    target_nd: 'ND00' / 'ND04' / 'ND16' ...
    """
    target_nd = target_nd.upper()
    if not re.fullmatch(r"ND\d{2}", target_nd):
        raise ValueError(f"target_nd must like 'ND00', got: {target_nd}")

    emlb_root = resolve_path(emlb_root)
    km_dir = resolve_path(km_dir)

    split_dir = emlb_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    if not km_dir.exists():
        raise FileNotFoundError(f"Keepmask dir not found: {km_dir}")

    # 只筛选目标 ND 的 aedat4
    all_aedat4 = sorted(split_dir.rglob("*.aedat4"))
    aedat4_files = [p for p in all_aedat4 if extract_nd_from_stem(p.stem) == target_nd]

    if len(aedat4_files) == 0:
        raise RuntimeError(f"No {target_nd} .aedat4 files found under: {split_dir}")

    print("=" * 100)
    print(f"[SPLIT] {split} | [TARGET_ND] {target_nd}")
    print(f"[AEDAT4 ROOT] {split_dir}")
    print(f"[KM DIR] {km_dir}")
    print(f"[BIN] bin_us={bin_us}")
    print(f"[FILES] total aedat4={len(all_aedat4)} | filtered({target_nd})={len(aedat4_files)}")
    print("=" * 100)

    ok = 0
    fail = 0
    missing_km = 0
    results = []

    for idx, aedat4_path in enumerate(aedat4_files, 1):
        rel = aedat4_path.relative_to(emlb_root)
        try:
            km_path = derive_km_path(aedat4_path, km_dir, target_nd)
            if not km_path.exists():
                missing_km += 1
                fail += 1
                msg = f"[MISS] {rel} -> km not found: {km_path.name}"
                print(msg)
                results.append({
                    "aedat4": str(rel),
                    "km": km_path.name,
                    "status": "MISSING_KM",
                })
                continue

            n_full, n_total, n_bins, _ = count_events_fullbins_by_time(aedat4_path, bin_us)
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
                "km": km_path.name,
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
            })

        if idx % 10 == 0:
            print(f"...progress {idx}/{len(aedat4_files)} | ok={ok} fail={fail} missing_km={missing_km}")

    print("=" * 100)
    print(f"[DONE] split={split} target_nd={target_nd} | ok={ok} fail={fail} missing_km={missing_km} | checked={len(aedat4_files)}")
    print("=" * 100)

    return results


if __name__ == "__main__":
    check_all(
        split="day",          # 后续改成 "night"
        target_nd="ND00",     # 后续改成 "ND04" / "ND16"
        emlb_root="data/emlb",
        km_dir="data/emlb_rcf_verify/keepmask",
        bin_us=10000,
        verbose_fail_only=True
    )
