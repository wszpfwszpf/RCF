import os
import sys
import csv
from pathlib import Path
from collections import Counter
import numpy as np

# ---------------------------
# Path helpers
# ---------------------------
def get_proj_root() -> Path:
    # proj_root = (this_file)/../
    return Path(__file__).resolve().parents[1]

def scan_files(root: Path, suffix: str):
    root = root.resolve()
    files = sorted([p for p in root.rglob(f"*{suffix}") if p.is_file()])
    return files

def suffix_stats(files):
    c = Counter([p.suffix.lower() for p in files])
    return c

def print_hits(files, max_show=30, prefix="[HITS]"):
    print(f"{prefix} showing up to {max_show} paths:")
    for p in files[:max_show]:
        print(f"  - {p}")

# ---------------------------
# Official readers (dat/txt)
# ---------------------------
def read_dv_dat_timerange(dat_path: Path, chunk_us: int = 500_000):
    """
    Use official PSEELoader.load_delta_t() to read a small time window.
    Returns dict with t0_us, t1_us, n, ok, err.
    """
    try:
        from beam.utils.io.psee_loader import PSEELoader  # your local copied official io
    except Exception as e:
        return {"ok": 0, "err": f"import PSEELoader failed: {e}", "t0_us": None, "t1_us": None, "n": 0}

    try:
        video = PSEELoader(str(dat_path))
        ev = video.load_delta_t(int(chunk_us))  # t in microseconds (per official demo)
        if ev is None or len(ev) == 0:
            return {"ok": 0, "err": "empty events in chunk", "t0_us": None, "t1_us": None, "n": 0}

        # Some loaders return structured array with fields. We only need t.
        t = ev["t"]
        t0 = int(np.min(t))
        t1 = int(np.max(t))
        return {"ok": 1, "err": "", "t0_us": t0, "t1_us": t1, "n": int(len(ev))}
    except Exception as e:
        return {"ok": 0, "err": f"read dat failed: {e}", "t0_us": None, "t1_us": None, "n": 0}

def read_ldv_txt_timerange(txt_path: Path, skiprows: int = 5):
    """
    Official LDV txt: usually header then numeric columns.
    Based on provided H-beam_LDV.py: np.loadtxt(skiprows=5), time in col0 (s).
    Returns dict with t0_s, t1_s, n, ok, err.
    """
    try:
        data = np.loadtxt(str(txt_path), skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            return {"ok": 0, "err": "empty ldv data", "t0_s": None, "t1_s": None, "n": 0}

        t = data[:, 0].astype(np.float64)
        t0 = float(np.min(t))
        t1 = float(np.max(t))
        return {"ok": 1, "err": "", "t0_s": t0, "t1_s": t1, "n": int(data.shape[0])}
    except Exception as e:
        return {"ok": 0, "err": f"read txt failed: {e}", "t0_s": None, "t1_s": None, "n": 0}

# ---------------------------
# Pairing + overlap check
# ---------------------------
def make_pairs(dv_files, ldv_files):
    dv_map = {p.stem: p for p in dv_files}     # exact stem match (you confirmed)
    ldv_map = {p.stem: p for p in ldv_files}
    keys = sorted(set(dv_map.keys()) & set(ldv_map.keys()))
    miss_dv = sorted(set(ldv_map.keys()) - set(dv_map.keys()))
    miss_ldv = sorted(set(dv_map.keys()) - set(ldv_map.keys()))
    pairs = [(k, dv_map[k], ldv_map[k]) for k in keys]
    return pairs, miss_dv, miss_ldv

def overlap_seconds(dv_t0_us, dv_t1_us, ldv_t0_s, ldv_t1_s):
    dv0 = dv_t0_us / 1e6
    dv1 = dv_t1_us / 1e6
    a0 = max(dv0, ldv_t0_s)
    a1 = min(dv1, ldv_t1_s)
    ov = max(0.0, a1 - a0)
    return ov, (a0, a1), (dv0, dv1)

def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def write_csv(rows, out_csv: Path):
    ensure_out_dir(out_csv.parent)
    header = [
        "key",
        "dv_path", "ldv_path",
        "dv_chunk_events", "dv_t0_us", "dv_t1_us",
        "ldv_samples", "ldv_t0_s", "ldv_t1_s",
        "overlap_s",
        "crop_start_s", "crop_end_s",
        "suggest_dv_crop_start_us", "suggest_dv_crop_end_us",
        "suggest_ldv_crop_start_s", "suggest_ldv_crop_end_s",
        "ok", "err"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

# ---------------------------
# Main
# ---------------------------
def main():
    proj_root = get_proj_root()
    dv_root = proj_root / "data" / "DV"
    ldv_root = proj_root / "data" / "LDV"
    out_csv = proj_root / "outputs" / "beam_check" / "beam_pairs_summary.csv"

    print("=" * 80)
    print(f"[INFO] script    : {Path(__file__).resolve()}")
    print(f"[INFO] proj_root : {proj_root}")
    print(f"[INFO] DV root   : {dv_root}")
    print(f"[INFO] LDV root  : {ldv_root}")
    print("=" * 80)

    dv_files = scan_files(dv_root, ".dat")
    ldv_files = scan_files(ldv_root, ".txt")

    print(f"[SCAN] {dv_root}  files={len(dv_files)}")
    print(f"[SUFFIX] {sorted(suffix_stats(dv_files).items(), key=lambda x: -x[1])}")
    print_hits(dv_files, max_show=30)

    print(f"[SCAN] {ldv_root}  files={len(ldv_files)}")
    print(f"[SUFFIX] {sorted(suffix_stats(ldv_files).items(), key=lambda x: -x[1])}")
    print_hits(ldv_files, max_show=30)

    print("=" * 80)
    pairs, miss_dv, miss_ldv = make_pairs(dv_files, ldv_files)
    print(f"[PAIR] DV={len(dv_files)}  LDV={len(ldv_files)}  to_check={len(pairs)}")
    if miss_dv:
        print(f"[PAIR] missing DV for {len(miss_dv)} keys, e.g. {miss_dv[:10]}")
    if miss_ldv:
        print(f"[PAIR] missing LDV for {len(miss_ldv)} keys, e.g. {miss_ldv[:10]}")
    print("=" * 80)

    rows = []
    ok_cnt = 0
    for i, (k, dv_p, ldv_p) in enumerate(pairs, 1):
        if i <= 5 or i in (20, 30):
            print(f"[{i:>4}/{len(pairs)}] key={k} dv={dv_p.name}  ldv={ldv_p.name}")

        dv = read_dv_dat_timerange(dv_p, chunk_us=500_000)  # 0.5s chunk to keep it light
        ldv = read_ldv_txt_timerange(ldv_p, skiprows=5)

        row = {
            "key": k,
            "dv_path": str(dv_p),
            "ldv_path": str(ldv_p),
            "dv_chunk_events": dv.get("n", 0),
            "dv_t0_us": dv.get("t0_us", ""),
            "dv_t1_us": dv.get("t1_us", ""),
            "ldv_samples": ldv.get("n", 0),
            "ldv_t0_s": ldv.get("t0_s", ""),
            "ldv_t1_s": ldv.get("t1_s", ""),
            "overlap_s": "",
            "crop_start_s": "",
            "crop_end_s": "",
            "suggest_dv_crop_start_us": "",
            "suggest_dv_crop_end_us": "",
            "suggest_ldv_crop_start_s": "",
            "suggest_ldv_crop_end_s": "",
            "ok": 0,
            "err": ""
        }

        if dv["ok"] and ldv["ok"]:
            ov, (c0, c1), (dv0, dv1) = overlap_seconds(dv["t0_us"], dv["t1_us"], ldv["t0_s"], ldv["t1_s"])
            row["overlap_s"] = ov
            row["crop_start_s"] = c0
            row["crop_end_s"] = c1

            # suggested crop (use overlap interval)
            row["suggest_dv_crop_start_us"] = int(c0 * 1e6)
            row["suggest_dv_crop_end_us"] = int(c1 * 1e6)
            row["suggest_ldv_crop_start_s"] = c0
            row["suggest_ldv_crop_end_s"] = c1

            # ok criterion: have positive overlap
            if ov > 0:
                row["ok"] = 1
                ok_cnt += 1
            else:
                row["err"] = "no temporal overlap (based on DV chunk and LDV range)"
        else:
            errs = []
            if not dv["ok"]:
                errs.append(dv["err"])
            if not ldv["ok"]:
                errs.append(ldv["err"])
            row["err"] = " | ".join(errs)

        rows.append(row)

    write_csv(rows, out_csv)
    print("=" * 80)
    print(f"[DONE] pairs={len(pairs)}  overlap_ok={ok_cnt}  not_ok={len(pairs)-ok_cnt}")
    print(f"[OUT ] {out_csv}")
    print("=" * 80)

    import json
    import numpy as np

    def estimate_ldv_dt_s(txt_path: Path, skiprows: int = 5, probe_n: int = 2000) -> float:
        data = np.loadtxt(str(txt_path), skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t = data[:min(len(data), probe_n), 0].astype(np.float64)
        if len(t) < 5:
            return float("nan")
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            return float("nan")
        return float(np.median(dt))

    # after you have `rows` list:
    align = {}
    for r in rows:
        if int(r.get("ok", 0)) != 1:
            continue
        k = r["key"]
        ldv_path = Path(r["ldv_path"])
        dt_s = estimate_ldv_dt_s(ldv_path, skiprows=5)
        align[k] = {
            "dv_path": r["dv_path"],
            "ldv_path": r["ldv_path"],
            "dv_crop_t0_us": int(float(r["suggest_dv_crop_start_us"])),
            "dv_crop_t1_us": int(float(r["suggest_dv_crop_end_us"])),
            "ldv_crop_t0_s": float(r["suggest_ldv_crop_start_s"]),
            "ldv_crop_t1_s": float(r["suggest_ldv_crop_end_s"]),
            "ldv_dt_s": dt_s,
            "align_mode": "overlap",
        }

    out_json = proj_root / "outputs" / "beam_check" / "beam_align_config.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(align, f, indent=2, ensure_ascii=False)
    print(f"[OUT ] {out_json}")


if __name__ == "__main__":
    main()
