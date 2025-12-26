# tools/eval_esr_npz.py
# Evaluate MESR on NPZ files:
# - raw MESR
# - denoised MESR at eta in {0.05, 0.10, 0.15, 0.20}
#
# Input : data/rcf-scored/*.npz containing t,x,y,(p),score (float32)
# Output: data/esr-eval/summary_esr.csv

from __future__ import annotations

import numpy as np
from pathlib import Path

from metrics.esr_config import DEFAULT_ESR_CONFIG as ESR_CFG
from metrics.mesr_eval_npz import iter_packets_by_N
from metrics.esr_core import compute_esr


# -------------------------
# CONFIG (match scan_eta.py style)
# -------------------------
IN_DIR_REL = Path("data/rcf-scored")
OUT_DIR_REL = Path("data/esr-eval")
RECURSIVE = True
OVERWRITE = True
VERBOSE = True

ETA_LIST = [0.05, 0.10, 0.15, 0.20]  # fixed 4 etas


# -------------------------
# Helpers (copied/minimized from scan_eta.py)
# -------------------------
def project_root() -> Path:
    # tools/eval_esr_npz.py -> project root = parent of tools
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


def _filter_packet(pkt: dict, mask: np.ndarray) -> dict:
    # Keep only per-event arrays
    out = {}
    n = pkt["x"].shape[0]
    for k, v in pkt.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def eval_file_mesr_raw_and_etas(npz_path: Path) -> dict:
    data = np.load(npz_path)

    # Required
    t = data["t"].astype(np.int64)  # not used by ESR, but keep aligned
    x = data["x"].astype(np.int32)
    y = data["y"].astype(np.int32)
    score = data["score"].astype(np.float32)

    # Optional
    p = data["p"].astype(np.int8) if "p" in data.files else np.zeros_like(t, dtype=np.int8)

    events = {"t": t, "x": x, "y": y, "p": p}
    n_total = int(x.shape[0])
    if n_total == 0:
        raise ValueError("empty event stream")

    # Accumulators
    raw_esr_list = []
    eta_esr_lists = {eta: [] for eta in ETA_LIST}
    eta_ret_lists = {eta: [] for eta in ETA_LIST}

    n_packets = 0

    # Iterate RAW packets by protocol N (use cfg)
    for _, start, end, pkt in iter_packets_by_N(events, cfg=ESR_CFG):
        n_packets += 1

        # raw ESR
        raw_esr = compute_esr(
            pkt,
            resolution=ESR_CFG.resolution,
            M=ESR_CFG.m_events_ref,
            hot_pixel_valid_mask=None if ESR_CFG.hot_pixel_valid_mask is None else np.asarray(ESR_CFG.hot_pixel_valid_mask),
            validate_xy=ESR_CFG.validate_xy,
        )
        raw_esr_list.append(float(raw_esr))

        # score slice aligned with raw packet indices
        s = score[start:end]

        # each eta: filter inside the packet, then ESR
        for eta in ETA_LIST:
            m = (s >= float(eta))
            kept = int(m.sum())
            ret = kept / float(end - start)
            eta_ret_lists[eta].append(float(ret))

            pkt_dn = _filter_packet(pkt, m)
            dn_esr = compute_esr(
                pkt_dn,
                resolution=ESR_CFG.resolution,
                M=ESR_CFG.m_events_ref,
                hot_pixel_valid_mask=None if ESR_CFG.hot_pixel_valid_mask is None else np.asarray(ESR_CFG.hot_pixel_valid_mask),
                validate_xy=ESR_CFG.validate_xy,
            )
            eta_esr_lists[eta].append(float(dn_esr))

    # Aggregate
    raw_arr = np.asarray(raw_esr_list, dtype=np.float64)
    out = {
        "file": npz_path.name,
        "N_total": n_total,
        "n_packets": n_packets,
        "mesr_raw": float(np.mean(raw_arr)) if n_packets > 0 else float("nan"),
        "mesr_raw_std": float(np.std(raw_arr)) if n_packets > 0 else float("nan"),
    }

    for eta in ETA_LIST:
        arr = np.asarray(eta_esr_lists[eta], dtype=np.float64)
        rarr = np.asarray(eta_ret_lists[eta], dtype=np.float64)
        out[f"mesr_eta{eta:.2f}"] = float(np.mean(arr)) if arr.size else float("nan")
        out[f"mesr_eta{eta:.2f}_std"] = float(np.std(arr)) if arr.size else float("nan")
        out[f"ret_eta{eta:.2f}"] = float(np.mean(rarr)) if rarr.size else float("nan")

    return out


def main():
    root = project_root()
    in_dir = (root / IN_DIR_REL).resolve()
    out_dir = (root / OUT_DIR_REL).resolve()
    ensure_dir(out_dir)

    out_csv = out_dir / "summary_esr.csv"
    if out_csv.exists() and (not OVERWRITE):
        print(f"[INFO] exists and OVERWRITE=False: {out_csv}")
        return

    if VERBOSE:
        print(f"[INFO] Project root: {root}")
        print(f"[INFO] Input dir   : {in_dir}")
        print(f"[INFO] Output dir  : {out_dir}")
        print(f"[INFO] ETA_LIST    : {ETA_LIST}")
        print(f"[INFO] ESR cfg     : res={ESR_CFG.resolution}, N={ESR_CFG.n_events_packet}, M={ESR_CFG.m_events_ref}, drop_last={ESR_CFG.drop_last}")

    files = list_npz_files(in_dir, recursive=RECURSIVE)
    if not files:
        print(f"[ERROR] No .npz files found in: {in_dir}")
        return

    rows = []
    ok, fail = 0, 0

    for f in files:
        try:
            row = eval_file_mesr_raw_and_etas(f)
            rows.append(row)
            ok += 1
            if VERBOSE:
                print(f"[OK] {f.name} | N={row['N_total']} | packets={row['n_packets']} | raw_mesr={row['mesr_raw']:.6f}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {f.name}: {repr(e)}")

    if rows:
        # stable column order: start with fixed keys
        base_cols = ["file", "N_total", "n_packets", "mesr_raw", "mesr_raw_std"]
        eta_cols = []
        for eta in ETA_LIST:
            eta_cols += [f"mesr_eta{eta:.2f}", f"mesr_eta{eta:.2f}_std", f"ret_eta{eta:.2f}"]
        # reorder dicts for consistent csv
        ordered_rows = []
        for r in rows:
            rr = {k: r.get(k, "") for k in (base_cols + eta_cols)}
            ordered_rows.append(rr)
        write_csv(ordered_rows, out_csv)

    print(f"\n[SUMMARY] files_ok={ok}, files_fail={fail}, total_files={len(files)}")
    print(f"[SUMMARY] saved: {out_csv}")


if __name__ == "__main__":
    main()
