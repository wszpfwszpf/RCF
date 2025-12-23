"""
Convert AEDAT4 event files to compressed NPZ for easy analysis.

This script is PyCharm "one-click run" friendly: default configuration is
provided at the top of the file and argparse only overrides those values when
explicit command-line parameters are given.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# --------------------------- CONFIG (editable defaults) ---------------------------
IN_DIR = "data/origin-aedat4"
OUT_DIR = "data/converted-npz"
NORMALIZE_T = True
OVERWRITE = False
VERBOSE = True
KEEP_POLARITY_SIGN = False  # False outputs p as {0,1}; True outputs p as {-1,+1}
RECURSIVE = True
# ----------------------------------------------------------------------------------


class MissingDependencyError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build argument parser with CONFIG defaults."""
    parser = argparse.ArgumentParser(
        description="Convert AEDAT4 event streams to NPZ with robust time scaling."
    )
    parser.add_argument("--in-dir", default=IN_DIR, help="Input directory with .aedat4 files.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for .npz files.")
    parser.add_argument(
        "--normalize-t",
        action="store_true",
        default=NORMALIZE_T,
        help="Normalize timestamps to start at zero.",
    )
    parser.add_argument(
        "--no-normalize-t",
        action="store_false",
        dest="normalize_t",
        help="Do not normalize timestamps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="Overwrite existing NPZ outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=VERBOSE,
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Silence per-file logs.",
    )
    parser.add_argument(
        "--keep-polarity-sign",
        action="store_true",
        default=KEEP_POLARITY_SIGN,
        help="Keep polarity as {-1,+1} instead of mapping to {0,1}.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=RECURSIVE,
        help="Recursively search for .aedat4 files.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_false",
        dest="recursive",
        help="Do not search subdirectories.",
    )
    return parser.parse_args(argv)


def ensure_dependency() -> None:
    """Ensure dv or dv-processing is importable."""
    try:
        import dv  # noqa: F401
    except ImportError as exc:  # pragma: no cover - informative path
        message = (
            "Missing dependency: install the 'dv' or 'dv-processing' package to read .aedat4 files. "
            "Try: pip install dv or pip install dv-processing"
        )
        raise MissingDependencyError(message) from exc


def load_events_from_aedat(path: Path) -> np.ndarray:
    """Load events from an AEDAT4 file using dv/dv-processing."""
    ensure_dependency()
    import dv

    reader_cls = None
    if hasattr(dv, "AedatFile"):
        reader_cls = dv.AedatFile
    elif hasattr(dv, "Aedat4"):
        reader_cls = dv.Aedat4

    if reader_cls is None:
        raise MissingDependencyError(
            "dv package is available but does not expose AedatFile/Aedat4 reader. "
            "Ensure dv or dv-processing is correctly installed."
        )

    packets: List[np.ndarray] = []
    with reader_cls(str(path)) as f:
        try:
            event_stream = f["events"] if isinstance(f, Mapping) else f.events()
        except Exception:  # pragma: no cover - defensive
            event_stream = getattr(f, "events", None)
        if event_stream is None:
            raise RuntimeError("Unable to access events stream in AEDAT file.")

        for packet in event_stream:
            arr = packet.numpy() if hasattr(packet, "numpy") else np.asarray(packet)
            if arr.size:
                packets.append(arr)

    if not packets:
        raise RuntimeError("No events found in file: %s" % path)

    events = np.concatenate(packets)
    return events


FieldMapping = Dict[str, np.ndarray]


KNOWN_T_NAMES = {"t", "timestamp", "time"}
KNOWN_X_NAMES = {"x"}
KNOWN_Y_NAMES = {"y"}
KNOWN_P_NAMES = {"p", "polarity", "pol", "sign"}


def _median_dt_us(t_scaled: np.ndarray) -> float:
    if t_scaled.size < 2:
        return float("inf")
    t0 = t_scaled - t_scaled[0]
    m = min(2000, t0.size)
    if m < 2:
        return float("inf")
    dt = np.diff(t0[:m])
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("inf")
    return float(np.median(dt))


def infer_time_unit_and_scale(t_raw: np.ndarray) -> Tuple[float, str]:
    """Infer scale to microseconds and provide reasoning."""
    t = np.asarray(t_raw)
    if t.size == 0:
        return 1.0, "Empty timestamp array; defaulting to scale=1 (us)."

    candidate_scales = [1.0, 1e3, 1e6]
    median_dts = {}
    for scale in candidate_scales:
        t_scaled = t.astype(np.float64) * scale
        if not np.all(np.diff(t_scaled) >= 0):
            # Sort for robustness when inspecting dt, but sorting happens later globally.
            t_scaled = np.sort(t_scaled)
        median_dts[scale] = _median_dt_us(t_scaled)

    typical_min, typical_max = 1.0, 2000.0

    def penalty(med_dt: float) -> float:
        if not np.isfinite(med_dt):
            return 1e6
        if typical_min <= med_dt <= typical_max:
            return 0.0
        if med_dt < typical_min:
            return typical_min - med_dt
        return med_dt - typical_max

    max_t = float(np.nanmax(t)) if t.size else 0.0
    dtype_is_float = np.issubdtype(t.dtype, np.floating)
    prior = {scale: 0.0 for scale in candidate_scales}
    if max_t >= 1e6:
        prior[1.0] -= 0.25
    if dtype_is_float and max_t < 1.0:
        prior[1e6] -= 0.25
    if 1.0 <= max_t < 1e6:
        prior[1e3] -= 0.1

    scores = {
        scale: penalty(median_dts[scale]) + prior[scale] for scale in candidate_scales
    }
    best_scale = min(candidate_scales, key=lambda s: scores[s])

    reason = (
        f"median dt (us scaled) per candidate: "
        f"{ {scale: round(median_dts[scale], 3) for scale in candidate_scales} }; "
        f"max_t={max_t}; dtype_is_float={dtype_is_float}; chosen scale_to_us={best_scale}"
    )
    return best_scale, reason


def infer_event_fields(events: np.ndarray) -> FieldMapping:
    """Infer t/x/y/p fields using names and value heuristics."""
    if events.dtype.names:
        names = {name.lower(): name for name in events.dtype.names}
        t_name = next((names[n] for n in names if n in KNOWN_T_NAMES), None)
        x_name = next((names[n] for n in names if n in KNOWN_X_NAMES), None)
        y_name = next((names[n] for n in names if n in KNOWN_Y_NAMES), None)
        p_name = next((names[n] for n in names if n in KNOWN_P_NAMES), None)
        if all([t_name, x_name, y_name, p_name]):
            return {
                "t": np.asarray(events[t_name]),
                "x": np.asarray(events[x_name]),
                "y": np.asarray(events[y_name]),
                "p": np.asarray(events[p_name]),
            }

    arr = np.asarray(events)
    if arr.ndim == 1 and arr.dtype.names:
        # Already handled above, but keep guard
        raise ValueError("Unable to map structured fields to t/x/y/p")

    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Events array must be 2D with at least four columns when unnamed.")

    candidates: List[Tuple[int, float]] = []
    for i in range(arr.shape[1]):
        column = arr[:, i]
        scale, _ = infer_time_unit_and_scale(column)
        median_dt = _median_dt_us(np.sort(column.astype(np.float64) * scale))
        monotonic = np.all(np.diff(np.sort(column)) >= 0)
        score = 0.0
        if typical_time := median_dt:
            score += abs(median_dt - 50.0)
        if monotonic:
            score -= 1.0
        candidates.append((i, score))
    candidates.sort(key=lambda x: x[1])
    t_idx = candidates[0][0]

    remaining = [i for i in range(arr.shape[1]) if i != t_idx]
    x_idx = remaining[0]
    y_idx = remaining[1]
    p_idx = remaining[2] if len(remaining) > 2 else remaining[-1]

    return {
        "t": arr[:, t_idx],
        "x": arr[:, x_idx],
        "y": arr[:, y_idx],
        "p": arr[:, p_idx],
    }


def map_polarity(p: np.ndarray, keep_sign: bool) -> np.ndarray:
    p_arr = np.asarray(p)
    unique_vals = np.unique(p_arr)
    if keep_sign:
        if set(unique_vals.tolist()) <= {-1, 1}:
            return p_arr.astype(np.int8)
        # Map {0,1} to {-1,+1}
        mapped = np.where(p_arr > 0, 1, -1)
        return mapped.astype(np.int8)

    # default to {0,1}
    if set(unique_vals.tolist()) <= {0, 1}:
        return p_arr.astype(np.int8)
    mapped = np.where(p_arr > 0, 1, 0)
    return mapped.astype(np.int8)


def process_events(fields: FieldMapping, normalize_t: bool, keep_sign: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    t_raw = np.asarray(fields["t"])
    scale, reason = infer_time_unit_and_scale(t_raw)
    t_us = np.rint(t_raw.astype(np.float64) * scale).astype(np.int64)

    sort_idx = np.argsort(t_us)
    t_us = t_us[sort_idx]

    x = np.asarray(fields["x"]).astype(np.int32)[sort_idx]
    y = np.asarray(fields["y"]).astype(np.int32)[sort_idx]
    p = map_polarity(fields["p"], keep_sign=keep_sign)[sort_idx]

    if normalize_t and t_us.size:
        t_us = t_us - t_us[0]

    return t_us, x, y, p, reason


def save_npz(out_path: Path, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, t=t, x=x, y=y, p=p)


def describe_range(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    return f"[{arr.min()}..{arr.max()}]"


def process_file(path: Path, args: argparse.Namespace) -> bool:
    try:
        events = load_events_from_aedat(path)
        fields = infer_event_fields(events)
        t_us, x, y, p, reason = process_events(
            fields, normalize_t=args.normalize_t, keep_sign=args.keep_polarity_sign
        )

        out_path = Path(args.out_dir) / path.relative_to(args.in_dir)
        out_path = out_path.with_suffix(".npz")
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing file (use --overwrite to replace): {out_path}")
            return True

        save_npz(out_path, t=t_us, x=x, y=y, p=p)

        if args.verbose:
            info = {
                "events": len(t_us),
                "t_unit_reason": reason,
                "t_range": describe_range(t_us),
                "x_range": describe_range(x),
                "y_range": describe_range(y),
                "p_values": np.unique(p).tolist(),
                "output": str(out_path),
            }
            print(f"Processed {path}: {info}")
        return True
    except Exception as exc:
        print(f"Failed to process {path}: {exc}")
        return False


def find_aedat_files(in_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.aedat4" if recursive else "*.aedat4"
    return sorted(Path(in_dir).glob(pattern))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.in_dir = os.fspath(args.in_dir)
    args.out_dir = os.fspath(args.out_dir)

    in_dir = Path(args.in_dir)
    files = find_aedat_files(in_dir, recursive=args.recursive)
    if not files:
        print(f"No .aedat4 files found in {in_dir} (recursive={args.recursive}).")
        return

    success = 0
    for path in files:
        if process_file(path, args):
            success += 1
    print(f"Conversion completed: {success}/{len(files)} files succeeded, {len(files) - success} failed.")


if __name__ == "__main__":
    main()

# Usage notes:
# - In PyCharm, simply click Run to execute with the CONFIG defaults.
# - Command-line overrides remain available, e.g.:
#     python tools/convert_aedat4_to_npz.py --in-dir data/in --out-dir data/out --overwrite
