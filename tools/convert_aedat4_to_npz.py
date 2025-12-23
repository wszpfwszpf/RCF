import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def list_files(in_dir: Path) -> List[Path]:
    files: List[Path] = []
    for root, _, filenames in os.walk(in_dir):
        for name in filenames:
            if name.lower().endswith(".aedat4"):
                files.append(Path(root) / name)
    return files


def _dict_to_structured_array(data: Dict[str, np.ndarray]) -> np.ndarray:
    length = len(next(iter(data.values()))) if data else 0
    dtype = [(key, np.asarray(value).dtype) for key, value in data.items()]
    array = np.empty(length, dtype=dtype)
    for key, value in data.items():
        array[key] = np.asarray(value)
    return array


def _batch_to_array(batch) -> np.ndarray:
    if isinstance(batch, np.ndarray):
        return batch
    if hasattr(batch, "numpy"):
        return batch.numpy()

    collected: Dict[str, np.ndarray] = {}
    for candidate in ("timestamps", "timestamp", "t", "time", "ts"):
        if hasattr(batch, candidate):
            collected["t"] = np.asarray(getattr(batch, candidate))
            break
    for candidate in ("x", "xs"):
        if hasattr(batch, candidate):
            collected["x"] = np.asarray(getattr(batch, candidate))
            break
    for candidate in ("y", "ys"):
        if hasattr(batch, candidate):
            collected["y"] = np.asarray(getattr(batch, candidate))
            break
    for candidate in ("p", "polarity", "pol", "on", "sign", "polarity"):  # duplicate for clarity
        if hasattr(batch, candidate):
            collected["p"] = np.asarray(getattr(batch, candidate))
            break

    if collected:
        lengths = {len(v) for v in collected.values()}
        if len(lengths) == 1:
            return _dict_to_structured_array(collected)

    if isinstance(batch, Sequence) and batch and not isinstance(batch, (bytes, str)):
        sample = batch[0]
        if hasattr(sample, "timestamp") and hasattr(sample, "x") and hasattr(sample, "y"):
            timestamps = [getattr(item, "timestamp") for item in batch]
            xs = [getattr(item, "x") for item in batch]
            ys = [getattr(item, "y") for item in batch]
            ps = [getattr(item, "polarity", getattr(item, "p", 0)) for item in batch]
            return _dict_to_structured_array({"t": np.array(timestamps), "x": np.array(xs), "y": np.array(ys), "p": np.array(ps)})

    raise TypeError("Unsupported batch type for conversion to numpy array")


def _stack_event_batches(batches: Iterable) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for batch in batches:
        try:
            arr = _batch_to_array(batch)
            if arr.size:
                arrays.append(arr)
        except Exception:
            continue
    if not arrays:
        return np.empty((0, 4))
    return np.concatenate(arrays)


def read_aedat4_events(path: Path) -> Tuple[np.ndarray, str]:
    if has_module("dv"):
        from dv import AedatFile  # type: ignore

        with AedatFile(str(path)) as f:
            try:
                source = f["events"]
            except Exception:
                source = f.events()
            events = _stack_event_batches(source)
            return events, "dv"

    if has_module("dv_processing"):
        from dv_processing import Aedat4Reader  # type: ignore

        reader = Aedat4Reader(str(path))
        batches = []
        while True:
            batch = reader.get_next_event_batch()
            if batch is None:
                break
            batches.append(batch)
        events = _stack_event_batches(batches)
        return events, "dv_processing"

    raise RuntimeError("Neither 'dv' nor 'dv_processing' packages are available. Please install one of them to read .aedat4 files.")


def _candidate_polarity(columns: Dict[str, np.ndarray]) -> Optional[str]:
    candidates = []
    for name, values in columns.items():
        uniques = np.unique(values)
        if uniques.size <= 3 and set(uniques).issubset({-1, 0, 1}):
            candidates.append(name)
    if candidates:
        return candidates[0]
    return None


def _candidate_coordinates(columns: Dict[str, np.ndarray], exclude: List[str]) -> Tuple[Optional[str], Optional[str]]:
    coord_candidates = []
    for name, values in columns.items():
        if name in exclude:
            continue
        if values.size == 0:
            continue
        min_v = np.min(values)
        max_v = np.max(values)
        if min_v >= 0 and max_v < 100000:  # reasonable coordinate range
            coord_candidates.append((name, max_v))
    coord_candidates.sort(key=lambda item: item[1])
    if len(coord_candidates) >= 2:
        return coord_candidates[0][0], coord_candidates[1][0]
    if len(coord_candidates) == 1:
        return coord_candidates[0][0], None
    return None, None


def _monotonic_ratio(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    diffs = np.diff(values)
    return float(np.mean(diffs >= 0))


def _candidate_time(columns: Dict[str, np.ndarray], exclude: List[str]) -> Optional[str]:
    best_name: Optional[str] = None
    best_score = -np.inf
    for name, values in columns.items():
        if name in exclude:
            continue
        ratio = _monotonic_ratio(np.asarray(values))
        magnitude = float(np.max(values)) if values.size else 0.0
        score = ratio * 2.0 + magnitude / (1e6 + magnitude)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def infer_fields(raw_events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if raw_events.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int8),
        )

    columns: Dict[str, np.ndarray] = {}
    if raw_events.dtype.names:
        for name in raw_events.dtype.names:
            columns[name] = np.asarray(raw_events[name]).reshape(-1)
    elif raw_events.ndim == 2 and raw_events.shape[1] >= 4:
        for idx in range(raw_events.shape[1]):
            columns[f"col{idx}"] = np.asarray(raw_events[:, idx]).reshape(-1)
    else:
        raise ValueError("Unsupported event array format for field inference")

    p_name = _candidate_polarity(columns)
    x_name, y_name = _candidate_coordinates(columns, exclude=[p_name] if p_name else [])
    t_name = _candidate_time(columns, exclude=[name for name in [p_name, x_name, y_name] if name])

    missing = [("t", t_name), ("x", x_name), ("y", y_name), ("p", p_name)]
    if any(name is None for _, name in missing):
        ordered = ", ".join([f"{key}={name}" for key, name in missing])
        raise ValueError(f"Failed to infer all fields; current mapping: {ordered}")

    t = np.asarray(columns[t_name], dtype=np.float64)
    x = np.asarray(columns[x_name], dtype=np.int32)
    y = np.asarray(columns[y_name], dtype=np.int32)
    p = np.asarray(columns[p_name], dtype=np.int8)

    return t, x, y, p


def infer_time_scale(timestamps: np.ndarray) -> Tuple[np.ndarray, str]:
    if timestamps.size == 0:
        return timestamps.astype(np.int64), "microseconds"

    max_v = float(np.max(timestamps))
    diffs = np.diff(timestamps[: min(1000, timestamps.size)]) if timestamps.size > 1 else np.array([0.0])
    median_diff = float(np.median(np.abs(diffs))) if diffs.size else 0.0

    scale = 1.0
    unit = "microseconds"
    if max_v < 1e3:
        if median_diff < 1:
            scale = 1e6
            unit = "seconds_to_microseconds"
        else:
            scale = 1e3
            unit = "milliseconds_to_microseconds"
    elif max_v < 1e6:
        if median_diff < 1:
            scale = 1e6
            unit = "seconds_to_microseconds"
        elif median_diff < 5e3:
            scale = 1e3
            unit = "milliseconds_to_microseconds"
    else:
        unit = "microseconds"

    scaled = (timestamps * scale).astype(np.int64)
    return scaled, unit


def normalize_and_sort(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, normalize_t: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    x_sorted = x[order]
    y_sorted = y[order]
    p_sorted = p[order]

    if normalize_t and t_sorted.size:
        t_sorted = t_sorted - t_sorted[0]
    return t_sorted, x_sorted, y_sorted, p_sorted


def save_npz(out_path: Path, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
    np.savez_compressed(out_path, t=t, x=x, y=y, p=p)


def process_file(
    in_file: Path,
    out_file: Path,
    normalize_t_flag: bool,
    keep_polarity_sign: bool,
    overwrite: bool,
    verbose: bool,
) -> bool:
    if out_file.exists() and not overwrite:
        if verbose:
            print(f"[skip] {in_file} -> {out_file} (exists)")
        return True

    try:
        raw_events, reader_name = read_aedat4_events(in_file)
        t_raw, x_raw, y_raw, p_raw = infer_fields(raw_events)
        t_scaled, time_unit = infer_time_scale(t_raw)
        p_processed = np.where(p_raw < 0, 0, p_raw) if not keep_polarity_sign else np.where(p_raw < 0, -1, p_raw)
        p_processed = p_processed.astype(np.int8)
        p_processed = np.where(p_processed > 1, 1, p_processed)

        t_final, x_final, y_final, p_final = normalize_and_sort(t_scaled, x_raw, y_raw, p_processed, normalize_t_flag)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        save_npz(out_file, t_final, x_final, y_final, p_final)

        if verbose:
            summary = (
                f"[ok] {in_file.name} events={len(t_final)} unit={time_unit} "
                f"t=[{t_final.min() if t_final.size else 0}, {t_final.max() if t_final.size else 0}] "
                f"x=[{x_final.min() if x_final.size else 0}, {x_final.max() if x_final.size else 0}] "
                f"y=[{y_final.min() if y_final.size else 0}, {y_final.max() if y_final.size else 0}] "
                f"p_set={set(np.unique(p_final).tolist())} -> {out_file} (reader={reader_name})"
            )
            print(summary)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[fail] {in_file}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AEDAT4 event files to NPZ (t,x,y,p)")
    parser.add_argument("--in_dir", type=str, default="data/origin-aedat4", help="Input directory containing .aedat4 files")
    parser.add_argument("--out_dir", type=str, default="data/converted-npz", help="Output directory for .npz files")
    parser.add_argument("--normalize_t", type=int, default=1, help="Whether to normalize timestamps to start at zero (1/0)")
    parser.add_argument("--keep_polarity_sign", type=int, default=0, help="Keep polarity as {-1,+1}; otherwise map to {0,1}")
    parser.add_argument("--overwrite", type=int, default=0, help="Overwrite existing .npz files")
    parser.add_argument("--verbose", type=int, default=1, help="Print verbose logs")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    normalize_t_flag = bool(args.normalize_t)
    keep_polarity_sign = bool(args.keep_polarity_sign)
    overwrite = bool(args.overwrite)
    verbose = bool(args.verbose)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    files = list_files(in_dir)
    if verbose:
        print(f"Found {len(files)} .aedat4 files under {in_dir}")

    success = 0
    for in_file in files:
        relative = in_file.relative_to(in_dir)
        out_file = out_dir / relative.with_suffix(".npz")
        if process_file(in_file, out_file, normalize_t_flag, keep_polarity_sign, overwrite, verbose):
            success += 1

    failed = len(files) - success
    print(f"Completed: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()

# Example:
# python tools/convert_aedat4_to_npz.py --in_dir data/origin-aedat4 --out_dir data/converted-npz --normalize_t 1 --overwrite 0
