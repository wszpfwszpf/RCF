"""Convert AEDAT4 event streams to NPZ (t, x, y, p) without CLI arguments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ===================== 配置区（必须在文件顶部） =====================
IN_DIR = "data/origin-aedat4"
OUT_DIR = "data/converted-npz"
NORMALIZE_T = True          # t_us -= t_us[0]
OVERWRITE = False
VERBOSE = True
KEEP_POLARITY_SIGN = False  # False输出p为0/1；True输出p为{-1,+1}
RECURSIVE = True            # 递归扫描子目录
# ===============================================================


# --------------------------- 工具函数 ---------------------------
def list_aedat4_files(in_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.aedat4" if recursive else "*.aedat4"
    return [p for p in in_dir.glob(pattern) if p.is_file()]


def _has_module(module_name: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module_name) is not None


def _batch_to_array(batch) -> np.ndarray:
    """Best-effort conversion of reader batches to numpy arrays."""
    if isinstance(batch, np.ndarray):
        return batch
    if hasattr(batch, "numpy"):
        return batch.numpy()

    collected: Dict[str, np.ndarray] = {}
    for candidate in ("timestamp", "timestamps", "t", "time", "ts"):
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
    for candidate in ("p", "polarity", "pol", "on", "sign"):
        if hasattr(batch, candidate):
            collected["p"] = np.asarray(getattr(batch, candidate))
            break

    if collected and len({len(v) for v in collected.values()}) == 1:
        dtype = [(k, np.asarray(v).dtype) for k, v in collected.items()]
        arr = np.empty(len(next(iter(collected.values()))), dtype=dtype)
        for k, v in collected.items():
            arr[k] = np.asarray(v)
        return arr

    if isinstance(batch, Sequence) and batch and not isinstance(batch, (bytes, str)):
        first = batch[0]
        if hasattr(first, "timestamp") and hasattr(first, "x") and hasattr(first, "y"):
            timestamps = [getattr(item, "timestamp") for item in batch]
            xs = [getattr(item, "x") for item in batch]
            ys = [getattr(item, "y") for item in batch]
            ps = [getattr(item, "polarity", getattr(item, "p", 0)) for item in batch]
            return np.array(list(zip(timestamps, xs, ys, ps)), dtype=[("t", float), ("x", int), ("y", int), ("p", int)])

    raise TypeError("Unsupported batch type for conversion to numpy array")


def _stack_event_batches(batches: Iterable) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for batch in batches:
        try:
            arr = _batch_to_array(batch)
        except Exception:
            continue
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.empty((0, 4))
    return np.concatenate(arrays)


def read_events_from_aedat4(path: Path) -> Tuple[np.ndarray, str]:
    if _has_module("dv"):
        from dv import AedatFile  # type: ignore

        with AedatFile(str(path)) as f:
            try:
                source = f["events"]
            except Exception:
                source = f.events()
            events = _stack_event_batches(source)
            return events, "dv"

    if _has_module("dv_processing"):
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

    raise RuntimeError(
        "Cannot read .aedat4: install 'dv' (pip install dv) or 'dv-processing' (pip install dv-processing)."
    )


def _collect_columns(raw_events: np.ndarray) -> Dict[str, np.ndarray]:
    columns: Dict[str, np.ndarray] = {}
    if raw_events.dtype.names:
        for name in raw_events.dtype.names:
            columns[name] = np.asarray(raw_events[name]).reshape(-1)
    elif raw_events.ndim == 2 and raw_events.shape[1] >= 4:
        for idx in range(raw_events.shape[1]):
            columns[f"col{idx}"] = np.asarray(raw_events[:, idx]).reshape(-1)
    else:
        raise ValueError("Unsupported event array format for field extraction")
    return columns


def _match_by_name(columns: Dict[str, np.ndarray]) -> Dict[str, Optional[str]]:
    mapping = {"t": None, "x": None, "y": None, "p": None}
    name_sets = {
        "t": {"timestamp", "timestamps", "t", "time", "ts"},
        "x": {"x", "xs"},
        "y": {"y", "ys"},
        "p": {"p", "polarity", "pol", "on", "sign"},
    }
    lower_to_original = {name.lower(): name for name in columns}
    for key, aliases in name_sets.items():
        for alias in aliases:
            if alias in lower_to_original:
                mapping[key] = lower_to_original[alias]
                break
    return mapping


def _candidate_polarity(columns: Dict[str, np.ndarray]) -> Optional[str]:
    best: Optional[str] = None
    for name, values in columns.items():
        uniques = np.unique(values)
        if uniques.size <= 3 and set(uniques).issubset({-1, 0, 1}):
            best = name
            break
    return best


def _candidate_coordinates(columns: Dict[str, np.ndarray], exclude: List[str]) -> Tuple[Optional[str], Optional[str]]:
    coords: List[Tuple[str, float]] = []
    for name, values in columns.items():
        if name in exclude:
            continue
        if values.size == 0:
            continue
        max_v = float(np.max(values))
        min_v = float(np.min(values))
        if min_v >= 0 and max_v < 5000:
            coords.append((name, max_v))
    coords.sort(key=lambda item: item[1], reverse=True)
    if len(coords) >= 2:
        return coords[0][0], coords[1][0]
    if len(coords) == 1:
        return coords[0][0], None
    return None, None


def _monotonic_score(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    diffs = np.diff(values)
    ratio = np.mean(diffs >= 0)
    return float(ratio)


def _candidate_time(columns: Dict[str, np.ndarray], exclude: List[str]) -> Optional[str]:
    best_name: Optional[str] = None
    best_score = -np.inf
    for name, values in columns.items():
        if name in exclude:
            continue
        ratio = _monotonic_score(np.asarray(values))
        magnitude = float(np.max(values)) if values.size else 0.0
        score = ratio * 2.0 + magnitude / (magnitude + 1.0)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def extract_txyp(raw_events: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if raw_events.size == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int8),
        )

    columns = _collect_columns(raw_events)
    mapping = _match_by_name(columns)

    if not all(mapping.values()):
        guessed_p = _candidate_polarity(columns)
        x_name, y_name = _candidate_coordinates(columns, exclude=[guessed_p] if guessed_p else [])
        t_name = _candidate_time(columns, exclude=[n for n in [guessed_p, x_name, y_name] if n])
        if mapping["p"] is None:
            mapping["p"] = guessed_p
        if mapping["x"] is None:
            mapping["x"] = x_name
        if mapping["y"] is None:
            mapping["y"] = y_name
        if mapping["t"] is None:
            mapping["t"] = t_name

    if not all(mapping.values()):
        ordered = ", ".join([f"{k}={v}" for k, v in mapping.items()])
        raise ValueError(f"Failed to infer fields, mapping={ordered}")

    t = np.asarray(columns[mapping["t"]], dtype=np.float64)
    x = np.asarray(columns[mapping["x"]], dtype=np.int32)
    y = np.asarray(columns[mapping["y"]], dtype=np.int32)
    p = np.asarray(columns[mapping["p"]], dtype=np.int8)

    return t, x, y, p


def _median_positive_dt(t_us: np.ndarray) -> float:
    if t_us.size < 2:
        return float("inf")
    t0 = t_us - t_us[0]
    m = min(2000, t0.size)
    diffs = np.diff(t0[:m])
    pos = diffs[diffs > 0]
    if pos.size == 0:
        return float("inf")
    return float(np.median(pos))


def infer_scale_to_us(t_raw: np.ndarray) -> Tuple[float, str]:
    if t_raw.size == 0:
        return 1.0, "empty timestamps -> scale=1"

    candidates = [1.0, 1e3, 1e6]
    target_primary = (1.0, 500.0)
    target_relaxed = (1.0, 2000.0)
    best_scale = candidates[0]
    best_score = -np.inf
    best_reason = ""
    max_raw = float(np.max(t_raw))
    is_float = np.issubdtype(t_raw.dtype, np.floating)

    for scale in candidates:
        t_us = np.round(t_raw * scale)
        med_dt = _median_positive_dt(t_us)
        if np.isinf(med_dt):
            score = -np.inf
        elif target_primary[0] <= med_dt <= target_primary[1]:
            score = 5.0 - abs(np.log10(med_dt / 50.0))
        elif target_relaxed[0] <= med_dt <= target_relaxed[1]:
            score = 3.0 - abs(np.log10(med_dt / 500.0))
        else:
            score = -abs(np.log10(med_dt / 500.0))

        # tie-break using magnitude and dtype hints
        if is_float and max_raw < 2.0 and scale == 1e6:
            score += 0.5
        if max_raw >= 1e6 and scale == 1.0:
            score += 0.3

        if score > best_score:
            best_score = score
            best_scale = scale
            best_reason = f"median dt={med_dt:.2f}us (scale {scale})"

    reason = best_reason or "fallback to scale=1"
    return best_scale, reason


def normalize_and_sort(
    t_us: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, normalize_t: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if t_us.size == 0:
        return (
            t_us.astype(np.int64),
            x.astype(np.int32),
            y.astype(np.int32),
            p.astype(np.int8),
        )

    order = np.argsort(t_us, kind="stable")
    t_sorted = np.asarray(t_us[order], dtype=np.int64)
    x_sorted = np.asarray(x[order], dtype=np.int32)
    y_sorted = np.asarray(y[order], dtype=np.int32)
    p_sorted = np.asarray(p[order], dtype=np.int8)

    if normalize_t:
        t_sorted = t_sorted - t_sorted[0]

    return t_sorted, x_sorted, y_sorted, p_sorted


def save_npz(out_path: Path, t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, t=t, x=x, y=y, p=p)


# --------------------------- 业务逻辑 ---------------------------
def _process_single_file(in_path: Path, out_path: Path) -> bool:
    if out_path.exists() and not OVERWRITE:
        if VERBOSE:
            print(f"[skip] {in_path} -> {out_path} (exists)")
        return True

    try:
        raw_events, reader = read_events_from_aedat4(in_path)
        t_raw, x_raw, y_raw, p_raw = extract_txyp(raw_events)
        scale, reason = infer_scale_to_us(t_raw)
        t_us = np.round(t_raw * scale).astype(np.int64)

        if KEEP_POLARITY_SIGN:
            p_processed = np.where(p_raw >= 0, 1, -1).astype(np.int8)
        else:
            p_processed = np.where(p_raw > 0, 1, 0).astype(np.int8)

        t, x, y, p = normalize_and_sort(t_us, x_raw, y_raw, p_processed, NORMALIZE_T)
        save_npz(out_path, t, x, y, p)

        if VERBOSE:
            t_range = (t.min() if t.size else 0, t.max() if t.size else 0)
            x_range = (x.min() if x.size else 0, x.max() if x.size else 0)
            y_range = (y.min() if y.size else 0, y.max() if y.size else 0)
            p_set = set(np.unique(p).tolist())
            print(
                f"[ok] {in_path.name} events={len(t)} scale={scale} ({reason}) "
                f"t={t_range} x={x_range} y={y_range} p_set={p_set} -> {out_path} (reader={reader})"
            )
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[fail] {in_path}: {exc}")
        return False


def main() -> None:
    in_dir = Path(IN_DIR)
    out_dir = Path(OUT_DIR)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    files = list_aedat4_files(in_dir, RECURSIVE)
    if VERBOSE:
        print(f"Found {len(files)} .aedat4 files under {in_dir} (recursive={RECURSIVE})")

    success = 0
    for in_file in files:
        relative = in_file.relative_to(in_dir)
        out_file = out_dir / relative.with_suffix(".npz")
        if _process_single_file(in_file, out_file):
            success += 1

    failed = len(files) - success
    print(f"Completed: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()

# 在 PyCharm 中直接点击 Run 即可。
