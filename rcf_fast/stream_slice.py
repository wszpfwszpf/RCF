# rcf_fast/stream_slice.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Any

import numpy as np


def _require_dvp():
    """
    dv-processing official python binding.
    """
    try:
        import dv_processing as dv  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import `dv_processing`.\n"
            "Please install dv-processing python bindings in your current env (py310 recommended)."
        ) from e
    return dv


@dataclass
class BatchInfo:
    """
    Lightweight metadata for debugging/verification (Step3 only).
    """
    n_events: int
    t_first: int
    t_last: int
    dt_us: int


class NumpyTimeSurface:
    """
    A minimal time-surface state for Step3:
    - stores last timestamp (us) for each pixel
    - only updates, no visualization / no decay / no score computation

    This keeps Step3 independent from dv.TimeSurface (optional).
    """
    def __init__(self, resolution_wh: Tuple[int, int]):
        W, H = resolution_wh
        if W <= 0 or H <= 0:
            raise ValueError(f"Invalid resolution (W,H)=({W},{H})")
        self.W = int(W)
        self.H = int(H)
        self.last_ts = np.full((self.H, self.W), -1, dtype=np.int64)

    def update_xy_t(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> None:
        # Assumes x,y within bounds; t int64
        self.last_ts[y, x] = t

    def update_from_eventstore(self, events: Any) -> BatchInfo:
        """
        Update using dv.EventStore (or any iterable of events).
        Returns minimal batch stats for logging.
        """
        # Try fast numpy extraction first
        x, y, t = _eventstore_to_xyt_numpy(events)
        if x is None:
            # Fallback to per-event iteration
            x_list, y_list, t_list = [], [], []
            for e in events:
                ex, ey, et = _read_single_event(e)
                if ex is None:
                    continue
                x_list.append(ex)
                y_list.append(ey)
                t_list.append(et)
            if len(x_list) == 0:
                return BatchInfo(n_events=0, t_first=0, t_last=0, dt_us=0)
            x = np.asarray(x_list, dtype=np.int32)
            y = np.asarray(y_list, dtype=np.int32)
            t = np.asarray(t_list, dtype=np.int64)

        # Clamp to valid range defensively (should already be valid)
        x = np.clip(x, 0, self.W - 1)
        y = np.clip(y, 0, self.H - 1)

        self.update_xy_t(x, y, t)

        t_first = int(t.min()) if t.size else 0
        t_last = int(t.max()) if t.size else 0
        return BatchInfo(n_events=int(t.size), t_first=t_first, t_last=t_last, dt_us=int(t_last - t_first))


def iter_event_batches(
    aedat4_path: str | Path,
) -> Iterator[Any]:
    """
    Stream-read aedat4 using dv_processing MonoCameraRecording.
    Yields dv.EventStore batches.
    """
    dv = _require_dvp()
    aedat4_path = str(Path(aedat4_path))

    reader = dv.io.MonoCameraRecording(aedat4_path)
    if not reader.isEventStreamAvailable():
        raise RuntimeError(f"No event stream available in: {aedat4_path}")

    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is not None:
            yield events


def run_time_slicer(
    aedat4_path: str | Path,
    dt_ms: int,
    on_bin: Callable[[Any, BatchInfo], None],
    *,
    max_bins: Optional[int] = None,
    use_dv_timesurface: bool = False,
) -> None:
    """
    Step3 main entry:
    - stream read (EventStore batches)
    - slice into strict dt_ms bins using dv.EventStreamSlicer
    - update a time-surface state (dv.TimeSurface optional; default uses numpy state)
    - call on_bin(events_10ms, info) for each bin

    NOTE:
    - Step3 does NOT compute scores, does NOT drop events, does NOT visualize.
    """
    dv = _require_dvp()
    aedat4_path = str(Path(aedat4_path))

    # Resolution (W,H)
    reader = dv.io.MonoCameraRecording(aedat4_path)
    if not reader.isEventStreamAvailable():
        raise RuntimeError(f"No event stream available in: {aedat4_path}")
    res = reader.getEventResolution()
    # dv returns something like (W,H)
    W, H = int(res[0]), int(res[1])

    # Choose time-surface backend
    dv_ts = None
    np_ts = None
    if use_dv_timesurface and hasattr(dv, "TimeSurface"):
        dv_ts = dv.TimeSurface(res)
    else:
        np_ts = NumpyTimeSurface((W, H))

    slicer = dv.EventStreamSlicer()

    bin_counter = {"k": 0}

    def _callback(events_bin: Any):
        # Update time surface + gather stats
        if dv_ts is not None:
            dv_ts.accept(events_bin)
            # Still compute minimal stats for logging (best-effort)
            info = _best_effort_batch_info(events_bin)
        else:
            assert np_ts is not None
            info = np_ts.update_from_eventstore(events_bin)

        bin_counter["k"] += 1
        on_bin(events_bin, info)

        if max_bins is not None and bin_counter["k"] >= max_bins:
            # Stop slicing by raising a private exception to break outer loop
            raise _StopSlicing()

    slicer.doEveryTimeInterval(timedelta(milliseconds=int(dt_ms)), _callback)

    try:
        # stream read loop (reusing the same reader object)
        while reader.isRunning():
            events = reader.getNextEventBatch()
            if events is not None:
                slicer.accept(events)
    except _StopSlicing:
        return


class _StopSlicing(Exception):
    pass


# ---------------------------
# Helpers
# ---------------------------

def _best_effort_batch_info(events: Any) -> BatchInfo:
    x, y, t = _eventstore_to_xyt_numpy(events)
    if x is None or t is None or t.size == 0:
        # fallback: iterate limited
        t_first, t_last, n = None, None, 0
        for e in events:
            ex, ey, et = _read_single_event(e)
            if ex is None:
                continue
            if t_first is None:
                t_first = et
            t_last = et
            n += 1
        if t_first is None:
            return BatchInfo(n_events=0, t_first=0, t_last=0, dt_us=0)
        return BatchInfo(n_events=n, t_first=int(t_first), t_last=int(t_last), dt_us=int(t_last - t_first))

    return BatchInfo(
        n_events=int(t.size),
        t_first=int(t.min()),
        t_last=int(t.max()),
        dt_us=int(t.max() - t.min()),
    )


def _eventstore_to_xyt_numpy(events: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Best-effort conversion:
    - If dv.EventStore exposes numpy(), try it.
    - Otherwise return (None,None,None).
    """
    # Common patterns: events.numpy() / events.toNumpy() / events.asNumpy()
    for fn in ("numpy", "toNumpy", "asNumpy"):
        if hasattr(events, fn):
            try:
                arr = getattr(events, fn)()
                # Expect structured array with fields: x,y,t (or timestamp)
                if isinstance(arr, np.ndarray) and arr.dtype.names:
                    names = set(arr.dtype.names)
                    if "x" in names and "y" in names:
                        x = arr["x"].astype(np.int32, copy=False)
                        y = arr["y"].astype(np.int32, copy=False)
                        if "t" in names:
                            t = arr["t"].astype(np.int64, copy=False)
                        elif "timestamp" in names:
                            t = arr["timestamp"].astype(np.int64, copy=False)
                        else:
                            return None, None, None
                        return x, y, t
            except Exception:
                pass
    return None, None, None


def _read_single_event(e: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Robustly read (x,y,t) from a single event object.
    Handles different bindings:
    - attributes: e.x, e.y, e.timestamp / e.t
    - methods: e.x(), e.y(), e.timestamp()
    """
    def _get(obj: Any, key: str):
        if not hasattr(obj, key):
            return None
        v = getattr(obj, key)
        return v() if callable(v) else v

    x = _get(e, "x")
    y = _get(e, "y")
    t = _get(e, "timestamp")
    if t is None:
        t = _get(e, "t")

    if x is None or y is None or t is None:
        return None, None, None
    return int(x), int(y), int(t)
