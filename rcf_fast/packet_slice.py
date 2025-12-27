# rcf_fast/packet_slice.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from rcf_fast.timeinterval_slice import _require_dvp  # reuse dv_processing import helper


@dataclass
class PacketInfo:
    """
    Fixed-count packet metadata for ESR slicing.
    raw_begin/raw_end are global indices in the *raw* event stream (left-closed, right-open).
    """
    packet_id: int
    raw_begin: int
    raw_end: int
    n_events: int
    t_first: int
    t_last: int
    dt_us: int


def _best_effort_batch_info(events: Any) -> Tuple[int, int, int]:
    """
    Return (n_events, t_first, t_last) best-effort.
    """
    # Try numpy conversion (fast)
    x, y, t = _eventstore_to_xyt_numpy(events)
    if t is not None and t.size > 0:
        return int(t.size), int(t.min()), int(t.max())

    # Fallback: iterate
    t_first = None
    t_last = None
    n = 0
    for e in events:
        ex, ey, et = _read_single_event(e)
        if et is None:
            continue
        if t_first is None:
            t_first = et
        t_last = et
        n += 1
    if t_first is None:
        return 0, 0, 0
    return n, int(t_first), int(t_last)


def _eventstore_to_xyt_numpy(events: Any):
    """
    Best-effort conversion:
    returns (x,y,t) numpy arrays or (None,None,None)
    """
    for fn in ("numpy", "toNumpy", "asNumpy"):
        if hasattr(events, fn):
            try:
                arr = getattr(events, fn)()
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


def _read_single_event(e: Any):
    """
    Read (x,y,t) from event object, tolerant to attribute/method styles.
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


def run_count_slicer(
    aedat4_path: str | Path,
    n_per_packet: int,
    on_packet: Callable[[Any, PacketInfo], None],
    *,
    max_packets: Optional[int] = None,
) -> None:
    """
    Step3-count:
    - stream read aedat4 with MonoCameraRecording
    - slice into fixed-count packets using dv.EventStreamSlicer.doEveryNumberOfElements()
    - for each packet, call on_packet(events_packet, PacketInfo)

    This is intended for ESR: packets are defined on the RAW stream.
    """
    if n_per_packet <= 0:
        raise ValueError(f"n_per_packet must be > 0, got {n_per_packet}")

    dv = _require_dvp()
    aedat4_path = str(Path(aedat4_path))

    reader = dv.io.MonoCameraRecording(aedat4_path)
    if not reader.isEventStreamAvailable():
        raise RuntimeError(f"No event stream available in: {aedat4_path}")

    slicer = dv.EventStreamSlicer()

    state = {
        "packet_id": 0,
        "raw_index": 0,   # global raw event index (counts how many raw events have been emitted into packets)
    }

    def _callback(events_packet: Any):
        # Compute metadata
        n_events, t_first, t_last = _best_effort_batch_info(events_packet)
        raw_begin = state["raw_index"]
        raw_end = raw_begin + n_events  # should equal raw_begin + n_per_packet (unless last tail)
        state["raw_index"] = raw_end

        state["packet_id"] += 1
        info = PacketInfo(
            packet_id=state["packet_id"],
            raw_begin=raw_begin,
            raw_end=raw_end,
            n_events=n_events,
            t_first=t_first,
            t_last=t_last,
            dt_us=(t_last - t_first) if n_events > 0 else 0,
        )
        on_packet(events_packet, info)

        if max_packets is not None and state["packet_id"] >= max_packets:
            raise _StopSlicing()

    # official API: slice by number of elements/events
    slicer.doEveryNumberOfElements(int(n_per_packet), _callback)

    try:
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is not None:
                slicer.accept(batch)
    except _StopSlicing:
        return


class _StopSlicing(Exception):
    pass
