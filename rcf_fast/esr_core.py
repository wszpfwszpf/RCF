# rcf_fast/esr_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


# =============================================================================
# ESR (V1) defaults / hyper-params (match official metric.py behavior)
# =============================================================================
SENSOR_W: int = 346
SENSOR_H: int = 260

DEFAULT_N_PACKET: int = 30000
M_RATIO: float = 2.0 / 3.0

_EPS: float = float(np.spacing(1))


@dataclass
class ESRStats:
    mean_esr: float
    std_esr: float
    n_packets: int


def _require_dvp():
    """
    Lazy import dv_processing to keep import errors localized.
    """
    try:
        import dv_processing as dv  # type: ignore
        return dv
    except Exception as e:
        raise ImportError(
            "Failed to import dv_processing. Please install dv-processing Python bindings "
            "in your current environment (e.g., conda py310)."
        ) from e


def _esr_v1_from_count_surface(n_map: np.ndarray, N: int, W: int, H: int) -> float:
    """
    ESR V1 as official metric.py:

        NTSS = sum_i n_i(n_i-1) / (N(N-1))
        L_N  = K - sum_i (1 - M/N)^(n_i),  M = floor(2/3 * N)
        ESR  = sqrt(NTSS * L_N)

    Notes:
      - n_map is per-pixel count map (ignore polarity, contribution=1, no decay).
    """
    if N < 2:
        return 0.0

    n = n_map.astype(np.float64, copy=False)
    K = float(W * H)

    M = int(np.floor(M_RATIO * N))
    base = 1.0 - (float(M) / (float(N) + _EPS))

    ntss = float((n * (n - 1.0)).sum()) / (float(N) + _EPS) / (float(N - 1) + _EPS)
    ln = float(K - np.power(base, n).sum())

    # Guard against tiny negative from numeric noise
    if ntss < 0.0:
        ntss = 0.0
    if ln < 0.0:
        ln = 0.0

    return float(np.sqrt(ntss * ln))


def compute_esr_v1(
    events: Any,
    *,
    resolution: Tuple[int, int] = (SENSOR_W, SENSOR_H),
) -> float:
    """
    Compute ESR V1 for a single dv_processing.EventStore (one packet),
    aligned to official metric.py:

      - Accumulator((W,H))
      - setMinPotential(-inf), setMaxPotential(+inf)
      - setEventContribution(1.0), setIgnorePolarity(True), Decay.NONE
      - clear() before accept()
      - n_map = getPotentialSurface()
      - N = events.size()  (size may be method or attribute)
      - ESR = sqrt(NTSS * L_N)
    """
    dv = _require_dvp()
    W, H = int(resolution[0]), int(resolution[1])

    acc = dv.Accumulator((W, H))
    # Critical: avoid saturation / clipping
    acc.setMinPotential(-np.inf)
    acc.setMaxPotential(np.inf)

    acc.setEventContribution(1.0)
    acc.setIgnorePolarity(True)
    acc.setDecayFunction(dv.Accumulator.Decay.NONE)

    if hasattr(acc, "clear"):
        acc.clear()

    acc.accept(events)

    n_map = np.asarray(acc.getPotentialSurface())

    # N: compatibility (size may be method or attribute)
    size_attr = getattr(events, "size", None)
    if size_attr is None:
        N = int(n_map.sum())
    else:
        try:
            N = int(size_attr() if callable(size_attr) else size_attr)
        except Exception:
            N = int(n_map.sum())

    return _esr_v1_from_count_surface(n_map, N=N, W=W, H=H)


def esr_list_stats(values: List[float]) -> ESRStats:
    if len(values) == 0:
        return ESRStats(mean_esr=0.0, std_esr=0.0, n_packets=0)
    arr = np.asarray(values, dtype=np.float64)
    return ESRStats(
        mean_esr=float(arr.mean()),
        std_esr=float(arr.std(ddof=0)),
        n_packets=int(arr.size),
    )


def compute_mesr_v1_from_aedat4_count_slicing(
    aedat4_path: str,
    *,
    n_per_packet: int = DEFAULT_N_PACKET,
    resolution: Tuple[int, int] = (SENSOR_W, SENSOR_H),
    drop_tail: bool = True,
    max_packets: Optional[int] = None,
    allow_out_of_order: bool = True,
    verbose_out_of_order: bool = True,
) -> Tuple[ESRStats, List[float]]:
    """
    Read aedat4 and compute MESR (mean ESR) by slicing RAW event stream into fixed-count packets.

    Implementation:
      - dv.io.MonoCameraRecording
      - dv.EventStreamSlicer.doEveryNumberOfElements(n_per_packet, callback)

    Robustness:
      - Some sequences may contain occasional out-of-order EventStores.
      - If allow_out_of_order=True, we will SKIP out-of-order batches BEFORE feeding slicer
        to prevent: IndexError: Tried adding event store to store out of order.
      - Tail (<N) is naturally dropped by slicer.

    Returns:
      - ESRStats (mean/std/n_packets over full packets)
      - list of per-packet ESR values
    """
    dv = _require_dvp()

    reader = dv.io.MonoCameraRecording(aedat4_path)
    if not reader.isEventStreamAvailable():
        raise RuntimeError(f"No event stream available in: {aedat4_path}")

    slicer = dv.EventStreamSlicer()
    esr_values: List[float] = []
    packet_counter = {"k": 0}

    class _Stop(Exception):
        pass

    def _on_packet(events_packet):
        packet_counter["k"] += 1
        esr = compute_esr_v1(events_packet, resolution=resolution)
        esr_values.append(float(esr))
        if max_packets is not None and packet_counter["k"] >= max_packets:
            raise _Stop()

    slicer.doEveryNumberOfElements(int(n_per_packet), _on_packet)

    # Out-of-order guard (batch-level)
    last_t_end: Optional[int] = None
    out_of_order_batches = 0
    skipped_events = 0

    def _get_time_range_us(store) -> Optional[Tuple[int, int]]:
        lo = getattr(store, "getLowestTime", None)
        hi = getattr(store, "getHighestTime", None)
        if callable(lo) and callable(hi):
            try:
                return int(lo()), int(hi())
            except Exception:
                return None
        # Rare fallback: try numpy (may not exist in all bindings)
        try:
            arr = store.numpy()
            names = set(arr.dtype.names or [])
            if "t" in names:
                t = arr["t"]
            elif "timestamp" in names:
                t = arr["timestamp"]
            else:
                return None
            if t.size == 0:
                return None
            return int(t.min()), int(t.max())
        except Exception:
            return None

    def _get_size(store) -> int:
        s = getattr(store, "size", None)
        if s is None:
            return 0
        try:
            return int(s() if callable(s) else s)
        except Exception:
            return 0

    try:
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None:
                continue

            if allow_out_of_order:
                tr = _get_time_range_us(batch)
                if tr is not None:
                    t0, t1 = tr
                    if last_t_end is not None and t0 < last_t_end:
                        out_of_order_batches += 1
                        skipped_events += _get_size(batch)
                        # Skip this batch to keep slicer monotonic
                        continue
                    last_t_end = t1

            slicer.accept(batch)

    except _Stop:
        pass
    except IndexError as e:
        # If allow_out_of_order=False, slicer may throw here; add context.
        raise RuntimeError(
            f"IndexError while slicing '{aedat4_path}'. "
            f"allow_out_of_order={allow_out_of_order}, "
            f"out_of_order_batches={out_of_order_batches}, skipped_events={skipped_events}. "
            f"Original: {e}"
        ) from e

    if verbose_out_of_order and out_of_order_batches > 0:
        print(
            f"[WARN] Out-of-order batches skipped: {out_of_order_batches}, "
            f"skipped_events={skipped_events} | file={aedat4_path}"
        )

    _ = drop_tail  # tail is naturally ignored by slicer; kept for API symmetry
    stats = esr_list_stats(esr_values)
    return stats, esr_values
