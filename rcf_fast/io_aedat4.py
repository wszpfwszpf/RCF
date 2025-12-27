# rcf_fast/io_aedat4.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


def _require_dv():
    try:
        import dv  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import dv-processing.\n"
            "Ensure Python=3.10 and dv-processing installed."
        ) from e
    return dv


def quick_open_and_peek_aedat4(
    path: str | Path,
    peek_events: int = 10_000,
) -> Dict[str, object]:
    """
    Step2 (final intention-aligned):
    - Open aedat4 via dv-processing
    - Report available streams
    - Report sensor resolution from events iterator (size_y, size_x)
    - Peek first `peek_events` events to confirm the stream works and data looks sane
      (NO full traversal; NOT trying to get total count)

    Returns:
      {
        'streams': list[str],
        'resolution': (W, H),
        'peek_n': int,
        't_first': int,
        't_last': int,
        'note': str
      }
    """
    dv = _require_dv()
    path = str(Path(path))

    with dv.AedatFile(path) as f:
        streams = list(f.names)
        if "events" not in streams:
            raise RuntimeError(f"No 'events' stream found. Available: {streams}")

        ev = f["events"]  # _AedatFileEventIterator
        # resolution
        try:
            W = int(getattr(ev, "size_x"))
            H = int(getattr(ev, "size_y"))
        except Exception:
            # fallback: some builds expose .size=(H,W)
            sz = getattr(ev, "size", None)
            if isinstance(sz, tuple) and len(sz) == 2:
                H, W = int(sz[0]), int(sz[1])
            else:
                W, H = -1, -1

        # Peek a small number of events to confirm stream works
        t_first: Optional[int] = None
        t_last: Optional[int] = None
        n = 0

        # dv iterator yields Event objects with attributes (x, y, timestamp, polarity) typically
        for e in ev:
            # Stop at peek limit
            if n >= peek_events:
                break

            # Try to read fields robustly
            # timestamp may be e.timestamp or e.t
            ts = getattr(e, "timestamp", None)
            if ts is None:
                ts = getattr(e, "t", None)
            if ts is None:
                raise RuntimeError("Event object has no timestamp field (timestamp/t).")

            # x/y
            x = getattr(e, "x", None)
            y = getattr(e, "y", None)
            if x is None or y is None:
                raise RuntimeError("Event object has no x/y fields.")

            if t_first is None:
                t_first = int(ts)
            t_last = int(ts)
            n += 1

        if t_first is None:
            t_first, t_last = 0, 0

        return {
            "streams": streams,
            "resolution": (W, H),
            "peek_n": int(n),
            "t_first": int(t_first),
            "t_last": int(t_last),
            "note": (
                "This dv Python binding exposes an event iterator without total event count APIs. "
                "Total counts will be accumulated during Step3 streaming/bin processing."
            ),
        }
