# metrics/mesr_eval_npz.py
# -*- coding: utf-8 -*-
"""
MESR evaluation pipeline for NPZ-based event streams.

- Packet by fixed event count N (from ESRConfig by default)
- MESR aggregation (mean/std over packet ESRs)
- No denoising logic inside (mask/threshold happens outside)

This file *uses* ESRConfig to keep protocol centralized and reproducible.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .esr_config import ESRConfig, DEFAULT_ESR_CONFIG
from .esr_core import compute_esr


def _slice_events(events: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
    """Slice all per-event fields (same length as events['x']) by [start:end]."""
    if "x" not in events:
        raise KeyError('events must contain key "x"')

    n = events["x"].shape[0]
    out: Dict[str, np.ndarray] = {}
    for k, v in events.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[start:end]
        else:
            # keep metadata / non-event arrays untouched
            out[k] = v
    return out


def iter_packets_by_N(
    events: Dict[str, np.ndarray],
    N: Optional[int] = None,
    drop_last: Optional[bool] = None,
    cfg: ESRConfig = DEFAULT_ESR_CONFIG,
) -> Iterator[Tuple[int, int, int, Dict[str, np.ndarray]]]:
    """
    Iterate event packets by fixed event count N.

    Yields:
        (packet_index, start, end, packet_events)

    Notes:
        - start/end indices refer to the original events array.
        - drop_last controls whether to discard the tail packet with <N events.
    """
    if "x" not in events or "y" not in events:
        raise KeyError('events must contain keys "x" and "y"')

    N = int(cfg.n_events_packet if N is None else N)
    drop_last = bool(cfg.drop_last if drop_last is None else drop_last)

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    n_total = int(events["x"].shape[0])
    if n_total == 0:
        return

    n_full = n_total // N
    n_packets = n_full if drop_last else (n_full + (1 if (n_total % N) else 0))

    for i in range(n_packets):
        start = i * N
        end = min((i + 1) * N, n_total)
        if drop_last and (end - start) < N:
            break
        yield i, start, end, _slice_events(events, start, end)


def compute_mesr(
    events: Dict[str, np.ndarray],
    cfg: ESRConfig = DEFAULT_ESR_CONFIG,
    *,
    # optional overrides
    resolution: Optional[Tuple[int, int]] = None,
    N: Optional[int] = None,
    M: Optional[int] = None,
    drop_last: Optional[bool] = None,
    return_lists: bool = True,
) -> Dict[str, object]:
    """
    Compute MESR over a full event stream using protocol from cfg (unless overridden).

    Returns:
        {
          "mesr_mean": float,
          "mesr_std": float,
          "n_packets": int,
          "esr_list": [float] (optional),
        }
    """
    resolution = cfg.resolution if resolution is None else resolution
    N = cfg.n_events_packet if N is None else int(N)
    M = cfg.m_events_ref if M is None else int(M)
    drop_last = cfg.drop_last if drop_last is None else bool(drop_last)

    esr_list: List[float] = []

    for _, _, _, pkt in iter_packets_by_N(events, N=N, drop_last=drop_last, cfg=cfg):
        esr = compute_esr(
            pkt,
            resolution=resolution,
            M=M,
            hot_pixel_valid_mask=None if cfg.hot_pixel_valid_mask is None else np.asarray(cfg.hot_pixel_valid_mask),
            validate_xy=cfg.validate_xy,
        )
        esr_list.append(float(esr))

    n_packets = len(esr_list)
    if n_packets == 0:
        out: Dict[str, object] = {
            "mesr_mean": float("nan"),
            "mesr_std": float("nan"),
            "n_packets": 0,
        }
        if return_lists:
            out["esr_list"] = []
        return out

    arr = np.asarray(esr_list, dtype=np.float64)
    out = {
        "mesr_mean": float(np.mean(arr)),
        "mesr_std": float(np.std(arr)),
        "n_packets": int(n_packets),
    }
    if return_lists:
        out["esr_list"] = esr_list
    return out


def compute_mesr_pair(
    events_raw: Dict[str, np.ndarray],
    events_dn: Dict[str, np.ndarray],
    cfg: ESRConfig = DEFAULT_ESR_CONFIG,
    *,
    resolution: Optional[Tuple[int, int]] = None,
    N: Optional[int] = None,
    M: Optional[int] = None,
    drop_last: Optional[bool] = None,
    return_lists: bool = False,
) -> Dict[str, object]:
    """
    Convenience wrapper when you already have two full streams.

    Note:
        For strict fairness in your setting, it's better to compute ESR packetwise on RAW packets
        and apply mask within each packet, rather than slicing dn independently.
        This wrapper is still useful for quick comparisons or when dn is aligned by construction.
    """
    raw_res = compute_mesr(
        events_raw, cfg,
        resolution=resolution, N=N, M=M, drop_last=drop_last,
        return_lists=return_lists,
    )
    dn_res = compute_mesr(
        events_dn, cfg,
        resolution=resolution, N=N, M=M, drop_last=drop_last,
        return_lists=return_lists,
    )

    raw_mean = float(raw_res["mesr_mean"])
    dn_mean = float(dn_res["mesr_mean"])
    delta = dn_mean - raw_mean if (not np.isnan(raw_mean) and not np.isnan(dn_mean)) else float("nan")

    out: Dict[str, object] = {
        "mesr_raw_mean": raw_mean,
        "mesr_raw_std": float(raw_res["mesr_std"]),
        "mesr_dn_mean": dn_mean,
        "mesr_dn_std": float(dn_res["mesr_std"]),
        "delta_mean": float(delta),
        "n_packets_raw": int(raw_res["n_packets"]),
        "n_packets_dn": int(dn_res["n_packets"]),
    }
    if return_lists:
        out["esr_list_raw"] = raw_res.get("esr_list", [])
        out["esr_list_dn"] = dn_res.get("esr_list", [])
    return out
