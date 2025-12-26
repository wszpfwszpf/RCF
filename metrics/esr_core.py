# metrics/esr_core.py
# -*- coding: utf-8 -*-
"""
ESR core computation (pure functions).

Design goals:
- Pure: no file I/O, no denoising, no masking logic from RCF.
- Input: events (x, y) or counts per pixel.
- Output: ESR score (float).

ESR definition follows the E-MLB protocol:
  ESR = sqrt(NTSS * L_N)
where:
  NTSS = sum_i n_i (n_i - 1) / [N (N - 1)]
  L_N  = K - sum_i (1 - M/N)^{n_i}
K = number of pixels (W*H), N = total events in packet, M = fixed reference count.

Notes:
- Requires N >= 2, else ESR is undefined; we return 0.0 for safety.
- For stability, computations use float64 internally.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _validate_resolution(resolution: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(resolution, tuple) and len(resolution) == 2):
        raise ValueError(f"resolution must be a tuple (W, H), got {resolution}")
    W, H = int(resolution[0]), int(resolution[1])
    if W <= 0 or H <= 0:
        raise ValueError(f"Invalid resolution: W={W}, H={H}")
    return W, H


def xy_to_counts(
    x: np.ndarray,
    y: np.ndarray,
    resolution: Tuple[int, int],
    hot_pixel_valid_mask: Optional[np.ndarray] = None,
    validate_xy: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Convert event coordinates (x, y) to per-pixel counts (flattened).

    Returns:
        counts_flat: shape (K,), int32
        N: total number of valid events used
        K: number of pixels (W*H)
    """
    W, H = _validate_resolution(resolution)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    if x.size == 0:
        K = W * H
        return np.zeros((K,), dtype=np.int32), 0, K

    # Validate / filter coordinates if requested
    if validate_xy:
        x_i = x.astype(np.int64, copy=False)
        y_i = y.astype(np.int64, copy=False)
        valid = (x_i >= 0) & (x_i < W) & (y_i >= 0) & (y_i < H)
        if hot_pixel_valid_mask is not None:
            # hot_pixel_valid_mask: (H, W) bool, True=keep
            m = np.asarray(hot_pixel_valid_mask, dtype=bool)
            if m.shape != (H, W):
                raise ValueError(f"hot_pixel_valid_mask must have shape (H, W)={(H, W)}, got {m.shape}")
            valid &= m[y_i, x_i]
        x_i = x_i[valid]
        y_i = y_i[valid]
    else:
        x_i = x.astype(np.int64, copy=False)
        y_i = y.astype(np.int64, copy=False)

    N = int(x_i.size)
    K = W * H
    if N == 0:
        return np.zeros((K,), dtype=np.int32), 0, K

    # Flatten index: idx = y*W + x
    idx = y_i * W + x_i
    counts_flat = np.bincount(idx, minlength=K).astype(np.int32, copy=False)
    return counts_flat, N, K


def compute_ntss(counts_flat: np.ndarray, N: int) -> float:
    """
    NTSS = sum_i n_i (n_i - 1) / [N (N - 1)]
    """
    if N < 2:
        return 0.0
    c = np.asarray(counts_flat, dtype=np.int64)
    num = np.sum(c * (c - 1), dtype=np.int64)
    den = N * (N - 1)
    return float(num) / float(den)


def compute_ln(counts_flat: np.ndarray, N: int, M: int, K: int) -> float:
    """
    L_N = K - sum_i (1 - M/N)^{n_i}

    where:
      - K = number of pixels
      - N = total events in current packet
      - M = fixed reference count (config)
    """
    if N <= 0 or M <= 0 or K <= 0:
        return 0.0
    if N < M:
        # In E-MLB usage, typically N > M.
        # If N < M happens (e.g., aggressive filtering), clamp ratio to avoid negative base.
        # This keeps L_N defined but will reduce interpretability; caller should avoid this in protocol.
        ratio = 1.0
    else:
        ratio = float(M) / float(N)

    base = 1.0 - ratio  # in [0, 1]
    # Use float64 for numerical stability
    c = np.asarray(counts_flat, dtype=np.float64)
    # base**c; when base=0, only pixels with c=0 contribute 1, others 0
    term = np.power(base, c, dtype=np.float64)
    s = float(np.sum(term, dtype=np.float64))
    return float(K) - s


def compute_esr_from_counts(counts_flat: np.ndarray, N: int, M: int, K: int) -> float:
    """
    ESR = sqrt(NTSS * L_N)
    """
    if N < 2:
        return 0.0
    ntss = compute_ntss(counts_flat, N)
    ln = compute_ln(counts_flat, N, M, K)
    val = ntss * ln
    if val <= 0.0:
        return 0.0
    return float(np.sqrt(val))


def compute_esr(
    events: Dict[str, np.ndarray],
    resolution: Tuple[int, int],
    M: int,
    hot_pixel_valid_mask: Optional[np.ndarray] = None,
    validate_xy: bool = True,
) -> float:
    """
    Compute ESR from an event dict containing at least:
      - events["x"], events["y"]

    Args:
        events: dict with numpy arrays.
        resolution: (W, H)
        M: reference event count for LN interpolation (fixed in config).
        hot_pixel_valid_mask: optional (H, W) bool mask, True=keep pixel.
        validate_xy: if True, drop out-of-range events.
    """
    if "x" not in events or "y" not in events:
        raise KeyError('events must contain keys "x" and "y"')

    counts_flat, N, K = xy_to_counts(
        events["x"],
        events["y"],
        resolution=resolution,
        hot_pixel_valid_mask=hot_pixel_valid_mask,
        validate_xy=validate_xy,
    )
    return compute_esr_from_counts(counts_flat, N=N, M=int(M), K=K)
