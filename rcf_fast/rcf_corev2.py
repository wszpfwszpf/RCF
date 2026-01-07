# rcf_fast/rcf_core.py
# ============================================================
# 日期：2026-01-07
# 作者：ChatGPT（按用户要求：回退到“原始版本 + 线性衰减”，补中文注释）
#
# 功能概述：
#   RCF 在每个时间 bin 内对事件逐个打分，并输出不同阈值 η 对应的 keepmask。
#   核心分两步：
#     1) score1：局部时空一致性（Time-Surface + 线性衰减核）
#     2) score2：跨块相对统计对比（基于每个 block 的 (均值ub, 方差sigmab) 与原型相似度）
#   最终融合：
#     score = score1 * score2
#     keep = (score >= η)
#
# 重要说明：
#   - “3ms 改 10ms” 属于 cfg.T_us 的修改，core 不需要改动；你已验证线性核更优。
#   - 本文件保持 v1 的 import 与函数签名不变，以确保 step4 调用无需修改。
# ============================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rcf_fast.rcf_state import RCFState
from rcf_fast.rcf_compute_config import RCFComputeConfig


# -------------------------------------------------
# 输出结果结构体：便于上层 step4 读取/保存
# -------------------------------------------------
@dataclass
class RCFResult:
    score1: np.ndarray                 # 每个事件的局部一致性分数
    score2: np.ndarray                 # 每个事件所在 block 的相对一致性分数
    score: np.ndarray                  # 融合后的最终分数 score1*score2
    keep_masks: Dict[float, np.ndarray]# 不同 η 对应的布尔保留掩码
    block_id: np.ndarray               # 每个事件所属 block 的 id
    ub: np.ndarray                     # 每个 block 的 score1 均值
    sigmab: np.ndarray                 # 每个 block 的 score1 标准差


# -------------------------------------------------
# 工具函数：从 dv EventStore 中抽取 t/x/y/p
# 说明：RCF 不依赖极性 p，但为了兼容数据结构仍读出来
# -------------------------------------------------
def _extract_txyp(events) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = events.numpy()
    names = arr.dtype.names

    # x, y：坐标
    x = arr["x"].astype(np.int32, copy=False)
    y = arr["y"].astype(np.int32, copy=False)

    # t：时间戳（微秒或同一单位的整数时间）
    if "t" in names:
        t = arr["t"].astype(np.int64, copy=False)
    elif "timestamp" in names:
        t = arr["timestamp"].astype(np.int64, copy=False)
    else:
        raise KeyError(f"EventStore has no time field. Available fields: {names}")

    # p：极性（可选）
    if "p" in names:
        p = arr["p"].astype(np.int8, copy=False)
    elif "polarity" in names:
        p = arr["polarity"].astype(np.int8, copy=False)
    else:
        p = np.zeros_like(x, dtype=np.int8)

    return t, x, y, p


def _safe_event_count(events) -> int:
    """兼容不同 EventStore 实现的 size() / size 属性。"""
    sz = getattr(events, "size", None)
    try:
        return int(sz() if callable(sz) else sz)
    except Exception:
        return 0


# -------------------------------------------------
# score1：Time-Surface + 线性衰减核
#
# 思路：
#   对每个事件 e_i=(t_i,x_i,y_i)，查看其空间邻域内每个像素最近一次事件时间 last_ts，
#   计算 dt = t_i - last_ts
#   若 0<dt<=T，则该邻域像素提供支持 w(dt)=1-dt/T
#   将邻域内所有支持相加，除以 K_sat 做归一化并截断到 [0,1]
#
# 注意：
#   - ts_map 存的是“每个像素最近一次事件的时间戳”
#   - 每处理一个事件，就更新 ts_map[y_i,x_i]=t_i（在线、因果、流式）
# -------------------------------------------------
try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap


@njit(cache=True, fastmath=True)
def _score1_ts_numba(t, x, y, ts_map, radius, T_us, K_sat):
    """
    Numba 加速版 score1 计算：
      - 输入 t/x/y 为事件数组（长度 N）
      - ts_map 为 (H,W) 的最近时间戳表
    """
    H, W = ts_map.shape
    out = np.zeros(t.shape[0], dtype=np.float32)

    T = float(T_us)
    K = float(K_sat)

    for i in range(t.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        ti = float(t[i])

        s = 0.0

        # 遍历 (2r+1)x(2r+1) 空间邻域
        for dy in range(-radius, radius + 1):
            yy = yi + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-radius, radius + 1):
                xx = xi + dx
                if xx < 0 or xx >= W:
                    continue

                last = ts_map[yy, xx]
                dt = ti - last

                # 只统计窗口内的支持
                if dt > 0.0 and dt <= T:
                    # 线性衰减核：越近权重越大，越远权重越小
                    s += (1.0 - dt / T)

        # 归一化 + 截断
        s = s / K
        if s > 1.0:
            s = 1.0
        out[i] = np.float32(s)

        # 更新 time-surface：该像素“最近事件时间”
        ts_map[yi, xi] = ti

    return out


def compute_score1_ts(t, x, y, state: RCFState, cfg: RCFComputeConfig) -> np.ndarray:
    """
    Python 包装：优先走 Numba；若环境没有 numba，则走 python loop（慢但正确）
    """
    ts_map = state.ts.last_ts  # shape (H,W)，float64

    if _NUMBA_OK:
        return _score1_ts_numba(t, x, y, ts_map, cfg.radius, cfg.T_us, cfg.K_sat)

    # fallback：纯 python（性能较差，主要用于 debug）
    H, W = ts_map.shape
    r = int(cfg.radius)
    T = float(cfg.T_us)
    Ksat = float(cfg.K_sat)

    score1 = np.zeros_like(t, dtype=np.float32)

    for i in range(len(t)):
        xi, yi, ti = int(x[i]), int(y[i]), float(t[i])

        s = 0.0
        for dy in range(-r, r + 1):
            yy = yi + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-r, r + 1):
                xx = xi + dx
                if xx < 0 or xx >= W:
                    continue
                dt = ti - ts_map[yy, xx]
                if 0.0 < dt <= T:
                    s += (1.0 - dt / T)

        s = s / Ksat
        if s > 1.0:
            s = 1.0

        score1[i] = np.float32(s)
        ts_map[yi, xi] = ti

    return score1


# -------------------------------------------------
# 统计每个 block 的 score1 分布：
#   - counts[b]：block b 内事件数
#   - ub[b]：score1 均值
#   - sigmab[b]：score1 标准差
#
# 这些统计量用于 score2 的“相对一致性对比”
# -------------------------------------------------
def compute_block_stats(score1: np.ndarray, block_id: np.ndarray, n_blocks: int):
    counts = np.bincount(block_id, minlength=n_blocks).astype(np.int32)

    sum1 = np.bincount(block_id, weights=score1, minlength=n_blocks)
    ub = np.zeros(n_blocks, dtype=np.float64)
    valid = counts > 0
    ub[valid] = sum1[valid] / counts[valid]

    sum2 = np.bincount(block_id, weights=score1 ** 2, minlength=n_blocks)
    sigmab = np.zeros(n_blocks, dtype=np.float64)
    sigmab[valid] = np.sqrt(
        np.maximum(sum2[valid] / counts[valid] - ub[valid] ** 2, 0.0)
    )

    return counts, ub, sigmab


def cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """余弦相似度（带 eps 防止 0 向量除零）。"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


# -------------------------------------------------
# score2：跨块相对统计对比
#
# 做法（概述）：
#   1) 只在“事件数足够”的 block 上计算 score2（稀疏 block 给固定值）
#   2) 用 ub 对 block 排序，取 top-k 作为“信号原型”，bottom-k 作为“噪声原型”
#   3) 每个 block 的特征 phi_b = [ub, sigmab]
#   4) 计算 phi_b 与两类原型的余弦相似度 c_sig, c_noise
#   5) score2_b = c_sig / (c_sig + c_noise + eps)
#   6) 映射回 event-wise：score2[event] = score2_block[block_id[event]]
# -------------------------------------------------
def compute_score2(
    score1: np.ndarray,
    block_id: np.ndarray,
    counts: np.ndarray,
    ub: np.ndarray,
    sigmab: np.ndarray,
    cfg: RCFComputeConfig,
) -> np.ndarray:

    n_blocks = ub.shape[0]
    score2_block = np.full(n_blocks, cfg.score2_sparse_value, dtype=np.float64)

    # 只对事件数足够的 block 计算 score2
    valid = counts >= cfg.min_events_per_block
    idx = np.where(valid)[0]
    if idx.size < 2:
        # 可计算 block 太少，直接返回默认值
        return score2_block[block_id].astype(np.float32)

    # 按 ub 排序，构造上下锚点集合
    order = idx[np.argsort(ub[idx])]
    k = max(1, int(cfg.anchor_ratio * order.size))
    bot = order[:k]
    top = order[-k:]

    # block 特征：均值+方差（或标准差）
    phi = np.stack([ub, sigmab], axis=1)

    # 用 log(1+count) 作为权重，弱化超大块的支配性
    w = np.log1p(counts)

    # 信号原型 / 噪声原型
    phi_sig = np.average(phi[top], axis=0, weights=w[top])
    phi_noise = np.average(phi[bot], axis=0, weights=w[bot])

    # 对每个有效 block 计算相对一致性
    for b in idx:
        c_sig = max(cosine(phi[b], phi_sig, cfg.eps), 0.0)
        c_noise = max(cosine(phi[b], phi_noise, cfg.eps), 0.0)
        score2_block[b] = c_sig / (c_sig + c_noise + cfg.eps)

    return score2_block[block_id].astype(np.float32)


# -------------------------------------------------
# 主入口：处理一个 bin 内的事件，输出 score1/score2/keepmasks
# -------------------------------------------------
def rcf_process_bin(
    events,
    state: RCFState,
    cfg: RCFComputeConfig,
) -> Optional[RCFResult]:

    N = _safe_event_count(events)
    if N <= 0:
        return None

    t, x, y, p = _extract_txyp(events)

    # 1) score1：局部时空一致性
    score1 = compute_score1_ts(t, x, y, state, cfg)

    # 2) block id：将像素坐标映射到 block 网格
    W, H = cfg.resolution
    B = cfg.block_size
    nbx = (W + B - 1) // B
    bx = x // B
    by = y // B
    block_id = bx + by * nbx
    n_blocks = nbx * ((H + B - 1) // B)

    # 3) block 统计量
    counts, ub, sigmab = compute_block_stats(score1, block_id, n_blocks)

    # 4) score2：跨块相对一致性
    score2 = compute_score2(score1, block_id, counts, ub, sigmab, cfg)

    # 5) 融合：结构优先（局部一致性 * 相对一致性）
    score = (score1 * score2).astype(np.float32)

    # 6) 不同阈值 η 生成 keepmask（上层用于 ESR 统计与可视化）
    keep_masks = {eta: (score >= eta) for eta in cfg.eta_list}

    return RCFResult(
        score1=score1,
        score2=score2,
        score=score,
        keep_masks=keep_masks,
        block_id=block_id,
        ub=ub,
        sigmab=sigmab,
    )
