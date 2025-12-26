# RCF scoring (TS approximation) - PyCharm one-click runnable
# Input:  data/converted-npz/*.npz with t(us,int64), x,y(int), p(int8)
# Output: data/rcf-scored/*.npz with extra score1, score2, score, block_id
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG (edit if needed)
# -------------------------
IN_DIR_REL = Path("data/converted-npz")
OUT_DIR_REL = Path("data/rcf-scored")
RECURSIVE = True
OVERWRITE = False
VERBOSE = True

# RCF global constants
SENSOR_WIDTH  = 346
SENSOR_HEIGHT = 260


# RCF fixed params (as agreed)
RADIUS = 2                 # 5x5 neighborhood
T_US = 3000                # 3ms
K_SAT = 3.0                # score1 saturation constant
BLOCK_SIZE = 16            # block size in pixels
BIN_US = 10000             # 10ms global bin
ANCHOR_RATIO = 0.10        # top/bottom 10%
N_MIN = 5                  # minimum events per block to be considered valid
EPS = 1e-12                # numerical

# time-surface init value
TS_INIT = -10**18


def project_root() -> Path:
    # tools/xxx.py -> root = parent of tools
    here = Path(__file__).resolve()
    return here.parent.parent


def list_npz_files(in_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(in_dir.glob(pattern))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: shape (2,)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_score1_ts(t: np.ndarray, x: np.ndarray, y: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Time-surface approximation:
    Maintain last timestamp per pixel. For each event, look at 5x5 neighbors,
    use dt = t_i - last_t[ny,nx] if 0<dt<=T_US, weight = 1 - dt/T_US.
    score1 = min(1, sum_w / K_SAT)
    """
    last_t = np.full((H, W), TS_INIT, dtype=np.int64)
    score1 = np.zeros_like(t, dtype=np.float32)

    r = RADIUS
    T = T_US
    invT = 1.0 / float(T)

    for i in range(len(t)):
        ti = int(t[i])
        xi = int(x[i])
        yi = int(y[i])

        # neighborhood bounds
        x0 = max(0, xi - r)
        x1 = min(W - 1, xi + r)
        y0 = max(0, yi - r)
        y1 = min(H - 1, yi + r)

        s = 0.0
        # iterate 5x5
        for yy in range(y0, y1 + 1):
            row = last_t[yy]
            for xx in range(x0, x1 + 1):
                tj = int(row[xx])
                if tj == TS_INIT:
                    continue
                dt = ti - tj
                if 0 < dt <= T:
                    s += (1.0 - dt * invT)

        # update current pixel time-surface
        last_t[yi, xi] = ti

        # saturating map to [0,1]
        sc1 = s / float(K_SAT)
        if sc1 > 1.0:
            sc1 = 1.0
        score1[i] = sc1

    return score1


def compute_block_id(x: np.ndarray, y: np.ndarray, W: int, H: int) -> tuple[np.ndarray, int, int]:
    nx = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
    ny = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    bx = (x // BLOCK_SIZE).astype(np.int32)
    by = (y // BLOCK_SIZE).astype(np.int32)
    block_id = by * nx + bx
    return block_id.astype(np.int32), nx, ny


def score_bin_score2(block_id_bin: np.ndarray, score1_bin: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Compute score2 per block for a single bin:
    - For blocks with n>=N_MIN: phi=[mu, sigma]
    - Select top/bottom ANCHOR_RATIO by mu as anchors
    - Prototype weighted by log(1+n)
    - score2 = s_sig / (s_sig + s_noise + eps), where s_* are clipped cosine>=0
    - For sparse blocks (n<N_MIN): score2=0.5
    """
    # score2_block = np.ones((n_blocks,), dtype=np.float32)
    #
    # # counts & sums
    # n = np.bincount(block_id_bin, minlength=n_blocks).astype(np.int32)
    # valid = n >= N_MIN
    # if valid.sum() < 2:
    #     # not enough valid blocks; degrade to score2=1
    #     return score2_block

    # initialize as neutral (0.5), not 1.0
    score2_block = np.full((n_blocks,), 0.5, dtype=np.float32)

    # block event counts
    n = np.bincount(block_id_bin, minlength=n_blocks).astype(np.int32)

    # block-level validity
    valid = n >= N_MIN  # N_MIN 例如 5

    # 如果整个 bin 内，连 2 个“可靠 block”都没有
    # → 全局信息不足，直接返回中性 score2=0.5
    if valid.sum() < 2:
        return score2_block

    # --------
    # 后面才是“正常情况”：可以做排序 / 原型 / 相似度
    # --------

    # 下面示意：只对 valid blocks 计算 score2，其余保持 0.5
    # score2_block[valid] = computed_score2_for_valid_blocks

    # mu
    sum_sc1 = np.bincount(block_id_bin, weights=score1_bin.astype(np.float64), minlength=n_blocks)
    mu = np.zeros((n_blocks,), dtype=np.float64)
    mu[valid] = sum_sc1[valid] / n[valid]

    # sigma: E[x^2]-mu^2
    sum_sq = np.bincount(
        block_id_bin,
        weights=(score1_bin.astype(np.float64) ** 2),
        minlength=n_blocks
    )
    var = np.zeros((n_blocks,), dtype=np.float64)
    var[valid] = sum_sq[valid] / n[valid] - (mu[valid] ** 2)
    var[var < 0] = 0.0
    sigma = np.sqrt(var)

    # phi for valid blocks
    phi = np.stack([mu, sigma], axis=1)  # (n_blocks,2)

    # sort valid blocks by mu
    valid_ids = np.nonzero(valid)[0]
    mus = mu[valid_ids]
    order = np.argsort(mus)  # ascending
    k = max(1, int(np.floor(ANCHOR_RATIO * len(valid_ids))))
    # anchors
    noise_ids = valid_ids[order[:k]]
    sig_ids = valid_ids[order[-k:]]

    # prototype weighted by log(1+n)
    w = np.log1p(n.astype(np.float64))  # (n_blocks,)
    w_sig = w[sig_ids].sum()
    w_noise = w[noise_ids].sum()
    if w_sig < EPS or w_noise < EPS:
        return score2_block

    phi_sig = (phi[sig_ids] * w[sig_ids, None]).sum(axis=0) / w_sig
    phi_noise = (phi[noise_ids] * w[noise_ids, None]).sum(axis=0) / w_noise

    # compute score2 for valid blocks
    for b in valid_ids:
        c_sig = cosine_sim(phi[b], phi_sig)
        c_noise = cosine_sim(phi[b], phi_noise)
        s_sig = c_sig if c_sig > 0 else 0.0
        s_noise = c_noise if c_noise > 0 else 0.0
        score2_block[b] = float(s_sig / (s_sig + s_noise + EPS))

    # sparse blocks keep 1 (already)
    return score2_block


def rcf_score_one_npz(npz_path: Path, out_path: Path) -> bool:
    data = np.load(npz_path)
    t = data["t"].astype(np.int64)
    x = data["x"].astype(np.int32)
    y = data["y"].astype(np.int32)
    p = data["p"].astype(np.int8) if "p" in data.files else np.zeros_like(t, dtype=np.int8)

    if len(t) == 0:
        print(f"[WARN] empty events: {npz_path.name}")
        return False

    # ensure sorted by time (should already be)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="mergesort")
        t, x, y, p = t[order], x[order], y[order], p[order]

    # infer W,H from max (assumes 0-based)
    # W = int(x.max()) + 1 SENSOR_WIDTH
    # H = int(y.max()) + 1
    W = SENSOR_WIDTH
    H = SENSOR_HEIGHT

    # compute score1 (TS approximation)
    score1 = compute_score1_ts(t, x, y, W=W, H=H)

    # block ids for all events
    block_id, nx, ny = compute_block_id(x, y, W=W, H=H)
    n_blocks = nx * ny

    # allocate outputs
    score2_event = np.ones_like(score1, dtype=np.float32)

    # process bins (bin-wise; score2 decided at bin end, then backfilled)
    # We assume t starts at 0 (from your conversion), but do not require it.
    start = 0
    N = len(t)
    while start < N:
        t0 = int(t[start])
        t_end = t0 + BIN_US

        # find end index (first index with t >= t_end)
        # use while for simplicity; can be np.searchsorted for speed
        end = np.searchsorted(t, t_end, side="left")
        if end <= start:
            end = start + 1

        # compute score2 for this bin
        bid_bin = block_id[start:end]
        sc1_bin = score1[start:end]
        score2_block = score_bin_score2(bid_bin, sc1_bin, n_blocks=n_blocks)

        # backfill to events
        score2_event[start:end] = score2_block[bid_bin]

        start = end

    score = (score1 * score2_event).astype(np.float32)

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        t=t.astype(np.int64),
        x=x.astype(np.int16),
        y=y.astype(np.int16),
        p=p.astype(np.int8),
        score1=score1.astype(np.float32),
        score2=score2_event.astype(np.float32),
        score=score.astype(np.float32),
        block_id=block_id.astype(np.int32),
    )

    if VERBOSE:
        # quick stats
        dt_med = None
        if len(t) >= 3:
            d = np.diff(t[: min(2000, len(t))])
            d = d[d > 0]
            if d.size > 0:
                dt_med = float(np.median(d))
        print(
            f"[OK] {npz_path.name} -> {out_path.name} | N={N} | "
            f"W,H=({W},{H}) | dt_med~{dt_med}us | "
            f"score1_mean={float(score1.mean()):.4f} | score2_mean={float(score2_event.mean()):.4f}"
        )
    return True


def main():
    root = project_root()
    in_dir = (root / IN_DIR_REL).resolve()
    out_dir = (root / OUT_DIR_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        print(f"[INFO] Project root: {root}")
        print(f"[INFO] Input dir   : {in_dir}")
        print(f"[INFO] Output dir  : {out_dir}")
        print(f"[INFO] Params: T={T_US}us, r={RADIUS}, K={K_SAT}, block={BLOCK_SIZE}, bin={BIN_US}us, p={ANCHOR_RATIO}, n_min={N_MIN}")

    files = list_npz_files(in_dir, recursive=RECURSIVE)
    if not files:
        print(f"[ERROR] No .npz files found in: {in_dir}")
        return

    ok, fail, skip = 0, 0, 0
    for f in files:
        out_path = out_dir / f.name
        if out_path.exists() and not OVERWRITE:
            skip += 1
            if VERBOSE:
                print(f"[SKIP] {f.name} (exists)")
            continue
        try:
            if rcf_score_one_npz(f, out_path):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {f.name}: {repr(e)}")

    print(f"\n[SUMMARY] ok={ok}, skip={skip}, fail={fail}, total={len(files)}")


if __name__ == "__main__":
    main()

# Run in PyCharm: just click Run on this file.

