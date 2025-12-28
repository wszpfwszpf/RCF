import numpy as np
from pathlib import Path
import dv_processing as dv


# -----------------------------
# 路径解析：相对路径按脚本目录
# -----------------------------
def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    base = Path(__file__).resolve().parent
    return (base / p).resolve()


# -----------------------------
# 读取 keepmask（强制取 key='keepmask'）
# -----------------------------
def load_keepmask(npz_path: str | Path, key: str = "keepmask"):
    npz_path = resolve_path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"KM npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found. Available keys: {list(data.keys())}")

    km = data[key]
    km = np.asarray(km)

    # 常见：按 bin 存成 object array，每个元素是一个 (Ni,) 或 (Ni,6) 的数组
    if km.ndim == 1 and km.dtype == object:
        parts = [np.asarray(x) for x in km]
        km = np.concatenate(parts, axis=0)

    return km


# -----------------------------
# 读取 aedat4 并收集事件（分批读，尽量不占内存）
# 这里我们只拿 t（timestamp）和 count
# -----------------------------
def iter_event_batches(aedat4_path: str | Path):
    aedat4_path = resolve_path(aedat4_path)
    if not aedat4_path.exists():
        raise FileNotFoundError(f"AEDAT4 not found: {aedat4_path}")

    reader = dv.io.MonoCameraRecording(str(aedat4_path))
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None:
            continue
        yield events


# -----------------------------
# 方式A：按固定事件数划 bin
# -----------------------------
def count_events_fullbins_by_count(aedat4_path: str | Path, bin_event_count: int):
    """
    返回：N_fullbins_events, N_total_events, N_fullbins
    丢弃尾部不足 bin_event_count 的事件。
    """
    total = 0
    for events in iter_event_batches(aedat4_path):
        total += events.size()

    n_fullbins = total // bin_event_count
    n_full = n_fullbins * bin_event_count
    return n_full, total, n_fullbins


# -----------------------------
# 方式B：按固定时间窗划 bin
# -----------------------------
def count_events_fullbins_by_time(aedat4_path: str | Path, bin_us: int):
    """
    bin_us：每个 bin 的时间长度（微秒）
    返回：N_fullbins_events, N_total_events, N_fullbins, last_fullbin_end_ts
    丢弃最后一个不完整 bin（末尾不足 bin_us 的时间段内的事件）。
    注意：这里默认以“第一个事件时间”为起点对齐 bin（常见做法）。
    """
    # 第一遍：拿到第一条事件时间 & 最后一条事件时间 & 总事件数
    t0 = None
    t_last = None
    total = 0

    for events in iter_event_batches(aedat4_path):
        if events.size() == 0:
            continue
        if t0 is None:
            t0 = int(events.getLowestTime())
        t_last = int(events.getHighestTime())
        total += events.size()

    if t0 is None:
        return 0, 0, 0, None

    # 计算完整 bins 的结束时间（不包含尾部不满一个 bin 的时间段）
    duration = t_last - t0  # us
    n_fullbins = duration // bin_us
    last_full_end = t0 + n_fullbins * bin_us  # 这个时间戳之前的都属于完整 bin

    # 第二遍：统计 t < last_full_end 的事件数
    n_full = 0
    for events in iter_event_batches(aedat4_path):
        if events.size() == 0:
            continue
        # dv_processing 的 EventStore 可以按时间范围切片
        sliced = events.sliceTime(t0, last_full_end)  # [t0, last_full_end)
        n_full += sliced.size()

    return n_full, total, n_fullbins, last_full_end


# -----------------------------
# 总检验：N_fullbins_events 对齐 keepmask
# -----------------------------
def check_align(aedat4_path, km_path, mode="count", bin_event_count=2000, bin_us=1000):
    """
    mode:
      - "count": 按固定事件数划 bin
      - "time" : 按固定时间窗划 bin
    """
    km = load_keepmask(km_path)
    km = np.asarray(km)

    # keepmask 可能是 (N,) 或 (N,6)
    if km.ndim == 1:
        n_km = km.shape[0]
    elif km.ndim == 2:
        n_km = km.shape[0]
    else:
        raise AssertionError(f"Unsupported keepmask ndim={km.ndim}, shape={km.shape}")

    if mode == "count":
        n_full, n_total, n_bins = count_events_fullbins_by_count(aedat4_path, bin_event_count)
        print(f"[MODE=count] bin_event_count={bin_event_count}")
        print(f"[AEDAT4] total_events={n_total}, fullbins={n_bins}, fullbins_events={n_full}")
    elif mode == "time":
        n_full, n_total, n_bins, last_end = count_events_fullbins_by_time(aedat4_path, bin_us)
        print(f"[MODE=time] bin_us={bin_us}")
        print(f"[AEDAT4] total_events={n_total}, fullbins={n_bins}, fullbins_events={n_full}, last_full_end={last_end}")
    else:
        raise ValueError("mode must be 'count' or 'time'")

    print(f"[KM] rows_or_len={n_km}, shape={km.shape}, dtype={km.dtype}")

    if n_full != n_km:
        raise AssertionError(f"Mismatch: fullbins_events={n_full} vs keepmask_events={n_km}")
    print("[PASS] fullbins_events == keepmask_events")


if __name__ == "__main__":
    AEDAT4_PATH = r"data/emlb/day/Architecture/Architecture-ND00-1.aedat4"
    KM_PATH     = r"data/emlb_rcf_verify/keepmask/Architecture-ND00-1_nd00_keepmask.npz"

    # 你只需要选一种 mode，并把 bin 参数改成你 RCF 生成 km 时用的那个
    # 方案1：按事件数
    # check_align(AEDAT4_PATH, KM_PATH, mode="count", bin_event_count=2000)

    # 方案2：按时间窗（微秒）
    check_align(AEDAT4_PATH, KM_PATH, mode="time", bin_us=10000)
