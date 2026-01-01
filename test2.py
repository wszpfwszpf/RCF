import os
import re
import glob
import pandas as pd
import numpy as np


# =========================
# 配置：csv 文件夹路径
# =========================
CSV_DIR = r"data/emlb_ynoise_verify/csv"   # 改成你的 csv 文件夹路径


# =========================
# 输出：与 csv 文件夹同级
# =========================
OUT_DIR = os.path.dirname(CSV_DIR)


# =========================
# 工具：从文件名解析 split + nd
# 例：dwf_profile_day_nd00.csv
# =========================
PAT = re.compile(r"(day|night).*?(nd\d+)\.csv$", re.IGNORECASE)

def parse_split_nd(basename: str):
    m = PAT.search(basename.lower())
    if not m:
        raise ValueError(f"无法从文件名解析 day/night 和 ndxx：{basename}")
    return m.group(1).lower(), m.group(2).lower()


def format_sig(x, sig=3):
    """保留 sig 位有效数字"""
    if pd.isna(x):
        return np.nan
    return float(f"{x:.{sig}g}")


# =========================
# 主流程
# =========================
rows = []
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"目录下没有 csv 文件：{CSV_DIR}")

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    split, nd = parse_split_nd(base)

    df = pd.read_csv(csv_path)

    # 必要列检查
    required = ["scene", "file", "bin_idx", "n_events", "time_ms", "keep_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{base} 缺少列：{missing}，实际列：{list(df.columns)}")

    # bin 级保留事件数
    df["kept_events"] = (
        df["n_events"].astype(np.float64) *
        df["keep_rate"].astype(np.float64)
    )

    # 按文件合并 bin
    per_file = (
        df.groupby("file", as_index=False)
          .agg(
              total_events=("n_events", "sum"),
              total_kept=("kept_events", "sum"),
              n_bins=("bin_idx", "count")
          )
    )

    per_file["rr_file"] = (
        per_file["total_kept"] /
        per_file["total_events"].replace(0, np.nan)
    )

    # 文件级 RR：对文件均匀加权
    rr_filelevel = per_file["rr_file"].mean()

    # 事件加权 RR（仅用于对照/调试）
    rr_eventweighted = (
        per_file["total_kept"].sum() /
        max(per_file["total_events"].sum(), 1)
    )

    rows.append({
        "split": split,
        "nd": nd,
        "method": base.split("_profile_")[0] if "_profile_" in base else "unknown",
        "n_files": int(per_file.shape[0]),
        "total_events_all_files": int(per_file["total_events"].sum()),
        "rr_filelevel_mean_over_files": format_sig(rr_filelevel, sig=3),
        "rr_eventweighted_all_events": rr_eventweighted,
        "source_csv": base
    })


# =========================
# 汇总输出
# =========================
out_df = pd.DataFrame(rows)

# 排序：day 在前，nd 从小到大
nd_order = {"nd00": 0, "nd04": 1, "nd16": 2, "nd64": 3}
out_df["split_order"] = out_df["split"].map({"day": 0, "night": 1})
out_df["nd_order"] = out_df["nd"].map(nd_order).fillna(999).astype(int)

out_df = (
    out_df.sort_values(["split_order", "nd_order"])
          .drop(columns=["split_order", "nd_order"])
)

out_path = os.path.join(OUT_DIR, "filelevelsummary.csv")
out_df.to_csv(out_path, index=False, float_format="%.6f")

print("Saved:", out_path)
print(out_df.to_string(index=False))
