# plot_delta_mesr_per_sequence.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# Config
# ------------------------
CSV_PATH = Path("data/esr-eval/summary_esr.csv")
ETA_LIST = [0.05, 0.10, 0.15, 0.20]

COL_RAW = "mesr_raw"
COL_ETA = {
    0.05: "mesr_eta0.05",
    0.10: "mesr_eta0.10",
    0.15: "mesr_eta0.15",
    0.20: "mesr_eta0.20",
}

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(CSV_PATH)

# sequence name: strip suffix if needed
df["seq"] = df["file"].apply(lambda x: x.replace(".npz", ""))

# ------------------------
# Compute delta MESR
# ------------------------
for eta in ETA_LIST:
    df[f"delta_{eta:.2f}"] = df[COL_ETA[eta]] - df[COL_RAW]

# ------------------------
# Plot
# ------------------------
seqs = df["seq"].tolist()
x = np.arange(len(seqs))
width = 0.18

plt.figure(figsize=(10, 4))

for i, eta in enumerate(ETA_LIST):
    plt.bar(
        x + (i - 1.5) * width,
        df[f"delta_{eta:.2f}"],
        width=width,
        label=f"η={eta:.2f}",
    )

plt.axhline(0.0, linestyle="--", linewidth=1)

plt.xticks(x, seqs, rotation=30, ha="right")
plt.ylabel("ΔMESR (Denoised − Raw)")
plt.xlabel("Sequence")
plt.legend()
plt.tight_layout()
plt.show()
