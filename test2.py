# # plot_avg_delta_mesr_vs_eta.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # ------------------------
# # Config
# # ------------------------
# CSV_PATH = Path("data/esr-eval/summary_esr.csv")
# ETA_LIST = [0.05, 0.10, 0.15, 0.20]
#
# COL_RAW = "mesr_raw"
# COL_ETA = {
#     0.05: "mesr_eta0.05",
#     0.10: "mesr_eta0.10",
#     0.15: "mesr_eta0.15",
#     0.20: "mesr_eta0.20",
# }
#
# # ------------------------
# # Load
# # ------------------------
# df = pd.read_csv(CSV_PATH)
#
# # ------------------------
# # Compute average delta
# # ------------------------
# avg_delta = []
# std_delta = []
#
# for eta in ETA_LIST:
#     delta = df[COL_ETA[eta]] - df[COL_RAW]
#     avg_delta.append(delta.mean())
#     std_delta.append(delta.std())
#
# # ------------------------
# # Plot
# # ------------------------
# plt.figure(figsize=(5, 4))
# plt.errorbar(
#     ETA_LIST,
#     avg_delta,
#     yerr=std_delta,
#     marker="o",
#     capsize=4,
# )
#
# plt.axhline(0.0, linestyle="--", linewidth=1)
# plt.xlabel("η")
# plt.ylabel("Average ΔMESR")
# plt.tight_layout()
# plt.show()
# import sys
# print(sys.version)
#
#
# import dv
# print(hasattr(dv, "AedatFile"))
# import dv
#
# with dv.AedatFile(r"data/emlb/day/Architecture/Architecture-ND00-1.aedat4") as f:
#     print("Streams:", f.names)
#     ev = f["events"]
#
#     print("type(ev):", type(ev))
#     print("dir(ev):")
#     for k in dir(ev):
#         if "size" in k.lower() or "time" in k.lower():
#             print(" ", k, "=>", getattr(ev, k))

import dv_processing as dv

# Open any camera
reader = dv.io.MonoCameraRecording(r"data/emlb/day/Architecture/Architecture-ND00-1.aedat4")

# Run the loop while camera is still connected
while reader.isRunning():
    # Read batch of events
    events = reader.getNextEventBatch()
    if events is not None:
        # Print received packet time range
        print(f"{events}")