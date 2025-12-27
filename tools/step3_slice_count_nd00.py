# tools/step3_slice_count_nd00.py
from __future__ import annotations

from pathlib import Path

from rcf_fast.packet_slice import run_count_slicer


# -------------------------
# PyCharm one-click config
# -------------------------
EMLB_DAY_ROOT = r"C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\emlb\day"
ND_FILTER = "ND00"

N_PER_PACKET = 30000
MAX_PACKETS = 10


def _find_first_nd_file(root: str, nd: str) -> Path:
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"EMLB root not found: {rootp}")

    files = sorted(rootp.rglob("*.aedat4"))
    if not files:
        raise FileNotFoundError(f"No .aedat4 files found under: {rootp}")

    nd_upper = nd.upper()
    cand = [p for p in files if nd_upper in p.name.upper()]
    if not cand:
        raise FileNotFoundError(f"No files matching '{nd_upper}' under: {rootp}")
    return cand[0]


def main() -> None:
    path = _find_first_nd_file(EMLB_DAY_ROOT, ND_FILTER)

    print("=" * 100)
    print("[Step3-Count] dv-processing read + fixed-count slicing (raw packets)")
    print(f"[Step3-Count] File        : {path}")
    print(f"[Step3-Count] ND          : {ND_FILTER}")
    print(f"[Step3-Count] N/packet    : {N_PER_PACKET}")
    print(f"[Step3-Count] max_packets : {MAX_PACKETS}")
    print("=" * 100)

    def on_packet(events_packet, info):
        print(
            f"[pkt {info.packet_id:04d}] "
            f"raw_idx=[{info.raw_begin:8d},{info.raw_end:8d})  "
            f"n={info.n_events:6d}  dt_us={info.dt_us:8d}  "
            f"t=[{info.t_first}; {info.t_last}]"
        )

    run_count_slicer(
        aedat4_path=path,
        n_per_packet=N_PER_PACKET,
        on_packet=on_packet,
        max_packets=MAX_PACKETS,
    )

    print("=" * 100)
    print("[Step3-Count] Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
