# tools/run_emlb_nd00.py
from __future__ import annotations

from rcf_fast.config import CFG
from rcf_fast.io_aedat4 import quick_open_and_peek_aedat4
from rcf_fast.utils_parse import find_emlb_files


def run_step2_emlb_nd00() -> None:
    emlb_root = CFG.EMLB_DATA_ROOT
    nd = CFG.ND_FILTER.upper()

    print("=" * 100)
    print("[Step2] E-MLB aedat4 open+peek check (ND00)  [PyCharm one-click]")
    print(f"[Step2] Data root : {emlb_root}")
    print(f"[Step2] ND filter : {nd}")
    print(f"[Step2] Peek events per file : 10_000 (configurable in code)")
    print("=" * 100)

    infos = find_emlb_files(emlb_root, nd_whitelist=[nd])
    if not infos:
        print(f"[Step2][ERROR] No files found under: {emlb_root} for nd={nd}")
        return

    if CFG.LIMIT_FILES > 0:
        infos = infos[: CFG.LIMIT_FILES]
        print(f"[Step2] LIMIT_FILES enabled: {len(infos)} file(s)")

    print(f"[Step2] Total files to check: {len(infos)}")
    print("-" * 100)

    ok = 0
    for idx, info in enumerate(infos, start=1):
        try:
            stats = quick_open_and_peek_aedat4(info.path, peek_events=10_000)
            ok += 1
            W, H = stats["resolution"]
            print(f"{idx:03d}. OK  {info.path}")
            print(f"     scene={info.scene}  seq={info.seq_id}")
            print(f"     streams={stats['streams']}")
            print(f"     resolution(W,H)=({W},{H})")
            print(f"     peek_n={stats['peek_n']}, t_first={stats['t_first']}, t_last={stats['t_last']}, dt={stats['t_last']-stats['t_first']}")
            # note only print once
            if idx == 1:
                print(f"     note={stats['note']}")
            print()
        except Exception as e:
            print(f"{idx:03d}. FAIL {info.path}")
            print(f"     scene={info.scene}  seq={info.seq_id}")
            print(f"     error={repr(e)}")
            print()
            # Step2阶段：遇到失败直接停，便于定位
            break

    print("-" * 100)
    print(f"[Step2] Files checked OK: {ok}/{len(infos)}")
    if ok == len(infos):
        print("[Step2] Status: OK (aedat4 readable, event stream usable).")
        print("[Step2] Done. Proceed to Step3 streaming/bin accumulation.")
    else:
        print("[Step2] Status: NOT OK (some file failed). Fix this before Step3.")
    print("=" * 100)


if __name__ == "__main__":
    run_step2_emlb_nd00()
