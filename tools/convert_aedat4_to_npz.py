# aedat4 -> npz (t,x,y,p), Windows/PyCharm friendly, no CLI.
# Requirements: dv_processing installed.
import dv_processing as dv
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG (edit if needed)
# -------------------------
IN_DIR_REL = Path("data/origin-aedat4")
OUT_DIR_REL = Path("data/converted-npz")
RECURSIVE = True
NORMALIZE_T = True          # t -= t[0]
OVERWRITE = False
VERBOSE = True

# Output polarity format:
# - If True: keep {-1,+1} (when possible)
# - If False: map to {0,1}
KEEP_POLARITY_SIGN = False


def project_root() -> Path:
    """
    Robustly locate project root so that 'data/...' works even if
    PyCharm working directory is not the project root.
    Heuristic: assume this file is under <root>/tools/.
    """
    here = Path(__file__).resolve()
    # tools/convert_aedat4_to_npz.py -> root = parent of tools
    root = here.parent.parent
    return root


def list_aedat4_files(in_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.aedat4" if recursive else "*.aedat4"
    return sorted(in_dir.glob(pattern))


def load_events_from_aedat4(path: Path) -> np.ndarray | None:
    """
    Read events from an .aedat4 file using dv_processing.io.MonoCameraRecording.
    Return ndarray of shape (N,4): [t_us(int64), x(int16), y(int16), p(int8)].
    NOTE: time unit is assumed to be microseconds as returned by dv_processing.
    """
    reader = dv.io.MonoCameraRecording(str(path))
    chunks = []

    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue

        # Prefer structured numpy
        try:
            arr = batch.numpy()  # structured array, e.g., ('timestamp','x','y','polarity')
            names = arr.dtype.names
            if names is None:
                raise AttributeError

            try:
                t = arr["timestamp"].astype(np.int64)
                x = arr["x"].astype(np.int16)
                y = arr["y"].astype(np.int16)
                p = arr["polarity"].astype(np.int8)
            except Exception as e:
                print(f"[ERROR] {path.name}: numpy dtype={arr.dtype}, fields mismatch: {e}")
                return None

        except AttributeError:
            # fallback getters
            t = batch.getTimestamps().astype(np.int64)
            x = batch.getX().astype(np.int16)
            y = batch.getY().astype(np.int16)
            p = batch.getPolarities().astype(np.int8)

        if t.size == 0:
            continue

        chunk = np.column_stack([t, x, y, p])
        chunks.append(chunk)

    if not chunks:
        print(f"[WARN] {path.name}: empty events.")
        return None

    events = np.concatenate(chunks, axis=0)

    # sort by time for safety
    order = np.argsort(events[:, 0], kind="mergesort")
    events = events[order]

    # normalize time to start at 0 (optional)
    if NORMALIZE_T:
        events[:, 0] = events[:, 0] - events[0, 0]

    # polarity normalization (optional)
    # dv_processing often yields polarity as 0/1, but keep this robust:
    p = events[:, 3].astype(np.int32)
    uniq = set(np.unique(p).tolist())
    if KEEP_POLARITY_SIGN:
        # convert {0,1} -> {-1,+1} if needed
        if uniq == {0, 1}:
            events[:, 3] = (p * 2 - 1).astype(np.int8)
        else:
            events[:, 3] = p.astype(np.int8)
    else:
        # convert {-1,+1} -> {0,1} if needed
        if uniq == {-1, 1}:
            events[:, 3] = ((p + 1) // 2).astype(np.int8)
        else:
            # already 0/1 or other two-valued set; map any nonzero to 1
            events[:, 3] = (p != 0).astype(np.int8)

    return events


def save_as_npz(events: np.ndarray, out_path: Path) -> None:
    """
    Save events to .npz with arrays t,x,y,p (txyp).
    """
    t = events[:, 0].astype(np.int64)
    x = events[:, 1].astype(np.int16)
    y = events[:, 2].astype(np.int16)
    p = events[:, 3].astype(np.int8)
    np.savez_compressed(str(out_path), t=t, x=x, y=y, p=p)


def main():
    root = project_root()
    in_dir = (root / IN_DIR_REL).resolve()
    out_dir = (root / OUT_DIR_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        print(f"[INFO] Project root: {root}")
        print(f"[INFO] Input dir   : {in_dir}")
        print(f"[INFO] Output dir  : {out_dir}")

    files = list_aedat4_files(in_dir, recursive=RECURSIVE)
    if not files:
        print(f"[ERROR] No .aedat4 files found in: {in_dir}")
        return

    ok, fail, skip = 0, 0, 0
    for f in files:
        out_path = out_dir / (f.stem + ".npz")
        if out_path.exists() and not OVERWRITE:
            skip += 1
            if VERBOSE:
                print(f"[SKIP] {f.name} -> {out_path.name} (exists)")
            continue

        try:
            events = load_events_from_aedat4(f)
            if events is None:
                fail += 1
                continue

            save_as_npz(events, out_path)
            ok += 1

            if VERBOSE:
                t = events[:, 0]
                x = events[:, 1]
                y = events[:, 2]
                p = events[:, 3]
                dt_med = None
                if len(t) >= 3:
                    d = np.diff(t[: min(2000, len(t))])
                    d = d[d > 0]
                    if d.size > 0:
                        dt_med = float(np.median(d))
                print(
                    f"[OK] {f.name} -> {out_path.name} | N={len(events)} | "
                    f"t=[{int(t[0])},{int(t[-1])}]us | "
                    f"dt_med~{dt_med}us | "
                    f"x=[{int(x.min())},{int(x.max())}] | "
                    f"y=[{int(y.min())},{int(y.max())}] | p={set(np.unique(p).tolist())}"
                )

        except Exception as e:
            fail += 1
            print(f"[FAIL] {f.name}: {repr(e)}")

    print(f"\n[SUMMARY] ok={ok}, skip={skip}, fail={fail}, total={len(files)}")


if __name__ == "__main__":
    main()
