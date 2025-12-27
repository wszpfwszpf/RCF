# rcf_fast/utils_parse.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# Example: Architecture-ND00-1.aedat4
# scene: Architecture
# nd: ND00
# seq_id: 1
_EMLB_RE = re.compile(r"^(?P<scene>.+)-(?P<nd>ND\d{2})-(?P<seq>\d+)\.aedat4$", re.IGNORECASE)


@dataclass(frozen=True)
class EMLBFileInfo:
    path: Path
    scene: str
    nd: str          # e.g., "ND00"
    seq_id: int      # e.g., 1
    basename: str    # stem without extension, e.g., "Architecture-ND00-1"

    def __str__(self) -> str:
        return f"{self.basename}  (scene={self.scene}, nd={self.nd}, seq={self.seq_id})"


def parse_emlb_name(path_or_name: str | Path) -> EMLBFileInfo:
    """
    Parse E-MLB aedat4 filename: <scene>-NDxx-<seq>.aedat4
    Returns EMLBFileInfo with scene/nd/seq.
    """
    p = Path(path_or_name)
    name = p.name
    m = _EMLB_RE.match(name)
    if not m:
        raise ValueError(
            f"Unrecognized E-MLB filename: {name!r}. Expected pattern: <scene>-NDxx-<seq>.aedat4"
        )

    scene = m.group("scene")
    nd = m.group("nd").upper()
    seq_id = int(m.group("seq"))
    basename = p.stem
    return EMLBFileInfo(path=p, scene=scene, nd=nd, seq_id=seq_id, basename=basename)


def find_emlb_files(
    emlb_root: str | Path,
    nd_whitelist: Optional[Iterable[str]] = None,
    scene_whitelist: Optional[Iterable[str]] = None,
    recursive: bool = True,
) -> list[EMLBFileInfo]:
    """
    Scan E-MLB directory and return parsed file infos.
    Directory layout assumed: <emlb_root>/<scene>/*.aedat4
    """
    root = Path(emlb_root)
    if not root.exists():
        raise FileNotFoundError(f"EMLB root not found: {root}")

    nd_set = {s.upper() for s in nd_whitelist} if nd_whitelist else None
    scene_set = {s for s in scene_whitelist} if scene_whitelist else None

    pattern = "**/*.aedat4" if recursive else "*.aedat4"
    infos: list[EMLBFileInfo] = []

    for fp in root.glob(pattern):
        if not fp.is_file():
            continue
        try:
            info = parse_emlb_name(fp)
        except ValueError:
            # Skip non-conforming files (e.g., hidden/temp)
            continue

        if scene_set is not None and info.scene not in scene_set:
            continue
        if nd_set is not None and info.nd.upper() not in nd_set:
            continue

        infos.append(info)

    # Sort for stable processing: scene -> nd -> seq
    infos.sort(key=lambda x: (x.scene, x.nd, x.seq_id))
    return infos


def build_mask_output_path(mask_root: str | Path, info: EMLBFileInfo, eta: float) -> Path:
    """
    Recommended mask output path:
      <mask_root>/<scene>/<basename>_eta015.mask
    """
    mask_root = Path(mask_root)
    eta_tag = f"eta{eta:.2f}".replace(".", "")
    return mask_root / info.scene / f"{info.basename}_{eta_tag}.mask"


def build_profile_csv_path(metric_root: str | Path, tag: str = "nd00") -> Path:
    """
    Recommended profiling csv output path:
      <metric_root>/profile_rcf_<tag>.csv
    """
    metric_root = Path(metric_root)
    return metric_root / f"profile_rcf_{tag}.csv"
