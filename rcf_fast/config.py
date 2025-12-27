
# rcf_fast/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_repo_root() -> Path:
    # <repo>/rcf_fast/config.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RCFConfig:
    # -----------------------------
    # Repo paths
    # -----------------------------
    REPO_ROOT: Path = get_repo_root()

    # Dataset root layout:
    # <repo>/data/emlb/day/<scene>/*.aedat4
    # <repo>/data/emlb/night/<scene>/*.aedat4
    EMLB_ROOT: Path = REPO_ROOT / "data" / "emlb"
    EMLB_SPLIT: str = "day"  # change to "night" later if needed
    EMLB_DATA_ROOT: Path = EMLB_ROOT / EMLB_SPLIT

    # Outputs
    OUT_MASK_ROOT: Path = REPO_ROOT / "data" / "emlb_masks"
    OUT_METRIC_ROOT: Path = REPO_ROOT / "data" / "emlb_esr"

    # -----------------------------
    # Run configs (PyCharm one-click)
    # -----------------------------
    ND_FILTER: str = "ND00"   # Step2 only processes this ND
    LIMIT_FILES: int = 0      # 0 = no limit
    MAX_EVENTS_PER_FILE: int = 0  # 0 = full pass; set e.g. 500_000 for quick sanity check

    # -----------------------------
    # Algorithm params (placeholders for now)
    # -----------------------------
    BIN_US: int = 10_000
    ETA: float = 0.15


CFG = RCFConfig()

