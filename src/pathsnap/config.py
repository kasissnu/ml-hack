"""Configuration loading and project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a path from config['paths'] relative to the repo root and create it if needed."""
    p = ROOT / load_config()["paths"][key]
    if key != "data_root":
        p.mkdir(parents=True, exist_ok=True)
    return p


def partition_dir(partition: str) -> Path:
    """Directory holding the trips of a partition (train / student_test / eval).

    The env var PATHSNAP_DATA_ROOT overrides the data root (used on the API laptop)."""
    root = Path(os.environ.get("PATHSNAP_DATA_ROOT", ROOT / load_config()["paths"]["data_root"]))
    return root / load_config()["partitions"][partition]
