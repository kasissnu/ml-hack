"""Reading raw trip files.

File formats (whitespace separated, no header):
  <id>.track : lon lat t          (GPS observations, t in seconds)
  <id>.nodes : lon lat            (row index = node id)
  <id>.arcs  : from_node to_node  (row index = arc id, directed)
  <id>.route : arc_id             (reference route, chronological)

The reference route is deliberately loaded by a separate function (`load_route`) so that the
prediction pipeline never has access to it. Only label creation and evaluation call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N, 3) lon, lat, t  -- cleaned, time-sorted
    nodes: np.ndarray          # (M, 2) lon, lat
    arcs: np.ndarray           # (A, 2) int64 from, to
    cleaning: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    return df.to_numpy(dtype=dtype)


def list_trips(partition_dir: Path) -> list[str]:
    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        return []
    return sorted(p.name for p in partition_dir.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


def read_raw_track(trip_dir: Path) -> np.ndarray:
    trip_dir = Path(trip_dir)
    return _read(trip_dir / f"{trip_dir.name}.track", np.float64)


def clean_track(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop non-finite / implausible rows, sort by time, drop duplicate timestamps."""
    info = {"raw_rows": int(len(raw))}
    finite = np.isfinite(raw).all(axis=1)
    plausible = (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    keep = finite & plausible
    info["dropped_nonfinite"] = int((~finite).sum())
    info["dropped_out_of_range"] = int((finite & ~plausible).sum())
    tr = raw[keep]
    info["was_sorted"] = bool(np.all(np.diff(tr[:, 2]) >= 0))
    order = np.argsort(tr[:, 2], kind="stable")
    tr = tr[order]
    # timestamps such as 0.9999999 vs 1.0 are float noise: compare on a 1e-3 s grid
    t_round = np.round(tr[:, 2], 3)
    dup = np.r_[False, np.diff(t_round) == 0]
    info["dropped_duplicate_time"] = int(dup.sum())
    tr = tr[~dup]
    info["clean_rows"] = int(len(tr))
    return tr, info


def load_trip(trip_dir: Path) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track, info = clean_track(read_raw_track(trip_dir))
    nodes = _read(trip_dir / f"{tid}.nodes", np.float64)
    arcs = _read(trip_dir / f"{tid}.arcs", np.int64)
    return Trip(tid, track, nodes, arcs, info)


def load_route(trip_dir: Path) -> np.ndarray | None:
    """Reference route (ground truth). Never call this inside the prediction pipeline."""
    trip_dir = Path(trip_dir)
    p = trip_dir / f"{trip_dir.name}.route"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return _read(p, np.int64).ravel()
