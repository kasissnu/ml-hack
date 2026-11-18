"""Load one PathSnap trip using the supplied raw file schemas."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TripData:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.Series | None


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the dataset root from an argument, environment, or local defaults."""

    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))
    if os.environ.get("PATHSNAP_DATA_ROOT"):
        candidates.append(Path(os.environ["PATHSNAP_DATA_ROOT"]))
    candidates.extend(
        [
            Path("data/raw"),
            Path(__file__).resolve().parents[3] / "Student folder",
        ]
    )
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find PathSnap data. Set PATHSNAP_DATA_ROOT or pass data_root."
    )


def _trip_dir(data_root: Path, partition: str, trip_id: str) -> Path:
    trip = str(trip_id).zfill(8)
    path = data_root / partition / trip
    if not path.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {path}")
    return path


def load_trip(
    trip_id: str,
    partition: str = "train",
    data_root: str | Path | None = None,
    include_route: bool = True,
) -> TripData:
    """Load a trip with explicit column names and dtypes."""

    root = resolve_data_root(data_root)
    trip = str(trip_id).zfill(8)
    directory = _trip_dir(root, partition, trip)
    prefix = directory / trip

    track = pd.read_csv(
        prefix.with_suffix(".track"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
        dtype={"longitude": "float64", "latitude": "float64", "timestamp": "float64"},
        engine="python",
    )
    nodes = pd.read_csv(
        prefix.with_suffix(".nodes"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
        dtype="float64",
        engine="python",
    )
    arcs = pd.read_csv(
        prefix.with_suffix(".arcs"),
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype="int64",
        engine="python",
    )
    route_path = prefix.with_suffix(".route")
    route = None
    if include_route and route_path.exists():
        route = pd.read_csv(route_path, header=None, names=["arc_id"], dtype="int64")["arc_id"]

    return TripData(
        trip_id=trip,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def list_trip_ids(
    partition: str = "train", data_root: str | Path | None = None
) -> list[str]:
    root = resolve_data_root(data_root)
    partition_dir = root / partition
    return sorted(path.name for path in partition_dir.iterdir() if path.is_dir())
