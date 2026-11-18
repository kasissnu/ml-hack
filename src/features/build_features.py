"""Build leakage-free candidate features for model training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import TripData


FEATURE_COLUMNS = [
    "point_segment_distance_m",
    "midpoint_distance_m",
    "along_segment_fraction",
    "segment_length_m",
    "gps_speed_mps",
    "time_gap_s",
    "gps_bearing_deg",
    "segment_bearing_deg",
    "heading_difference_deg",
    "from_node_degree",
    "to_node_degree",
    "candidate_rank",
    "previous_candidate_connected",
]


def add_features(trip: TripData, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add features that can be computed without reference route labels."""

    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    nodes = trip.nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    endpoints = arcs[result["arc_id"].to_numpy(dtype=np.int64)]
    result["from_node_id"] = endpoints[:, 0]
    result["to_node_id"] = endpoints[:, 1]
    lat0 = float(nodes[:, 1].mean())
    lon_scale = 111_320.0 * np.cos(np.deg2rad(lat0))
    lat_scale = 110_540.0
    segment_delta = nodes[endpoints[:, 1]] - nodes[endpoints[:, 0]]
    result["segment_length_m"] = np.sqrt(
        (segment_delta[:, 0] * lon_scale) ** 2 + (segment_delta[:, 1] * lat_scale) ** 2
    )

    degree = np.bincount(arcs.ravel(), minlength=len(nodes))
    result["from_node_degree"] = degree[endpoints[:, 0]]
    result["to_node_degree"] = degree[endpoints[:, 1]]

    track = trip.track[["longitude", "latitude", "timestamp"]].to_numpy(dtype=float)
    gps_delta = np.diff(track[:, :2], axis=0, prepend=track[:1, :2])
    gps_distance = np.sqrt((gps_delta[:, 0] * lon_scale) ** 2 + (gps_delta[:, 1] * lat_scale) ** 2)
    gps_time_gap = np.diff(track[:, 2], prepend=track[0, 2])
    safe_time_gap = np.maximum(gps_time_gap, 1e-3)
    gps_speed = gps_distance / safe_time_gap
    result["gps_speed_mps"] = result["observation_index"].map(dict(enumerate(gps_speed)))
    result["time_gap_s"] = result["observation_index"].map(dict(enumerate(gps_time_gap)) )

    previous_endpoints: dict[int, set[int]] = {}
    for observation_index, group in result.groupby("observation_index", sort=True):
        previous_endpoints[observation_index] = set(group["from_node_id"]).union(group["to_node_id"])
    connected = []
    for row in result.itertuples(index=False):
        previous = previous_endpoints.get(row.observation_index - 1, set())
        connected.append(bool(row.from_node_id in previous or row.to_node_id in previous))
    result["previous_candidate_connected"] = connected
    return result


def split_features_and_label(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and the binary candidate-correctness target separately."""

    missing = [column for column in FEATURE_COLUMNS if column not in candidates]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return candidates[FEATURE_COLUMNS].copy(), candidates["reference_in_candidates"].astype(int).copy()
