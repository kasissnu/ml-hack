"""Validation checks for PathSnap raw trip files."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .loader import TripData


@dataclass(frozen=True)
class TripValidation:
    trip_id: str
    partition: str
    track_rows: int
    node_rows: int
    arc_rows: int
    route_rows: int
    missing_track_values: int
    invalid_coordinate_rows: int
    timestamp_non_increasing: int
    timestamp_gap_count_over_1s: int
    max_timestamp_gap_seconds: float
    invalid_arc_endpoint_count: int
    invalid_route_arc_count: int
    duplicate_track_rows: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_trip(trip: TripData) -> TripValidation:
    track = trip.track
    arcs = trip.arcs
    nodes = trip.nodes
    route = trip.route

    coordinates = track[["longitude", "latitude"]].to_numpy()
    invalid_coordinates = (~np.isfinite(coordinates).all(axis=1)) | (np.abs(coordinates) > 180).any(axis=1)
    timestamps = track["timestamp"].to_numpy()
    gaps = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)
    endpoint_values = arcs[["from_node_id", "to_node_id"]]
    invalid_endpoints = endpoint_values.lt(0).any(axis=1) | endpoint_values.ge(len(nodes)).any(axis=1)
    invalid_route = 0
    if route is not None:
        invalid_route = int((~route.between(0, len(arcs) - 1)).sum())

    return TripValidation(
        trip_id=trip.trip_id,
        partition=trip.partition,
        track_rows=len(track),
        node_rows=len(nodes),
        arc_rows=len(arcs),
        route_rows=0 if route is None else len(route),
        missing_track_values=int(track.isna().sum().sum()),
        invalid_coordinate_rows=int(invalid_coordinates.sum()),
        timestamp_non_increasing=int((gaps <= 0).sum()),
        timestamp_gap_count_over_1s=int((gaps > 1).sum()),
        max_timestamp_gap_seconds=float(gaps.max()) if len(gaps) else 0.0,
        invalid_arc_endpoint_count=int(invalid_endpoints.sum()),
        invalid_route_arc_count=invalid_route,
        duplicate_track_rows=int(track.duplicated().sum()),
    )
