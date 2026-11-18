"""Generate road-arc candidates for each GPS observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data.loader import TripData


@dataclass(frozen=True)
class Projection:
    longitude_scale: float
    latitude_scale: float
    longitude_origin: float
    latitude_origin: float

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                (coordinates[:, 0] - self.longitude_origin) * self.longitude_scale,
                (coordinates[:, 1] - self.latitude_origin) * self.latitude_scale,
            )
        )


def make_projection(track: pd.DataFrame, nodes: pd.DataFrame) -> Projection:
    coordinates = np.vstack(
        (
            track[["longitude", "latitude"]].to_numpy(dtype=float),
            nodes[["longitude", "latitude"]].to_numpy(dtype=float),
        )
    )
    latitude_origin = float(coordinates[:, 1].mean())
    return Projection(
        longitude_scale=111_320.0 * np.cos(np.deg2rad(latitude_origin)),
        latitude_scale=110_540.0,
        longitude_origin=float(coordinates[:, 0].mean()),
        latitude_origin=latitude_origin,
    )


def _point_segment_projection(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = ends - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    safe_lengths_squared = np.where(lengths_squared == 0, 1.0, lengths_squared)
    factors = np.einsum("ij,ij->i", points - starts, vectors) / safe_lengths_squared
    factors = np.clip(factors, 0.0, 1.0)
    projected = starts + factors[:, None] * vectors
    distance = np.linalg.norm(points - projected, axis=1)
    return distance, factors


def align_track_to_route(trip: TripData, route_k: int = 32) -> np.ndarray:
    """Return provisional reference arc IDs aligned to GPS rows.

    The route is shorter than the track, so this uses nearest route-arc
    midpoints followed by a non-decreasing route-index constraint.
    """

    if trip.route is None or len(trip.route) == 0:
        return np.full(len(trip.track), -1, dtype=np.int64)
    nodes = trip.nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    route_ids = trip.route.to_numpy(dtype=np.int64)
    route_arcs = arcs[route_ids]
    midpoints = (nodes[route_arcs[:, 0]] + nodes[route_arcs[:, 1]]) / 2.0
    gps = trip.track[["longitude", "latitude"]].to_numpy(dtype=float)
    projection = make_projection(trip.track, trip.nodes)
    route_xy = projection.transform(midpoints)
    gps_xy = projection.transform(gps)
    actual_k = min(route_k, len(route_ids))
    distances, nearby_indices = cKDTree(route_xy).query(gps_xy, k=actual_k)
    if actual_k == 1:
        distances = distances[:, None]
        nearby_indices = nearby_indices[:, None]

    observations = len(gps_xy)
    route_length = len(route_ids)
    selected = np.empty(observations, dtype=np.int64)
    previous_index = 0
    for observation_index in range(observations):
        expected = observation_index * (route_length - 1) / max(observations - 1, 1)
        indices = nearby_indices[observation_index].astype(np.int64)
        candidate_distances = distances[observation_index]
        forward = indices >= previous_index
        if forward.any():
            scores = candidate_distances + 0.1 * np.abs(indices - expected)
            scores = np.where(forward, scores, np.inf)
        else:
            scores = candidate_distances + 5.0 * (previous_index - indices)
        selected_index = int(np.argmin(scores))
        previous_index = int(indices[selected_index])
        selected[observation_index] = previous_index
    return route_ids[selected]


def generate_candidates(trip: TripData, k: int = 8) -> pd.DataFrame:
    """Generate up to ``k`` nearest arc candidates for every GPS observation."""

    if k < 1:
        raise ValueError("k must be at least 1")
    projection = make_projection(trip.track, trip.nodes)
    nodes = trip.nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    starts = projection.transform(nodes[arcs[:, 0]])
    ends = projection.transform(nodes[arcs[:, 1]])
    midpoints = (starts + ends) / 2.0
    gps = projection.transform(trip.track[["longitude", "latitude"]].to_numpy(dtype=float))
    actual_k = min(k, len(arcs))
    distances, candidate_ids = cKDTree(midpoints).query(gps, k=actual_k)
    if actual_k == 1:
        distances = distances[:, None]
        candidate_ids = candidate_ids[:, None]

    rows = []
    timestamp = trip.track["timestamp"].to_numpy(dtype=float)
    bearings = np.degrees(np.arctan2(np.diff(gps[:, 0], prepend=gps[0, 0]), np.diff(gps[:, 1], prepend=gps[0, 1])))
    for observation_index in range(len(gps)):
        ids = candidate_ids[observation_index]
        candidate_starts = starts[ids]
        candidate_ends = ends[ids]
        point_distances, along_fractions = _point_segment_projection(
            np.repeat(gps[observation_index][None, :], actual_k, axis=0),
            candidate_starts,
            candidate_ends,
        )
        segment_bearings = np.degrees(
            np.arctan2(
                candidate_ends[:, 0] - candidate_starts[:, 0],
                candidate_ends[:, 1] - candidate_starts[:, 1],
            )
        )
        heading_delta = np.abs((bearings[observation_index] - segment_bearings + 180.0) % 360.0 - 180.0)
        for rank, (arc_id, midpoint_distance, point_distance, along_fraction, segment_bearing, delta) in enumerate(
            zip(ids, distances[observation_index], point_distances, along_fractions, segment_bearings, heading_delta)
        ):
            rows.append(
                {
                    "trip_id": trip.trip_id,
                    "partition": trip.partition,
                    "observation_index": observation_index,
                    "timestamp": timestamp[observation_index],
                    "candidate_rank": rank,
                    "arc_id": int(arc_id),
                    "midpoint_distance_m": float(midpoint_distance),
                    "point_segment_distance_m": float(point_distance),
                    "along_segment_fraction": float(along_fraction),
                    "segment_bearing_deg": float(segment_bearing),
                    "gps_bearing_deg": float(bearings[observation_index]),
                    "heading_difference_deg": float(delta),
                }
            )
    candidates = pd.DataFrame(rows)
    labels = align_track_to_route(trip)
    candidates["reference_arc_id"] = candidates["observation_index"].map(dict(enumerate(labels))).astype("int64")
    candidates["reference_in_candidates"] = candidates["arc_id"].eq(candidates["reference_arc_id"])
    return candidates
