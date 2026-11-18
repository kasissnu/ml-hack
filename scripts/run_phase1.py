"""Run the repeatable Phase 1 inventory, validation, network, and alignment audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from data.loader import list_trip_ids, load_trip, resolve_data_root
from data.validation import validate_trip


def _metric_distance_degrees(points: np.ndarray, lat0: float) -> np.ndarray:
    lon_scale = 111_320.0 * np.cos(np.deg2rad(lat0))
    lat_scale = 110_540.0
    return np.sqrt((points[:, 0] * lon_scale) ** 2 + (points[:, 1] * lat_scale) ** 2)


def audit_trip(trip) -> dict[str, object]:
    validation = validate_trip(trip).as_dict()
    arcs = trip.arcs.to_numpy(dtype=np.int64)
    nodes = trip.nodes.to_numpy(dtype=float)
    route = trip.route.to_numpy(dtype=np.int64) if trip.route is not None else np.array([], dtype=np.int64)

    degrees = np.zeros(len(nodes), dtype=np.int64)
    if len(arcs):
        np.add.at(degrees, arcs[:, 0], 1)
        np.add.at(degrees, arcs[:, 1], 1)
    graph = csr_matrix(
        (np.ones(len(arcs) * 2, dtype=np.uint8), (np.r_[arcs[:, 0], arcs[:, 1]], np.r_[arcs[:, 1], arcs[:, 0]])),
        shape=(len(nodes), len(nodes)),
    ) if len(arcs) else csr_matrix((len(nodes), len(nodes)))
    components, labels = connected_components(graph, directed=False)
    validation.update(
        {
            "isolated_nodes": int((degrees == 0).sum()),
            "weak_components": int(components),
            "largest_component_nodes": int(np.bincount(labels).max()) if len(labels) else 0,
        }
    )

    route_transition_breaks = 0
    route_midpoint_distance_median = np.nan
    route_midpoint_distance_p95 = np.nan
    route_index_non_decreasing_fraction = np.nan
    route_index_coverage = np.nan
    if len(route):
        route_arcs = arcs[route]
        route_transition_breaks = int((route_arcs[:-1, 1] != route_arcs[1:, 0]).sum()) if len(route) > 1 else 0
        mids = (nodes[route_arcs[:, 0]] + nodes[route_arcs[:, 1]]) / 2
        gps = trip.track[["longitude", "latitude"]].to_numpy(dtype=float)
        nearest = cKDTree(mids).query(gps, k=1)[1]
        distances = _metric_distance_degrees(gps - mids[nearest], float(gps[:, 1].mean()))
        steps = np.diff(nearest)
        route_midpoint_distance_median = float(np.median(distances))
        route_midpoint_distance_p95 = float(np.percentile(distances, 95))
        route_index_non_decreasing_fraction = float((steps >= 0).mean()) if len(steps) else 1.0
        route_index_coverage = float(len(np.unique(nearest)) / len(route))

    validation.update(
        {
            "route_transition_breaks": route_transition_breaks,
            "route_midpoint_distance_median_m": route_midpoint_distance_median,
            "route_midpoint_distance_p95_m": route_midpoint_distance_p95,
            "route_index_non_decreasing_fraction": route_index_non_decreasing_fraction,
            "route_index_coverage": route_index_coverage,
        }
    )
    return validation


def write_representative_map(trip, output: Path) -> None:
    track = trip.track[["longitude", "latitude"]].to_numpy(dtype=float)
    route = trip.route.to_numpy(dtype=np.int64)
    arcs = trip.arcs.to_numpy(dtype=np.int64)
    nodes = trip.nodes.to_numpy(dtype=float)
    route_nodes = nodes[arcs[route, 0]]
    center = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    folium.PolyLine(track[:, ::-1].tolist(), color="blue", weight=3, opacity=0.7, tooltip="GPS track").add_to(fmap)
    folium.PolyLine(route_nodes[:, ::-1].tolist(), color="green", weight=3, opacity=0.8, tooltip="Reference route arcs").add_to(fmap)
    folium.LayerControl().add_to(fmap)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="reports/eda")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    root = resolve_data_root(args.data_root)

    rows = []
    for partition in ("train", "student_test"):
        for trip_id in list_trip_ids(partition, root):
            rows.append(audit_trip(load_trip(trip_id, partition, root)))
    frame = pd.DataFrame(rows).sort_values(["partition", "trip_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "trip_audit.csv", index=False)
    frame.groupby("partition", as_index=False).agg(
        trips=("trip_id", "count"),
        gps_rows=("track_rows", "sum"),
        route_rows=("route_rows", "sum"),
        isolated_nodes_mean=("isolated_nodes", "mean"),
        weak_components_mean=("weak_components", "mean"),
        route_transition_breaks_total=("route_transition_breaks", "sum"),
        midpoint_distance_median_mean=("route_midpoint_distance_median_m", "mean"),
        midpoint_distance_p95_mean=("route_midpoint_distance_p95_m", "mean"),
        monotonic_fraction_mean=("route_index_non_decreasing_fraction", "mean"),
        route_coverage_mean=("route_index_coverage", "mean"),
    ).to_csv(output_dir / "partition_summary.csv", index=False)

    sample_id = list_trip_ids("train", root)[0]
    sample = load_trip(sample_id, "train", root)
    write_representative_map(sample, output_dir / f"trip_{sample_id}_track_route.html")
    print(f"Wrote Phase 1 audit for {len(frame)} trips to {output_dir}")


if __name__ == "__main__":
    main()
