
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def load_trip(trip_dir):
    trip_id = trip_dir.name

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=float)

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=float)

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=int)

    return track, nodes, arcs


def generate_candidates(
    track,
    nodes,
    arcs,
    radius_m=100,
    top_n=8
):
    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError("GPS track or road network is empty.")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Invalid road-network node IDs.")

    # Local metric coordinate system
    center_lon = float(np.mean(track[:, 0]))
    center_lat = float(np.mean(track[:, 1]))

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} "
        f"+lon_0={center_lon} +datum=WGS84 "
        "+units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    # Construct actual road-segment geometries.
    geometries = [
        LineString([
            (node_x[u], node_y[u]),
            (node_x[v], node_y[v])
        ])
        for u, v in arcs
    ]

    # Spatial index over full road geometries.
    tree = STRtree(geometries)

    output = []

    for i, (px, py) in enumerate(zip(gps_x, gps_y)):

        point = Point(px, py)

        # Retrieve every road segment intersecting
        # the search-radius buffer.
        search_area = point.buffer(radius_m)
        candidate_ids = tree.query(search_area)

        candidate_ids = np.asarray(
            candidate_ids, dtype=int
        )

        # If no segment is found, expand the search.
        if len(candidate_ids) == 0:
            candidate_ids = np.asarray(
                tree.query(point.buffer(radius_m * 2)),
                dtype=int
            )

        if len(candidate_ids) == 0:
            candidate_ids = np.asarray(
                tree.query_nearest(point),
                dtype=int
            )

        # Calculate exact point-to-segment distances.
        distances = []
        bearings = []
        lengths = []

        for arc_id in candidate_ids:

            line = geometries[arc_id]

            distance = point.distance(line)

            u, v = arcs[arc_id]

            dx = node_x[v] - node_x[u]
            dy = node_y[v] - node_y[u]

            bearing = (
                np.degrees(np.arctan2(dx, dy)) % 360
            )

            distances.append(distance)
            bearings.append(bearing)
            lengths.append(np.hypot(dx, dy))

        distances = np.asarray(distances)
        bearings = np.asarray(bearings)
        lengths = np.asarray(lengths)

        order = np.argsort(distances)[:top_n]

        # GPS heading from consecutive observations.
        if i < len(track) - 1:
            heading = np.degrees(
                np.arctan2(
                    gps_x[i + 1] - px,
                    gps_y[i + 1] - py
                )
            ) % 360
        elif i > 0:
            heading = np.degrees(
                np.arctan2(
                    px - gps_x[i - 1],
                    py - gps_y[i - 1]
                )
            ) % 360
        else:
            heading = np.nan

        for rank, j in enumerate(order, start=1):

            arc_id = int(candidate_ids[j])

            heading_difference = (
                (bearings[j] - heading + 180) % 360 - 180
            )

            output.append({
                "observation_id": i,
                "timestamp": track[i, 2],
                "arc_id": arc_id,
                "candidate_rank": rank,
                "distance_m": float(distances[j]),
                "segment_length_m": float(lengths[j]),
                "road_bearing_deg": float(bearings[j]),
                "gps_heading_deg": float(heading),
                "heading_difference_deg": float(
                    heading_difference
                )
            })

    return pd.DataFrame(output)


if __name__ == "__main__":

    trip_dir = Path("data/raw/train/00000001")

    track, nodes, arcs = load_trip(trip_dir)

    candidates = generate_candidates(
        track,
        nodes,
        arcs,
        radius_m=100,
        top_n=8
    )

    output_dir = Path("data/processed/candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "00000001_candidates.csv"

    candidates.to_csv(output_file, index=False)

    counts = candidates.groupby("observation_id").size()

    print("=" * 60)
    print("PATHSNAP - STRTREE CANDIDATE GENERATION")
    print("=" * 60)

    print("GPS observations:", len(track))
    print("Road arcs:", len(arcs))
    print("Candidate records:", len(candidates))

    print("\nCandidates per observation:")
    print(counts.describe())

    print("\nObservations with fewer than 8 candidates:",
          int((counts < 8).sum()))

    print("\nSample candidates:")
    print(candidates.head(10).to_string(index=False))

    print("\nSaved to:", output_file)
