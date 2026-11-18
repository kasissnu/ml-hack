
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


TRAIN_DIR = Path("data/raw/train")
CANDIDATE_DIR = Path("data/processed/candidates")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)


def validate_trip(trip_dir):

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

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None
    ).iloc[:, 0].to_numpy(dtype=int)

    candidates = pd.read_csv(
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid reference route arc ID")

    # Project coordinates into metres
    center_lon = float(track[:, 0].mean())
    center_lat = float(track[:, 1].mean())

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} "
        f"+lon_0={center_lon} +datum=WGS84 "
        "+units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    # Build geometries for the chronological reference route
    route_arcs = arcs[route]

    geometries = [
        LineString([
            (node_x[u], node_y[u]),
            (node_x[v], node_y[v])
        ])
        for u, v in route_arcs
    ]

    route_tree = STRtree(geometries)

    # Candidate arc IDs available for each observation
    candidate_map = (
        candidates.groupby("observation_id")["arc_id"]
        .apply(set)
        .to_dict()
    )

    covered = 0
    nearest_distances = []
    missing_observations = 0

    for obs_id, (x, y) in enumerate(zip(gps_x, gps_y)):

        point = Point(x, y)

        # Nearest reference-route segment
        nearest_index = int(
            np.asarray(
                route_tree.query_nearest(point)
            ).reshape(-1)[0]
        )

        reference_arc_id = int(route[nearest_index])

        nearest_distance = point.distance(
            geometries[nearest_index]
        )

        nearest_distances.append(nearest_distance)

        available = candidate_map.get(obs_id, set())

        if reference_arc_id in available:
            covered += 1
        else:
            missing_observations += 1

    total = len(track)

    return {
        "trip_id": trip_id,
        "gps_observations": total,
        "covered_observations": covered,
        "missed_observations": missing_observations,
        "candidate_recall_proxy_pct": (
            covered / total * 100 if total else 0
        ),
        "mean_nearest_route_distance_m": np.mean(
            nearest_distances
        ),
        "max_nearest_route_distance_m": np.max(
            nearest_distances
        )
    }


def main():

    trip_dirs = sorted(
        p for p in TRAIN_DIR.iterdir() if p.is_dir()
    )

    results = []

    print("=" * 65)
    print("PATHSNAP - ALL-TRIP CANDIDATE VALIDATION")
    print("=" * 65)

    for i, trip_dir in enumerate(trip_dirs, start=1):

        try:
            result = validate_trip(trip_dir)
            results.append(result)

            print(
                f"[{i}/{len(trip_dirs)}] "
                f"{result['trip_id']} : "
                f"{result['candidate_recall_proxy_pct']:.2f}%"
            )

        except Exception as error:
            print(f"FAILED: {trip_dir.name} - {error}")

    if not results:
        raise RuntimeError("No trips were validated.")

    df = pd.DataFrame(results)

    output = REPORT_DIR / "all_trip_candidate_validation.csv"
    df.to_csv(output, index=False)

    total_observations = df["gps_observations"].sum()
    total_covered = df["covered_observations"].sum()

    overall_recall = (
        total_covered / total_observations * 100
    )

    print("\n" + "=" * 65)
    print("OVERALL VALIDATION")
    print("=" * 65)

    print("Trips validated:", len(df))
    print("Total GPS observations:", total_observations)
    print("Covered observations:", total_covered)
    print(
        "Missed observations:",
        df["missed_observations"].sum()
    )
    print(f"Overall candidate recall proxy: {overall_recall:.2f}%")

    print("\nLOWEST COVERAGE TRIPS")
    print(
        df.nsmallest(10, "candidate_recall_proxy_pct")[
            ["trip_id", "candidate_recall_proxy_pct",
             "missed_observations"]
        ].to_string(index=False)
    )

    print("\nValidation report saved to:", output)


if __name__ == "__main__":
    main()
