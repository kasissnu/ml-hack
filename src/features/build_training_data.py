
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


TRIP_ID = "00000001"

TRIP_DIR = Path("data/raw/train") / TRIP_ID
CANDIDATE_FILE = (
    Path("data/processed/candidates")
    / f"{TRIP_ID}_candidates.csv"
)

OUTPUT_DIR = Path("data/processed/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # Load original data
    track = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.track",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=float)

    nodes = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.nodes",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=float)

    arcs = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.arcs",
        sep=r"\s+",
        header=None
    ).to_numpy(dtype=int)

    route = pd.read_csv(
        TRIP_DIR / f"{TRIP_ID}.route",
        sep=r"\s+",
        header=None
    ).iloc[:, 0].to_numpy(dtype=int)

    candidates = pd.read_csv(CANDIDATE_FILE)

    # Validate reference route
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid reference route arc IDs.")

    # Project geographic coordinates into metres
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

    # Get reference route geometry in chronological order
    route_arcs = arcs[route]

    sx = node_x[route_arcs[:, 0]]
    sy = node_y[route_arcs[:, 0]]
    ex = node_x[route_arcs[:, 1]]
    ey = node_y[route_arcs[:, 1]]

    dx = ex - sx
    dy = ey - sy

    length_squared = dx**2 + dy**2
    denominator = np.maximum(length_squared, 1e-12)

    labels = []
    nearest_distances = []
    nearest_route_positions = []
    second_best_distances = []

    # Find nearest reference-route segment for each GPS observation
    for px, py in zip(gps_x, gps_y):

        t = (
            (px - sx) * dx + (py - sy) * dy
        ) / denominator

        t = np.clip(t, 0, 1)

        projected_x = sx + t * dx
        projected_y = sy + t * dy

        distances = np.sqrt(
            (px - projected_x)**2 +
            (py - projected_y)**2
        )

        order = np.argsort(distances)

        best_position = int(order[0])

        labels.append(int(route[best_position]))
        nearest_distances.append(float(distances[best_position]))
        nearest_route_positions.append(best_position)

        second_best_distances.append(
            float(distances[order[1]])
            if len(order) > 1 else np.nan
        )

    # Attach preliminary labels to candidates
    candidates["label_arc_id"] = candidates[
        "observation_id"
    ].map(dict(enumerate(labels)))

    candidates["reference_distance_m"] = candidates[
        "observation_id"
    ].map(dict(enumerate(nearest_distances)))

    candidates["reference_route_position"] = candidates[
        "observation_id"
    ].map(dict(enumerate(nearest_route_positions)))

    candidates["reference_second_best_distance_m"] = candidates[
        "observation_id"
    ].map(dict(enumerate(second_best_distances)))

    # Binary classification target
    candidates["label"] = (
        candidates["arc_id"] == candidates["label_arc_id"]
    ).astype(int)

    # Candidate recall at each GPS observation
    observation_recall = (
        candidates.groupby("observation_id")["label"]
        .max()
    )

    recall = float(observation_recall.mean() * 100)

    # Save dataset
    output_file = OUTPUT_DIR / f"{TRIP_ID}_training_data.csv"
    candidates.to_csv(output_file, index=False)

    print("=" * 65)
    print("PATHSNAP - OBSERVATION-LEVEL TRAINING DATA")
    print("=" * 65)

    print("GPS observations:", len(track))
    print("Candidate records:", len(candidates))
    print("Positive candidate records:", int(candidates["label"].sum()))
    print("Negative candidate records:", int((candidates["label"] == 0).sum()))

    print(f"\nObservation-level candidate recall: {recall:.2f}%")

    print("\nLabel distribution:")
    print(candidates["label"].value_counts())

    print("\nSample training records:")
    print(candidates.head(10).to_string(index=False))

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    main()
