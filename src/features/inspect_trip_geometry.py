
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

TRAIN_DIR = Path("data/raw/train")
CANDIDATE_DIR = Path("data/processed/candidates")

TRIPS = [
    "00000001",
    "00000033",
    "00000054",
    "00000072",
]


def load_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None)


def inspect_trip(trip_id):
    folder = TRAIN_DIR / trip_id

    track = load_data(folder / f"{trip_id}.track")
    nodes = load_data(folder / f"{trip_id}.nodes")
    arcs = load_data(folder / f"{trip_id}.arcs")
    route = load_data(folder / f"{trip_id}.route")

    gps = track.iloc[:, :2].to_numpy(dtype=float)
    node_coords = nodes.iloc[:, :2].to_numpy(dtype=float)

    print("\n" + "=" * 70)
    print(f"TRIP: {trip_id}")
    print("=" * 70)

    print("GPS shape:", track.shape)
    print("Nodes shape:", nodes.shape)
    print("Arcs shape:", arcs.shape)
    print("Route shape:", route.shape)

    print("\nGPS coordinate bounds:")
    print("Longitude:", gps[:, 0].min(), "to", gps[:, 0].max())
    print("Latitude :", gps[:, 1].min(), "to", gps[:, 1].max())

    print("\nNode coordinate bounds:")
    print("Longitude:", node_coords[:, 0].min(), "to", node_coords[:, 0].max())
    print("Latitude :", node_coords[:, 1].min(), "to", node_coords[:, 1].max())

    # Build a local metric projection for meaningful distance comparison.
    center_lon = float(np.mean(gps[:, 0]))
    center_lat = float(np.mean(gps[:, 1]))

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True
    )

    gx, gy = transformer.transform(gps[:, 0], gps[:, 1])
    nx, ny = transformer.transform(node_coords[:, 0], node_coords[:, 1])

    gps_m = np.column_stack([gx, gy])
    nodes_m = np.column_stack([nx, ny])

    print("\nGPS-to-nearest-network-node distances (metres):")

    # Sample up to 2000 GPS observations to keep this diagnostic quick.
    sample_indices = np.linspace(
        0, len(gps_m) - 1, min(2000, len(gps_m)), dtype=int
    )

    sample_gps = gps_m[sample_indices]

    # Chunk the calculation to avoid allocating a huge distance matrix.
    nearest_distances = []

    for start in range(0, len(sample_gps), 100):
        chunk = sample_gps[start:start + 100]
        distances = np.sqrt(
            ((chunk[:, None, :] - nodes_m[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_distances.extend(distances.min(axis=1))

    print(
        pd.Series(nearest_distances).describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(2).to_string()
    )

    # Validate route arc IDs and endpoints.
    arc_array = arcs.iloc[:, :2].to_numpy(dtype=int)
    route_ids = route.iloc[:, 0].to_numpy(dtype=int)

    invalid_route_ids = route_ids[
        (route_ids < 0) | (route_ids >= len(arc_array))
    ]

    print("\nRoute ID diagnostics:")
    print("Route entries:", len(route_ids))
    print("Unique route IDs:", len(np.unique(route_ids)))
    print("Invalid route IDs:", len(invalid_route_ids))

    valid_ids = route_ids[
        (route_ids >= 0) & (route_ids < len(arc_array))
    ]

    if len(valid_ids):
        endpoints = arc_array[valid_ids]

        invalid_endpoints = (
            (endpoints < 0) | (endpoints >= len(nodes))
        ).any(axis=1)

        print("Route arcs with invalid node endpoints:",
              int(invalid_endpoints.sum()))

        print("First 10 route IDs:", valid_ids[:10].tolist())
        print("First 10 route arc endpoints:")
        print(endpoints[:10])

    # Inspect candidate distance values.
    candidate_file = (
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if candidate_file.exists():
        candidates = pd.read_csv(candidate_file)

        print("\nCandidate columns:", candidates.columns.tolist())
        print("Candidate records:", len(candidates))

        if "distance_m" in candidates.columns:
            print("\nCandidate distance distribution (metres):")
            print(
                candidates["distance_m"]
                .describe(
                    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
                )
                .round(2)
                .to_string()
            )

            per_observation = candidates.groupby(
                "observation_id"
            )["distance_m"].min()

            print("\nNearest candidate distance per observation:")
            print(
                per_observation.describe(
                    percentiles=[0.50, 0.90, 0.95, 0.99]
                ).round(2).to_string()
            )
    else:
        print("\nCandidate file not found:", candidate_file)


for trip in TRIPS:
    inspect_trip(trip)

print("\n" + "=" * 70)
print("GEOMETRY INSPECTION COMPLETED")
print("=" * 70)
