
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

TRAIN_DIR = Path("data/raw/train")

TRIPS = [
    "00000001",
    "00000033",
    "00000054",
    "00000072",
]


def load_trip(trip_id):
    folder = TRAIN_DIR / trip_id

    track = pd.read_csv(
        folder / f"{trip_id}.track",
        sep=r"\s+", header=None
    ).to_numpy(dtype=float)

    nodes = pd.read_csv(
        folder / f"{trip_id}.nodes",
        sep=r"\s+", header=None
    ).to_numpy(dtype=float)

    arcs = pd.read_csv(
        folder / f"{trip_id}.arcs",
        sep=r"\s+", header=None
    ).to_numpy(dtype=int)

    route = pd.read_csv(
        folder / f"{trip_id}.route",
        sep=r"\s+", header=None
    ).iloc[:, 0].to_numpy(dtype=int)

    return track, nodes, arcs, route


def inspect_alignment(trip_id):
    track, nodes, arcs, route = load_trip(trip_id)

    gps = track[:, :2]

    center_lon = gps[:, 0].mean()
    center_lat = gps[:, 1].mean()

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True
    )

    gx, gy = transformer.transform(gps[:, 0], gps[:, 1])
    nx, ny = transformer.transform(nodes[:, 0], nodes[:, 1])

    gps_m = np.column_stack([gx, gy])
    nodes_m = np.column_stack([nx, ny])

    # Construct route segments in supplied route order.
    valid_route = route[
        (route >= 0) & (route < len(arcs))
    ]

    route_geometries = []
    valid_route_positions = []

    for route_position, arc_id in enumerate(route):
        if arc_id < 0 or arc_id >= len(arcs):
            continue

        start_node, end_node = arcs[arc_id]

        if (
            start_node < 0 or end_node < 0
            or start_node >= len(nodes_m)
            or end_node >= len(nodes_m)
        ):
            continue

        line = LineString([
            nodes_m[start_node],
            nodes_m[end_node]
        ])

        if line.length > 0:
            route_geometries.append(line)
            valid_route_positions.append(route_position)

    tree = STRtree(route_geometries)

    matched_positions = []
    matched_distances = []

    # Find nearest route segment for every GPS point.
    for x, y in gps_m:
        point = Point(x, y)
        nearest_indices = tree.query_nearest(point)

        # Resolve ties by choosing the first returned route segment.
        index = int(np.asarray(nearest_indices).reshape(-1)[0])

        matched_positions.append(valid_route_positions[index])
        matched_distances.append(point.distance(route_geometries[index]))

    positions = np.asarray(matched_positions)
    distances = np.asarray(matched_distances)

    # A backward jump means the nearest route position moves backwards.
    position_differences = np.diff(positions)
    backward_jumps = int((position_differences < 0).sum())

    print("\n" + "=" * 65)
    print("TRIP:", trip_id)
    print("=" * 65)

    print("GPS observations:", len(gps_m))
    print("Reference route entries:", len(route))
    print("Valid route segments:", len(route_geometries))

    print("\nNearest route distance (metres):")
    print(
        pd.Series(distances).describe(
            percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
        ).round(2)
    )

    print("\nRoute-position progression:")
    print("First matched position:", positions[0])
    print("Last matched position:", positions[-1])
    print("Backward jumps:", backward_jumps)
    print("Forward jumps:", int((position_differences > 0).sum()))
    print("Unchanged positions:", int((position_differences == 0).sum()))

    # Show representative observations.
    sample = np.linspace(
        0, len(positions) - 1, min(15, len(positions)), dtype=int
    )

    print("\nSample chronological matches:")
    print(
        pd.DataFrame({
            "observation_id": sample,
            "timestamp": track[sample, 2],
            "matched_route_position": positions[sample],
            "distance_m": distances[sample],
        }).round(2).to_string(index=False)
    )


for trip_id in TRIPS:
    inspect_alignment(trip_id)

print("\nROUTE ALIGNMENT DIAGNOSTIC COMPLETED")
