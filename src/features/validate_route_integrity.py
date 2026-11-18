
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train"

TRIP_IDS = ["00000033", "00000054", "00000001", "00000072"]


def read_file(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )


def validate_trip(trip_id):

    folder = DATA_DIR / trip_id

    nodes = read_file(folder / f"{trip_id}.nodes")
    arcs = read_file(folder / f"{trip_id}.arcs")
    route = read_file(folder / f"{trip_id}.route").iloc[:, 0]

    route = pd.to_numeric(route, errors="coerce").dropna().astype(int)

    invalid_arc_ids = [
        arc_id for arc_id in route
        if arc_id < 0 or arc_id >= len(arcs)
    ]

    valid_route = [
        arc_id for arc_id in route
        if 0 <= arc_id < len(arcs)
    ]

    route_edges = []

    invalid_node_refs = 0

    for arc_id in valid_route:

        start = int(arcs.iloc[arc_id, 0])
        end = int(arcs.iloc[arc_id, 1])

        if not (
            0 <= start < len(nodes)
            and 0 <= end < len(nodes)
        ):
            invalid_node_refs += 1
            continue

        route_edges.append((start, end))

    disconnected = []

    for i in range(len(route_edges) - 1):

        current_end = route_edges[i][1]
        next_start = route_edges[i + 1][0]

        if current_end != next_start:
            disconnected.append(i)

    print(f"\nTRIP: {trip_id}")
    print("-" * 40)
    print("Route arcs:", len(route))
    print("Invalid arc IDs:", len(invalid_arc_ids))
    print("Invalid node references:", invalid_node_refs)
    print("Valid route edges:", len(route_edges))
    print("Disconnected transitions:", len(disconnected))

    if disconnected:
        print("First disconnected positions:", disconnected[:10])

    else:
        print("Route continuity: PASS")


if __name__ == "__main__":

    for trip_id in TRIP_IDS:
        validate_trip(trip_id)
