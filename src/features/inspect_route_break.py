
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRIP_ID = "00000033"

folder = ROOT / "data" / "raw" / "train" / TRIP_ID

nodes = pd.read_csv(
    folder / f"{TRIP_ID}.nodes",
    sep=r"[\s,]+",
    header=None,
    engine="python"
)

arcs = pd.read_csv(
    folder / f"{TRIP_ID}.arcs",
    sep=r"[\s,]+",
    header=None,
    engine="python"
)

route = pd.read_csv(
    folder / f"{TRIP_ID}.route",
    sep=r"[\s,]+",
    header=None,
    engine="python"
).iloc[:, 0].astype(int).tolist()


def show_arc(route_position):

    arc_id = route[route_position]

    start_node = int(arcs.iloc[arc_id, 0])
    end_node = int(arcs.iloc[arc_id, 1])

    start = nodes.iloc[start_node]
    end = nodes.iloc[end_node]

    print(f"\nRoute position: {route_position}")
    print(f"Arc ID: {arc_id}")
    print(f"Start node: {start_node}")
    print(f"End node: {end_node}")

    print(
        f"Start coordinates: "
        f"({start.iloc[0]}, {start.iloc[1]})"
    )

    print(
        f"End coordinates: "
        f"({end.iloc[0]}, {end.iloc[1]})"
    )

    return start_node, end_node, start, end


print("\n========== ROUTE BREAK INSPECTION ==========")

previous = show_arc(164)
following = show_arc(165)

previous_end = previous[1]
next_start = following[0]

print("\n========== CONNECTIVITY CHECK ==========")

print("Previous arc ends at:", previous_end)
print("Next arc starts at:", next_start)

print("Previous end == Next start:", previous_end == next_start)

# Check whether the two arcs are geographically close
p = previous[3]
q = following[2]

lon1, lat1 = float(p.iloc[0]), float(p.iloc[1])
lon2, lat2 = float(q.iloc[0]), float(q.iloc[1])

lat_mid = np.radians((lat1 + lat2) / 2)

dx = (lon2 - lon1) * 111320 * np.cos(lat_mid)
dy = (lat2 - lat1) * 111320

distance = np.sqrt(dx**2 + dy**2)

print(f"Geographic gap: {distance:.2f} metres")

print("\n========== NEIGHBOURING ARCS ==========")

for i in range(162, min(168, len(route))):
    arc_id = route[i]
    print(
        f"Position {i}: Arc {arc_id}, "
        f"{int(arcs.iloc[arc_id, 0])} -> "
        f"{int(arcs.iloc[arc_id, 1])}"
    )
