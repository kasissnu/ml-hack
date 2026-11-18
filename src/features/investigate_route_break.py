
from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
TRIP_ID = "00000033"

folder = ROOT / "data" / "raw" / "train" / TRIP_ID

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

# Build directed road network
G = nx.DiGraph()

for arc_id, row in arcs.iterrows():
    start = int(row.iloc[0])
    end = int(row.iloc[1])
    G.add_edge(start, end, arc_id=arc_id)

# Investigate the disconnected transition
previous_end = 1283
next_start = 563

print("\n========== DIRECT CONNECTION CHECK ==========")

matching = arcs[
    (arcs[0] == previous_end) &
    (arcs[1] == next_start)
]

print("Direct connecting arcs:", matching.index.tolist())

reverse = arcs[
    (arcs[0] == next_start) &
    (arcs[1] == previous_end)
]

print("Reverse connecting arcs:", reverse.index.tolist())

print("\n========== SHORTEST PATH CHECK ==========")

try:
    path = nx.shortest_path(
        G,
        source=previous_end,
        target=next_start
    )

    print("Shortest directed path:", path)
    print("Number of edges:", len(path) - 1)

    for u, v in zip(path[:-1], path[1:]):
        print(f"{u} -> {v}: arc {G[u][v]['arc_id']}")

except nx.NetworkXNoPath:
    print("No directed path exists.")

print("\n========== NEARBY ROUTE ARCS ==========")

for position in range(160, 169):
    arc_id = route[position]
    row = arcs.iloc[arc_id]

    print(
        f"Position {position}: "
        f"Arc {arc_id}: "
        f"{int(row.iloc[0])} -> {int(row.iloc[1])}"
    )

print("\nInvestigation complete.")
