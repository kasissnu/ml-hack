
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "train"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def read_file(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )


def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def inspect_trip(folder):

    trip_id = folder.name

    nodes = read_file(folder / f"{trip_id}.nodes")
    arcs = read_file(folder / f"{trip_id}.arcs")
    route = read_file(folder / f"{trip_id}.route").iloc[:, 0]

    route = pd.to_numeric(
        route, errors="coerce"
    ).dropna().astype(int).tolist()

    edges = []

    for position, arc_id in enumerate(route):

        if not 0 <= arc_id < len(arcs):
            continue

        start = int(arcs.iloc[arc_id, 0])
        end = int(arcs.iloc[arc_id, 1])

        if not (
            0 <= start < len(nodes)
            and 0 <= end < len(nodes)
        ):
            continue

        edges.append({
            "position": position,
            "arc_id": arc_id,
            "start": start,
            "end": end
        })

    breaks = []

    for i in range(len(edges) - 1):

        current = edges[i]
        following = edges[i + 1]

        if current["end"] != following["start"]:

            p = nodes.iloc[current["end"]]
            q = nodes.iloc[following["start"]]

            gap = haversine(
                float(p.iloc[0]),
                float(p.iloc[1]),
                float(q.iloc[0]),
                float(q.iloc[1])
            )

            breaks.append({
                "trip_id": trip_id,
                "route_position": current["position"],
                "current_arc": current["arc_id"],
                "current_end_node": current["end"],
                "next_arc": following["arc_id"],
                "next_start_node": following["start"],
                "geographic_gap_m": round(float(gap), 2)
            })

    return breaks


def main():

    all_breaks = []

    folders = sorted(
        folder for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    for folder in folders:

        breaks = inspect_trip(folder)
        all_breaks.extend(breaks)

        if breaks:
            print(
                f"{folder.name}: "
                f"{len(breaks)} disconnected transitions"
            )

    df = pd.DataFrame(all_breaks)

    output = REPORT_DIR / "all_route_breaks.csv"
    df.to_csv(output, index=False)

    print("\n========== SUMMARY ==========")
    print("Trips with breaks:", df["trip_id"].nunique() if not df.empty else 0)
    print("Total disconnected transitions:", len(df))

    if not df.empty:
        print("\nBreak details:")
        print(df.to_string(index=False))

    print("\nSaved report:", output)


if __name__ == "__main__":
    main()
