
from pathlib import Path

import pandas as pd
import folium


# ==================================================
# PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"

TRIP_IDS = ["00000033", "00000054"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

def read_data_file(file_path):
    """Read comma-separated, tab-separated or whitespace-separated data."""

    return pd.read_csv(
        file_path,
        sep=r"[\s,]+",
        engine="python",
        header=None
    )


def load_trip_data(trip_id):

    trip_dir = DATA_DIR / trip_id

    gps = read_data_file(trip_dir / f"{trip_id}.track")
    nodes = read_data_file(trip_dir / f"{trip_id}.nodes")
    arcs = read_data_file(trip_dir / f"{trip_id}.arcs")
    route = read_data_file(trip_dir / f"{trip_id}.route")

    gps = gps.iloc[:, :3].copy()
    gps.columns = ["lon", "lat", "timestamp"]

    nodes = nodes.iloc[:, :2].copy()
    nodes.columns = ["lon", "lat"]

    arcs = arcs.iloc[:, :2].copy()
    arcs.columns = ["from_node", "to_node"]

    route = route.iloc[:, 0].copy()

    gps = gps.apply(pd.to_numeric, errors="coerce").dropna()
    nodes = nodes.apply(pd.to_numeric, errors="coerce").dropna()
    arcs = arcs.apply(pd.to_numeric, errors="coerce").dropna()
    route = pd.to_numeric(route, errors="coerce").dropna()

    return gps, nodes, arcs, route


# ==================================================
# RECONSTRUCT REFERENCE ROUTE
# ==================================================

def reconstruct_route(nodes, arcs, route):

    route_coordinates = []

    for arc_id in route:

        arc_index = int(arc_id)

        if not (0 <= arc_index < len(arcs)):
            continue

        arc = arcs.iloc[arc_index]

        from_node = int(arc["from_node"])
        to_node = int(arc["to_node"])

        if not (
            0 <= from_node < len(nodes)
            and 0 <= to_node < len(nodes)
        ):
            continue

        start = nodes.iloc[from_node]
        end = nodes.iloc[to_node]

        if not route_coordinates:
            route_coordinates.append(
                [start["lat"], start["lon"]]
            )

        route_coordinates.append(
            [end["lat"], end["lon"]]
        )

    return route_coordinates


# ==================================================
# CREATE VISUALIZATION
# ==================================================

def create_alignment_map(trip_id):

    print(f"\nProcessing trip: {trip_id}")

    gps, nodes, arcs, route = load_trip_data(trip_id)

    print(f"GPS observations: {len(gps)}")
    print(f"Network nodes: {len(nodes)}")
    print(f"Network arcs: {len(arcs)}")
    print(f"Reference route arcs: {len(route)}")

    if gps.empty:
        print("No GPS observations found. Skipping.")
        return

    # GPS coordinates
    gps_coordinates = gps[
        ["lat", "lon"]
    ].values.tolist()

    # Reconstruct reference route
    route_coordinates = reconstruct_route(
        nodes,
        arcs,
        route
    )

    # Center map around GPS trajectory
    center_lat = gps["lat"].median()
    center_lon = gps["lon"].median()

    # No external tile provider
    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None
    )

    # ==================================================
    # GPS TRAJECTORY - BLUE
    # ==================================================

    gps_layer = folium.FeatureGroup(
        name="GPS Trajectory (Blue)",
        show=True
    )

    gps_sample = gps_coordinates[::3]

    if gps_sample[-1] != gps_coordinates[-1]:
        gps_sample.append(gps_coordinates[-1])

    folium.PolyLine(
        locations=gps_sample,
        color="blue",
        weight=4,
        opacity=0.9,
        tooltip="GPS Trajectory"
    ).add_to(gps_layer)

    folium.Marker(
        location=gps_coordinates[0],
        tooltip="GPS Start",
        icon=folium.Icon(color="green")
    ).add_to(gps_layer)

    folium.Marker(
        location=gps_coordinates[-1],
        tooltip="GPS End",
        icon=folium.Icon(color="red")
    ).add_to(gps_layer)

    gps_layer.add_to(map_object)

    # ==================================================
    # REFERENCE ROUTE - RED
    # ==================================================

    route_layer = folium.FeatureGroup(
        name="Reference Route (Red)",
        show=True
    )

    if route_coordinates:

        folium.PolyLine(
            locations=route_coordinates,
            color="red",
            weight=4,
            opacity=0.9,
            tooltip="Reference Route"
        ).add_to(route_layer)

        route_layer.add_to(map_object)

    else:
        print("Warning: Could not reconstruct reference route.")

    # ==================================================
    # MAP CONTROLS
    # ==================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(map_object)

    # Fit map to GPS and route coordinates
    all_coordinates = gps_coordinates + route_coordinates

    if all_coordinates:
        map_object.fit_bounds(all_coordinates)

    # ==================================================
    # SAVE HTML
    # ==================================================

    output_file = (
        OUTPUT_DIR / f"{trip_id}_route_alignment.html"
    )

    map_object.save(str(output_file))

    print(f"Map saved: {output_file}")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    for trip_id in TRIP_IDS:

        try:
            create_alignment_map(trip_id)

        except Exception as error:
            print(f"Error processing {trip_id}: {error}")

    print("\nAll maps generated successfully!")
