import os
import math
import pandas as pd
import folium
from folium.plugins import Fullscreen

def _read_table(path, names):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")

def load_trip(trip_dir):
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    track = _read_table(os.path.join(trip_dir, f"{trip_id}.track"),
                        ["longitude", "latitude", "timestamp"])
    nodes = _read_table(os.path.join(trip_dir, f"{trip_id}.nodes"),
                        ["longitude", "latitude"])
    arcs = _read_table(os.path.join(trip_dir, f"{trip_id}.arcs"),
                       ["from_node_id", "to_node_id"])
    if track is None or nodes is None or arcs is None or track.empty or nodes.empty:
        return None, None, None
    for df in (track, nodes):
        df.dropna(subset=["longitude", "latitude"], inplace=True)
    return track, nodes, arcs

def build_trip_map(trip_dir, predicted_arc_ids=None, show_network=True, max_network_edges=15000):
    track, nodes, arcs = load_trip(trip_dir)
    if track is None:
        return None

    center = [track["latitude"].mean(), track["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="CartoDB positron")
    Fullscreen(position="topright").add_to(m)

    # Road-network layer. Arc row index is the arc_id per the supplied dataset guide.
    if show_network and arcs is not None:
        network = folium.FeatureGroup(name="Road network", show=False)
        limit = min(len(arcs), max_network_edges)
        for arc_id in range(limit):
            row = arcs.iloc[arc_id]
            a, b = int(row["from_node_id"]), int(row["to_node_id"])
            if 0 <= a < len(nodes) and 0 <= b < len(nodes):
                p1 = [float(nodes.iloc[a]["latitude"]), float(nodes.iloc[a]["longitude"])]
                p2 = [float(nodes.iloc[b]["latitude"]), float(nodes.iloc[b]["longitude"])]
                folium.PolyLine([p1, p2], color="#9aa4b2", weight=1, opacity=0.45).add_to(network)
        network.add_to(m)

    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    coords = track[["latitude", "longitude"]].astype(float).values.tolist()
    folium.PolyLine(coords, color="#2474e8", weight=3, opacity=0.75, tooltip="GPS trajectory").add_to(gps_layer)
    for i, point in enumerate(coords):
        folium.CircleMarker(point, radius=3, color="#2474e8", fill=True,
                            fill_opacity=0.8, tooltip=f"GPS point {i+1}").add_to(gps_layer)
    if coords:
        folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(gps_layer)
        folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(gps_layer)
    gps_layer.add_to(m)

    if predicted_arc_ids and arcs is not None:
        predicted_layer = folium.FeatureGroup(name="Predicted route", show=True)
        route_points = []
        for raw_id in predicted_arc_ids:
            try:
                arc_id = int(raw_id)
                if 0 <= arc_id < len(arcs):
                    row = arcs.iloc[arc_id]
                    a, b = int(row["from_node_id"]), int(row["to_node_id"])
                    if 0 <= a < len(nodes) and 0 <= b < len(nodes):
                        p1 = [float(nodes.iloc[a]["latitude"]), float(nodes.iloc[a]["longitude"])]
                        p2 = [float(nodes.iloc[b]["latitude"]), float(nodes.iloc[b]["longitude"])]
                        folium.PolyLine([p1, p2], color="#e53935", weight=5, opacity=0.9,
                                        tooltip=f"Predicted arc {arc_id}").add_to(predicted_layer)
                        route_points.extend([p1, p2])
            except (TypeError, ValueError):
                continue
        predicted_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track["latitude"].min(), track["longitude"].min()],
                  [track["latitude"].max(), track["longitude"].max()]])
    return m
