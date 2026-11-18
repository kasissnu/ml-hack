"""Folium map builders shared by EDA, evaluation maps and the UI."""
from __future__ import annotations

import folium
import numpy as np

COLORS = {
    "gps": "#1f4fd1",        # blue
    "network": "#8c8c8c",    # grey
    "reference": "#1a9e3a",  # green
    "classifier": "#f28e1c", # orange
    "hmm": "#d62728",        # red
    "wrong": "#9b1fb5",      # purple
}


def base_map(lonlat: np.ndarray, tiles: str = "OpenStreetMap") -> folium.Map:
    m = folium.Map(location=[float(np.mean(lonlat[:, 1])), float(np.mean(lonlat[:, 0]))],
                   zoom_start=15, tiles=tiles, control_scale=True)
    m.fit_bounds([[float(lonlat[:, 1].min()), float(lonlat[:, 0].min())],
                  [float(lonlat[:, 1].max()), float(lonlat[:, 0].max())]])
    return m


def add_gps(m, track: np.ndarray, name="GPS observations", color=COLORS["gps"], every: int = 1, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for i in range(0, len(track), every):
        lon, lat = track[i, 0], track[i, 1]
        tip = f"#{i}  t={track[i, 2]:.0f}s" if track.shape[1] > 2 else f"#{i}"
        folium.CircleMarker([lat, lon], radius=2.5, color=color, fill=True, fill_opacity=0.9,
                            weight=1, tooltip=tip).add_to(fg)
    folium.Marker([track[0, 1], track[0, 0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker([track[-1, 1], track[-1, 0]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)
    return fg


def arcs_to_lines(nodes_lonlat: np.ndarray, arcs: np.ndarray, arc_ids) -> list:
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    a = nodes_lonlat[arcs[arc_ids, 0]]
    b = nodes_lonlat[arcs[arc_ids, 1]]
    return [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]


def add_network(m, nodes_lonlat, arcs, track, margin_deg: float = 0.002, name="Road network", show=True):
    """Draw only arcs inside the trip bounding box (+margin) so large networks stay light."""
    lo = track[:, :2].min(0) - margin_deg
    hi = track[:, :2].max(0) + margin_deg
    a = nodes_lonlat[arcs[:, 0]]
    b = nodes_lonlat[arcs[:, 1]]
    inside = ((a >= lo) & (a <= hi)).all(1) | ((b >= lo) & (b <= hi)).all(1)
    ids = np.flatnonzero(inside)
    # two-way roads appear twice; draw one copy per undirected pair
    key = np.sort(arcs[ids], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    ids = ids[first]
    fg = folium.FeatureGroup(name=f"{name} ({len(ids)} segments)", show=show)
    folium.PolyLine(arcs_to_lines(nodes_lonlat, arcs, ids), color=COLORS["network"],
                    weight=2, opacity=0.6).add_to(fg)
    fg.add_to(m)
    return fg


def add_route(m, nodes_lonlat, arcs, route, name, color, weight=5, opacity=0.8, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    if len(route):
        folium.PolyLine(arcs_to_lines(nodes_lonlat, arcs, route), color=color,
                        weight=weight, opacity=opacity, tooltip=name).add_to(fg)
    fg.add_to(m)
    return fg


def add_legend(m, items: list[tuple[str, str]], title="Legend"):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in items)
    html = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #888;border-radius:4px;font-size:12px">'
            f'<b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def finish(m):
    folium.LayerControl(collapsed=False).add_to(m)
    return m
