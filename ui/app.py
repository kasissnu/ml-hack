"""PathSnap evaluator UI (runs on Laptop 2).

Start:
    streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
Set the API address in the sidebar (or env PATHSNAP_API, e.g. http://192.168.1.20:8000).

The UI never reads data files: the trip list, the prediction and the metrics all come from the API.
"""
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

COL = {"gps": "#1f4fd1", "route": "#d62728", "snapped": "#f28e1c"}

st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API address", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def get_json(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


health = None
try:
    health = get_json(f"{api}/health")
    st.sidebar.success(f"Connected - {health['model']}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}\n\n{e}")

trips = []
if health:
    try:
        trips = get_json(f"{api}/trips")["trips"]
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not load trips: {e}")

st.title("PathSnap - map-matching & route reconstruction")
st.caption("Select an evaluation trip. The API runs candidate generation, the XGBoost classifier and "
           "HMM/Viterbi decoding, then scores the route against its server-side reference.")

if not trips:
    st.info("No trips available yet. Check the API address and that the API has trips loaded.")
    st.stop()

labels = {f"{t['trip_id']}  ({t['n_points']} GPS points)": t["trip_id"] for t in trips}
# optional deep link: ?trip=<id>&run=1 preselects a trip and runs it once
qp_trip = st.query_params.get("trip")
default_idx = next((i for i, v in enumerate(labels.values()) if v == qp_trip), 0)
choice = st.sidebar.radio(f"Evaluation trips ({len(trips)})", list(labels), index=default_idx)
show_snapped = st.sidebar.checkbox("Show snapped positions", value=False)
run = st.sidebar.button("Reconstruct route", type="primary", use_container_width=True)
if qp_trip and st.query_params.get("run") == "1" and not st.session_state.get("_deeplink_done"):
    st.session_state["_deeplink_done"] = True
    run = True

if run:
    with st.spinner(f"Running PathSnap on trip {labels[choice]} ..."):
        try:
            r = requests.post(f"{api}/predict", json={"trip_id": labels[choice]}, timeout=300)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if not res:
    st.info("Choose a trip in the sidebar and press **Reconstruct route**.")
    st.stop()

st.subheader(f"Trip {res['trip_id']}")
m = res.get("metrics")
c = st.columns(5)
if m:
    c[0].metric("Top-1 (HMM)", f"{100 * m['top1_accuracy']:.1f}%",
                f"{100 * (m['top1_accuracy'] - m['classifier_only_top1']):+.1f} pp", help="vs classifier only")
    c[1].metric("Top-1 (clf)", f"{100 * m['classifier_only_top1']:.1f}%", help="classifier-only Top-1")
    c[2].metric("Continuity", f"{100 * m['plausible_continuity']:.1f}%", help="plausible route continuity")
    c[3].metric("Route F1", f"{100 * m['route_f1']:.1f}%")
else:
    c[0].metric("Top-1", "n/a")
c[4].metric("API time", f"{res['timings_s']['total_s'] + res['timings_s']['evaluation_s']:.1f}s",
            help="processing time on the API laptop")

gps = res["gps"]
lats = [p[0] for p in gps]; lons = [p[1] for p in gps]
fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, control_scale=True)
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
fg_gps = folium.FeatureGroup(name=f"GPS observations ({len(gps)})")
step = max(1, len(gps) // 4000)
for i in range(0, len(gps), step):
    folium.CircleMarker(gps[i], radius=2.5, color=COL["gps"], fill=True, fill_opacity=0.9, weight=1,
                        tooltip=f"GPS #{i}").add_to(fg_gps)
fg_gps.add_to(fmap)
fg_route = folium.FeatureGroup(name=f"Predicted route ({len(res['route_arc_ids'])} arcs)")
for line in res["route_polylines"]:
    folium.PolyLine(line, color=COL["route"], weight=5, opacity=0.8).add_to(fg_route)
fg_route.add_to(fmap)
if show_snapped:
    fg_s = folium.FeatureGroup(name="Snapped positions")
    for i in range(0, len(res["snapped"]), step):
        if res["snapped"][i]:
            folium.CircleMarker(res["snapped"][i], radius=2, color=COL["snapped"], fill=True).add_to(fg_s)
    fg_s.add_to(fmap)
folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
legend = ('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px;'
          'border:1px solid #888;border-radius:4px;font-size:12px"><b>Legend</b>'
          f'<div><span style="color:{COL["gps"]}">●</span> GPS observations</div>'
          f'<div><span style="color:{COL["route"]}">━</span> Predicted route (HMM/Viterbi)</div>'
          + (f'<div><span style="color:{COL["snapped"]}">●</span> Snapped positions</div>' if show_snapped else "")
          + "</div>")
fmap.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(fmap)
components.html(fmap.get_root().render(), height=620)

left, right = st.columns(2)
with left:
    st.markdown("**Evaluation (computed by the API)**")
    if m:
        st.table({k: [round(v, 4) if isinstance(v, float) else v] for k, v in m.items()})
with right:
    st.markdown("**Pipeline timings (s)**")
    st.table({k: [v] for k, v in res["timings_s"].items()})
    st.caption(f"Chain breaks: {res['chain_breaks']}  |  unbridged route gaps: {res['unbridged_gaps']}  |  "
               f"HMM params: {res['model']['hmm_params']}")
with st.expander("Raw API response (truncated)"):
    st.json({k: (v[:20] if isinstance(v, list) else v) for k, v in res.items()})
