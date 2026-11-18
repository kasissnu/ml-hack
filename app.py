import os
import requests
import streamlit as st
import streamlit.components.v1 as components
from map_utils import build_trip_map

st.set_page_config(page_title="PathSnap | Route Reconstruction", page_icon="🗺️", layout="wide")

API_BASE_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
TRIPS = ["00000000", "00000007", "00000009", "00000040", "00000052",
         "00000056", "00000065", "00000076", "00000077", "00000091"]
DATA_ROOT = os.getenv("PATHSNAP_DATA_DIR", "data/student_test")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #f4f7fb; padding: 14px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🗺️ PathSnap")
st.caption("Distributed Map-Matching & Route Reconstruction")
with st.sidebar:
    st.header("Evaluation controls")
    trip_id = st.selectbox("Select a student-test trip", TRIPS)
    st.caption("The UI sends only the selected trip ID to the model laptop.")
    st.divider()
    st.write("**Backend**")
    st.code(API_BASE_URL)
    if st.button("Check API connection", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            st.success(f"API reachable · HTTP {r.status_code}")
            st.json(r.json())
        except requests.RequestException as e:
            st.error(f"Cannot reach API: {e}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Selected trip")
    st.metric("Trip ID", trip_id)
    st.write("Choose **Run map matching** to send this trip to Laptop 1.")
with right:
    st.subheader("System status")
    st.info("The prediction API is called only when the evaluator runs a trip.")

if "result" not in st.session_state:
    st.session_state.result = None
if st.button("▶ Run map matching", type="primary", use_container_width=True):
    try:
        with st.spinner("Sending trip to the model server and waiting for route reconstruction..."):
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"trip_id": trip_id},
                timeout=180
            )
            response.raise_for_status()
            st.session_state.result = {"trip_id": trip_id, "payload": response.json()}
    except requests.RequestException as e:
        st.session_state.result = None
        st.error(f"Prediction request failed: {e}")
        st.info("Check that Laptop 1 is running FastAPI, that the endpoint is /predict, and that both laptops are on the same network.")

result = st.session_state.result
if result and result["trip_id"] == trip_id:
    payload = result["payload"]
    st.divider()
    st.subheader(f"Results · Trip {trip_id}")
    # Accommodate common metric key names; confirm exact response schema with Laptop 1.
    metrics = payload.get("metrics", {})
    cols = st.columns(4)
    metric_candidates = [
        ("Top-1 Accuracy", ["top1_accuracy", "accuracy"]),
        ("Macro F1-score", ["macro_f1", "macro_f1_score"]),
        ("Log Loss", ["log_loss"]),
        ("Inference Time", ["inference_time", "inference_time_ms"]),
    ]
    for col, (label, keys) in zip(cols, metric_candidates):
        value = next((metrics[k] for k in keys if k in metrics), None)
        if value is None:
            value = next((payload[k] for k in keys if k in payload), None)
        col.metric(label, "—" if value is None else str(value))

    st.subheader("Predicted route map")
    predicted = payload.get("predicted_route", payload.get("predicted_arc_ids", []))
    if isinstance(predicted, dict):
        predicted = predicted.get("arc_ids", [])
    trip_dir = os.path.join(DATA_ROOT, trip_id)
    map_obj = build_trip_map(trip_dir, predicted_arc_ids=predicted)
    if map_obj is not None:
        components.html(map_obj.get_root().render(), height=650, scrolling=True)
    else:
        st.warning("Could not build the map. Check the local trip data directory and file names.")

    with st.expander("Raw API response (for debugging)"):
        st.json(payload)
else:
    st.divider()
    st.subheader("Trip visualisation")
    trip_dir = os.path.join(DATA_ROOT, trip_id)
    map_obj = build_trip_map(trip_dir)
    if map_obj is not None:
        components.html(map_obj.get_root().render(), height=600, scrolling=True)
    else:
        st.warning(f"Trip files not found in {trip_dir}. Copy the student_test folder into data/.")
