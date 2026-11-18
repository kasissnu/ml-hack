import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pathsnap.config import ROOT, partition_dir
from pathsnap.geo import LocalProjection
from pathsnap.graph import RoadGraph
from pathsnap.hmm import build_lattice, emission_logp, stitch_route, viterbi
from pathsnap.io import list_trips
from pathsnap.routing import Router


def _corner_graph():
    """Two roads leaving node 1: 0->1->2 goes east, 1->3 goes north (plus reverse twins)."""
    xy = np.array([[0, 0], [100, 0], [200, 0], [100, 100]], float)
    pr = LocalProjection(0.0, 0.0)
    arcs = np.array([[0, 1], [1, 2], [1, 3], [1, 0], [2, 1], [3, 1]])
    return RoadGraph(pr.to_lonlat(xy), arcs, pr)


def test_router_paths():
    g = _corner_graph(); r = Router(g)
    assert r.distance(0, 2, 500) == pytest.approx(200, rel=1e-3)
    assert r.arc_path(0, 3, 500) == [0, 2]
    assert r.arc_path(0, 2, 50) is None


def test_viterbi_prefers_connected_route():
    """A noisy middle point slightly favours a disconnected-looking choice; the HMM must keep the
    physically consistent east-bound route."""
    g = _corner_graph(); r = Router(g)
    P = np.array([[50, 2], [100, 2], [150, 2]], float)
    t = np.array([0, 5, 10], float)
    # candidates per point: east-bound arc vs the reverse twin
    obs = np.array([0, 0, 1, 1, 2, 2])
    arc = np.array([0, 3, 0, 3, 1, 4])
    u = np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.5])
    lat = build_lattice(g, r, P, t, obs, arc, u)
    p = np.array([0.9, 0.1, 0.4, 0.6, 0.9, 0.1])     # classifier wrongly prefers twin at t=1
    ch, breaks = viterbi(lat, emission_logp(p, lat.starts), dict(beta=5, uturn_m=50, gap_s=60, w_e=1))
    assert list(arc[ch]) == [0, 0, 1] and breaks == 0
    route, gaps = stitch_route(arc[ch], g, r)
    assert route == [0, 1] and gaps == 0


def test_viterbi_breaks_on_gap():
    g = _corner_graph(); r = Router(g)
    P = np.array([[50, 2], [150, 2]], float)
    lat = build_lattice(g, r, P, np.array([0, 500.0]), np.array([0, 1]), np.array([0, 1]), np.array([0.5, 0.5]))
    ch, breaks = viterbi(lat, emission_logp(np.array([1.0, 1.0]), lat.starts), dict(gap_s=60))
    assert breaks == 1 and list(ch) == [0, 1]


HAS_ART = (ROOT / "artifacts" / "model.joblib").exists()
ST = partition_dir("student_test")


@pytest.mark.skipif(not (HAS_ART and ST.exists()), reason="needs exported artifacts and student_test data")
def test_api_end_to_end():
    os.environ["PATHSNAP_TRIPS_DIR"] = str(ST)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    import api.main as api_main
    api_main.TRIPS_DIR = Path(ST)
    with TestClient(api_main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        trips = [t["trip_id"] for t in client.get("/trips").json()["trips"]]
        assert trips == list_trips(ST)
        for tid in trips[:2]:
            r = client.post("/predict", json={"trip_id": tid})
            assert r.status_code == 200
            body = r.json()
            # ground truth must never leave the server: only aggregate metrics are returned
            assert set(body) == {"trip_id", "n_points", "gps", "snapped", "point_arc_ids", "route_arc_ids",
                                 "route_polylines", "chain_breaks", "unbridged_gaps", "metrics", "timings_s", "model"}
            assert len(body["point_arc_ids"]) == body["n_points"]
            assert 0 <= body["metrics"]["top1_accuracy"] <= 1
        assert client.post("/predict", json={"trip_id": "does-not-exist"}).status_code == 404
