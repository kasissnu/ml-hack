import shutil

import numpy as np
import pytest

from pathsnap.candidates import ArcIndex
from pathsnap.config import partition_dir
from pathsnap.features import build_features, prepare_trip
from pathsnap.geo import LocalProjection, angle_diff_deg, haversine, project_point_segment
from pathsnap.graph import RoadGraph
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import derive_labels

TRAIN = partition_dir("train")
HAS_DATA = TRAIN.exists() and len(list_trips(TRAIN)) > 0
needs_data = pytest.mark.skipif(not HAS_DATA, reason="dataset not present in data/")


def test_projection_matches_haversine():
    rng = np.random.default_rng(0)
    for lat0 in (0.0, 45.0, 56.0):
        ll = np.column_stack([10 + rng.uniform(-0.1, 0.1, 200), lat0 + rng.uniform(-0.1, 0.1, 200)])
        pr = LocalProjection.for_points(ll)
        xy = pr.to_xy(ll)
        d_xy = np.hypot(*(xy[1:] - xy[:-1]).T)
        d_h = haversine(ll[:-1, 0], ll[:-1, 1], ll[1:, 0], ll[1:, 1])
        assert np.max(np.abs(d_xy - d_h) / d_h) < 0.005
        assert np.allclose(pr.to_lonlat(xy), ll)


def test_point_segment():
    A = np.array([[0.0, 0.0]]); B = np.array([[10.0, 0.0]])
    d, u, _ = project_point_segment(np.array([[5.0, 3.0]]), A, B)
    assert d[0] == pytest.approx(3.0) and u[0] == pytest.approx(0.5)
    d, u, _ = project_point_segment(np.array([[-4.0, 3.0]]), A, B)
    assert d[0] == pytest.approx(5.0) and u[0] == 0.0


def test_angle_diff():
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(0, 180) == pytest.approx(180)


def _toy_graph():
    # a straight two-way road 0-1-2 along the x axis, 100 m segments
    nodes_xy = np.array([[0, 0], [100, 0], [200, 0]], float)
    pr = LocalProjection(0.0, 0.0)
    nodes_ll = pr.to_lonlat(nodes_xy)
    arcs = np.array([[0, 1], [1, 2], [1, 0], [2, 1]])
    return RoadGraph(nodes_ll, arcs, pr), pr


def test_graph_twins_and_paths():
    g, _ = _toy_graph()
    assert list(g.reverse_twin) == [2, 3, 0, 1]
    assert g.shortest_arc_path(0, 2) == [0, 1]
    assert g.node_distances(0, np.array([2]), limit=500)[0] == pytest.approx(200, rel=1e-3)
    assert np.isinf(g.node_distances(0, np.array([2]), limit=50)[0])


def test_labels_monotone_and_direction():
    g, _ = _toy_graph()
    x = np.arange(5, 200, 10.0)
    P = np.column_stack([x, np.full_like(x, 3.0)])       # driving east, 3 m off the road
    t = np.arange(len(x), dtype=float)
    labels, r, _ = derive_labels(P, t, g, np.array([0, 1]))
    assert np.all(np.diff(r) >= 0)
    assert set(labels[x < 95]) == {0} and set(labels[x > 105]) == {1}


@needs_data
def test_candidates_match_brute_force():
    tid = list_trips(TRAIN)[0]
    trip = load_trip(TRAIN / tid)
    pr = LocalProjection.for_points(trip.track[:, :2])
    g = RoadGraph(trip.nodes, trip.arcs, pr)
    P = pr.to_xy(trip.track[:, :2])
    c = ArcIndex(g, P, 200).query(P, 50, 8)
    for i in np.random.default_rng(1).choice(len(P), 40, replace=False):
        d, _, _ = project_point_segment(P[i][None], g.A, g.B)
        bf = np.sort(d[d <= 50])[:8]
        mine = np.sort(c.dist[c.obs == i])[: len(bf)]
        assert np.allclose(bf, mine)


@needs_data
def test_features_do_not_depend_on_route(tmp_path):
    """Leakage check: removing the .route file must not change any feature."""
    tid = list_trips(TRAIN)[0]
    src = TRAIN / tid
    dst = tmp_path / tid
    shutil.copytree(src, dst)
    f1 = build_features(prepare_trip(load_trip(dst), 50, 8))
    (dst / f"{tid}.route").unlink()
    assert load_route(dst) is None
    f2 = build_features(prepare_trip(load_trip(dst), 50, 8))
    assert f1.equals(f2)
