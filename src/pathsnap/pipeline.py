"""End-to-end inference used by the API (and verified against the offline experiments).

predict_trip(trip_dir, artifacts)  -> prediction; never reads the .route file
evaluate_prediction(trip_dir, pred) -> metrics; reads the .route file (server-side only), AFTER
                                       prediction, to score it. Its output contains numbers only.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import FEATURE_COLS, build_features, prepare_trip
from .geo import project_point_segment
from .hmm import build_lattice, emission_logp, stitch_route, transition_stats, viterbi
from .io import load_route, load_trip
from .labels import derive_labels
from .metrics import argmax_per_group, route_prf
from .routing import Router


@dataclass
class Artifacts:
    model: object
    feature_cols: list
    candidate_params: dict
    hmm_params: dict
    model_card: dict

    @classmethod
    def load(cls, art_dir: Path) -> "Artifacts":
        art_dir = Path(art_dir)
        feature_cols = json.load(open(art_dir / "feature_cols.json"))
        if feature_cols != FEATURE_COLS:
            raise RuntimeError("feature_cols.json does not match the code's FEATURE_COLS")
        return cls(model=joblib.load(art_dir / "model.joblib"), feature_cols=feature_cols,
                   candidate_params=json.load(open(art_dir / "candidate_params.json")),
                   hmm_params=json.load(open(art_dir / "hmm_params.json")),
                   model_card=json.load(open(art_dir / "model_card.json")))


def predict_trip(trip_dir: Path, art: Artifacts) -> dict:
    timings = {}
    t0 = time.perf_counter()
    trip = load_trip(trip_dir)
    timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cp = art.candidate_params
    td = prepare_trip(trip, cp["radius_m"], cp["k"], cp.get("densify_step_m", 10.0))
    timings["candidates_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = build_features(td)
    timings["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = art.model.predict_proba(feats[art.feature_cols].to_numpy(np.float32))[:, 1]
    timings["classifier_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    router = Router(td.g)
    lat = build_lattice(td.g, router, td.P, td.t, td.cands.obs, td.cands.arc, td.cands.u)
    em = emission_logp(p, lat.starts)
    choice, breaks = viterbi(lat, em, art.hmm_params)
    timings["hmm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    point_arcs = np.where(choice >= 0, lat.arc[np.maximum(choice, 0)], -1)
    route, unbridged = stitch_route(point_arcs, td.g, router)
    timings["stitch_s"] = time.perf_counter() - t0

    sel = argmax_per_group(td.cands.obs, p)
    clf_arcs = np.full(len(point_arcs), -1, dtype=np.int64)
    clf_arcs[td.cands.obs[sel]] = td.cands.arc[sel]
    # snapped position of each GPS point on its predicted arc
    ok = point_arcs >= 0
    snapped = np.full((len(point_arcs), 2), np.nan)
    if ok.any():
        _, _, Q = project_point_segment(td.P[ok], td.g.A[point_arcs[ok]], td.g.B[point_arcs[ok]])
        snapped[ok] = td.proj.to_lonlat(Q)
    timings["total_s"] = sum(timings.values())
    return dict(trip_id=trip.trip_id, n_points=trip.n_points, point_arcs=point_arcs, classifier_arcs=clf_arcs,
                route_arcs=route, unbridged_gaps=unbridged, chain_breaks=breaks, snapped_lonlat=snapped,
                choice=choice, timings=timings, _td=td, _lat=lat, _router=router)


def evaluate_prediction(trip_dir: Path, pred: dict, max_match_dist_m: float = 40.0) -> dict | None:
    """Server-side scoring against the reference route. Returns None if no route is available."""
    route = load_route(trip_dir)
    if route is None:
        return None
    td, lat = pred["_td"], pred["_lat"]
    labels, _, _ = derive_labels(td.P, td.t, td.g, route, max_match_dist_m)
    m = labels >= 0
    pa, ca = pred["point_arcs"], pred["classifier_arcs"]
    n_dir, n_pl, n_ch = transition_stats(lat, pred["choice"], td.g)
    prf = route_prf(pred["route_arcs"], route, td.g.length)
    return dict(n_points=int(len(labels)), n_matched=int(m.sum()),
                top1_accuracy=float((pa[m] == labels[m]).mean()) if m.any() else None,
                strict_accuracy=float((pa == labels).sum() / len(labels)),
                classifier_only_top1=float((ca[m] == labels[m]).mean()) if m.any() else None,
                direct_continuity=n_dir / n_ch if n_ch else 1.0,
                plausible_continuity=n_pl / n_ch if n_ch else 1.0,
                **{k: float(v) for k, v in prf.items()})
