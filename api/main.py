"""PathSnap inference API (runs on Laptop 1).

Start:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Environment:
    PATHSNAP_TRIPS_DIR   folder that holds the trips offered to the evaluator (default: data/eval)
    PATHSNAP_ARTIFACTS   folder with the exported model artifacts      (default: artifacts)

Endpoints
    GET  /health            service status and model information
    GET  /trips             ids of the available trips (no ground truth)
    POST /predict           {"trip_id": "..."} -> predicted route + server-side evaluation metrics

Ground truth (.route) is read only inside pathsnap.pipeline.evaluate_prediction, AFTER the
prediction has been made, and only aggregate metrics are returned - never the reference route.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pathsnap.io import list_trips  # noqa: E402
from pathsnap.pipeline import Artifacts, evaluate_prediction, predict_trip  # noqa: E402

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "eval")).resolve()
ART_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts")).resolve()

(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / "api.log", encoding="utf-8")):
    h.setFormatter(_fmt)
    log.addHandler(h)
log.propagate = False

ART: Artifacts | None = None


class PredictRequest(BaseModel):
    trip_id: str


def startup():
    global ART
    ART = Artifacts.load(ART_DIR)
    trips = list_trips(TRIPS_DIR)
    log.info("=" * 70)
    log.info("PathSnap API started")
    log.info(f"model      : {ART.model_card.get('classifier')}  (artifacts: {ART_DIR})")
    log.info(f"candidates : radius={ART.candidate_params['radius_m']} m, K={ART.candidate_params['k']}")
    log.info(f"hmm params : {ART.hmm_params}")
    log.info(f"trips dir  : {TRIPS_DIR}  ({len(trips)} trips: {', '.join(trips)})")
    if not trips:
        log.warning("no trips found - set PATHSNAP_TRIPS_DIR or copy the evaluation trips into data/eval")
    log.info("=" * 70)


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} "
             f"({1000 * (time.perf_counter() - t0):.0f} ms)")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": ART.model_card.get("classifier"), "trained_on": ART.model_card.get("trained_on"),
            "candidate_params": ART.candidate_params, "hmm_params": ART.hmm_params,
            "n_trips": len(list_trips(TRIPS_DIR))}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(TRIPS_DIR):
        p = TRIPS_DIR / tid / f"{tid}.track"
        with open(p, "rb") as f:
            n = sum(1 for line in f if line.strip())
        out.append({"trip_id": tid, "n_points": n})
    return {"trips": out}


def _r(x, nd=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _route_polylines(route, nodes, arcs, g_conn):
    """Join consecutive connected arcs into polylines of [lat, lon] points."""
    lines, cur = [], []
    for i, a in enumerate(route):
        p, q = nodes[arcs[a, 0]], nodes[arcs[a, 1]]
        if cur and arcs[route[i - 1], 1] == arcs[a, 0]:
            cur.append([_r(q[1]), _r(q[0])])
        else:
            if cur:
                lines.append(cur)
            cur = [[_r(p[1]), _r(p[0])], [_r(q[1]), _r(q[0])]]
    if cur:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if tid not in list_trips(TRIPS_DIR):
        log.warning(f"predict: unknown trip_id '{tid}'")
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{tid}'")
    client = request.client.host if request.client else "?"
    log.info(f"PREDICT request from {client}: trip {tid}")
    try:
        pred = predict_trip(TRIPS_DIR / tid, ART)
        t0 = time.perf_counter()
        metrics = evaluate_prediction(TRIPS_DIR / tid, pred)
        eval_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        log.exception(f"predict failed for trip {tid}")
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}") from e

    td = pred["_td"]
    tm = {k: round(v, 3) for k, v in pred["timings"].items()}
    log.info(f"  trip {tid}: {pred['n_points']} GPS points, {len(td.cands.obs)} candidates, "
             f"{len(pred['route_arcs'])} route arcs, chain breaks={pred['chain_breaks']}, "
             f"unbridged gaps={pred['unbridged_gaps']}")
    log.info(f"  timings (s): " + ", ".join(f"{k}={v}" for k, v in tm.items()) + f", evaluation_s={eval_s:.3f}")
    if metrics:
        log.info(f"  metrics: Top-1={metrics['top1_accuracy']:.4f}  classifier-only Top-1={metrics['classifier_only_top1']:.4f}  "
                 f"strict acc={metrics['strict_accuracy']:.4f}  continuity={metrics['plausible_continuity']:.4f}  "
                 f"route F1={metrics['route_f1']:.4f}")
    else:
        log.info("  metrics: no reference route available for this trip")

    tr = td.trip.track
    return {
        "trip_id": tid,
        "n_points": pred["n_points"],
        "gps": [[_r(lat), _r(lon)] for lon, lat in tr[:, :2]],
        "snapped": [None if not np.isfinite(lon) else [_r(lat), _r(lon)] for lon, lat in pred["snapped_lonlat"]],
        "point_arc_ids": [int(a) for a in pred["point_arcs"]],
        "route_arc_ids": [int(a) for a in pred["route_arcs"]],
        "route_polylines": _route_polylines(pred["route_arcs"], td.trip.nodes, td.trip.arcs, None),
        "chain_breaks": pred["chain_breaks"],
        "unbridged_gaps": pred["unbridged_gaps"],
        "metrics": metrics,
        "timings_s": {**tm, "evaluation_s": round(eval_s, 3)},
        "model": {"classifier": ART.model_card.get("classifier"), "hmm_params": ART.hmm_params},
    }
