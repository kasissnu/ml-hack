"""FastAPI app: serves the UI and exposes POST /api/predict.

Currently uses MOCK predictions. When Person 1's model is ready, replace
`predict_route` below with: from src.decoder import predict_route
"""
import math
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

UI_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"
CENTER = (41.1496, -8.6109)  # change to your dataset's city

app = FastAPI(title="RouteRecon API")


class PredictRequest(BaseModel):
    trip_id: str


def predict_route(trip_id: str) -> dict:
    """MOCK. Real version should return arc ids + score (+ optional coordinates)."""
    rnd = random.Random(trip_id)
    lat, lon = CENTER
    angle = rnd.random() * 6.28
    coords, arcs = [[lat, lon]], []
    for _ in range(8 + rnd.randint(0, 9)):
        angle += (rnd.random() - 0.5) * 1.1
        lat += math.sin(angle) * 0.0035
        lon += math.cos(angle) * 0.0045
        coords.append([lat, lon])
        arcs.append(rnd.randint(1000, 91000))
    return {
        "trip_id": trip_id,
        "predicted_arc_ids": arcs,
        "route_score": round(0.6 + rnd.random() * 0.38, 2),
        "coordinates": coords,  # optional: [[lat, lon], ...] for map drawing
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.trip_id.strip():
        raise HTTPException(400, "trip_id is required")
    return predict_route(req.trip_id.strip())


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
