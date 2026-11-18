"""Feature engineering: one row per (GPS point, candidate arc).

Every feature is invariant to where in the world the trip is (no absolute coordinates), and none
uses the reference route: `prepare_trip` and `build_features` only see the track and the network.

Feature groups
  distance    : how far the point is from the arc, absolute and relative to the other candidates
  heading     : agreement between direction of travel and arc direction (separates two-way twins)
  position    : where on the arc the point projects, arc length
  topology    : node degrees, two-way flag, how crowded the candidate set is
  context     : distances of neighbouring points (i+-1, +-3, +-5) to the same arc, and whether the
                neighbours' nearest arcs are this arc or connected to it (route consistency)
  kinematics  : speed, time gaps, turn rate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import ArcIndex, Candidates
from .geo import LocalProjection, angle_diff_deg, project_point_segment
from .graph import RoadGraph
from .io import Trip
from .labels import point_headings

SIGMA_GPS = 5.0
CONTEXT_OFFSETS = (-5, -3, -1, 1, 3, 5)
CONN_WINDOW = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)

FEATURE_COLS = [
    # distance
    "d", "log_d", "d_minus_min", "d_ratio", "rank", "softmin", "gap12",
    # heading
    "cos_dh", "abs_dh", "heading_valid",
    # position on arc
    "u", "dist_to_start", "dist_to_end", "log_arc_len",
    # topology
    "out_deg_to", "in_deg_from", "is_two_way", "n_cand", "n_within_10m",
    # context
    *[f"ctx_d_{'m' if k < 0 else 'p'}{abs(k)}" for k in CONTEXT_OFFSETS],
    "ctx_d_mean", "ctx_conn_frac", "is_nearest_prev", "is_nearest_next",
    # kinematics
    "speed", "dt_prev", "dt_next", "turn_rate",
]


@dataclass
class TripData:
    trip: Trip
    proj: LocalProjection
    g: RoadGraph
    P: np.ndarray
    t: np.ndarray
    cands: Candidates


def prepare_trip(trip: Trip, radius: float, k: int, step: float = 10.0) -> TripData:
    proj = LocalProjection.for_points(trip.track[:, :2])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    P = proj.to_xy(trip.track[:, :2])
    index = ArcIndex(g, P, max_radius=4 * radius, step=step)
    cands = index.query(P, radius, k)
    return TripData(trip, proj, g, P, trip.track[:, 2].copy(), cands)


def build_features(td: TripData) -> pd.DataFrame:
    g, P, t, c = td.g, td.P, td.t, td.cands
    n = len(P)
    o, a, d, u, rank = c.obs, c.arc, c.dist, c.u, c.rank

    heading, speed, hvalid = point_headings(P, t)
    dt = np.diff(t)
    dt_prev = np.r_[1.0, dt]; dt_next = np.r_[dt, 1.0]
    im1 = np.clip(np.arange(n) - 1, 0, n - 1); ip1 = np.clip(np.arange(n) + 1, 0, n - 1)
    turn = angle_diff_deg(heading[ip1], heading[im1]) / np.maximum(t[ip1] - t[im1], 1e-6)

    # per-observation summaries
    n_cand = np.bincount(o, minlength=n)
    first = np.r_[0, np.cumsum(n_cand)[:-1]]
    has = n_cand > 0
    min_d = np.full(n, np.inf); min_d[has] = d[first[has]]
    second_d = np.full(n, np.inf)
    two = n_cand > 1
    second_d[two] = d[first[two] + 1]
    gap12 = np.full(n, 50.0)
    gap12[two] = second_d[two] - min_d[two]
    nearest = np.full(n, -1, dtype=np.int64); nearest[has] = a[first[has]]

    w = np.exp(-0.5 * (d / SIGMA_GPS) ** 2) + 1e-12
    softmin = w / np.bincount(o, weights=w, minlength=n)[o]
    within = np.bincount(o, weights=(d <= min_d[o] + 10.0).astype(float), minlength=n)

    dh = angle_diff_deg(heading[o], g.bearing[a])
    hv = hvalid[o]
    cos_dh = np.where(hv, np.cos(np.radians(dh)), 0.0)
    abs_dh = np.where(hv, dh, 90.0)

    f = {
        "d": d, "log_d": np.log1p(d), "d_minus_min": d - min_d[o], "d_ratio": d / (min_d[o] + 1.0),
        "rank": rank, "softmin": softmin, "gap12": np.minimum(gap12[o], 50.0),
        "cos_dh": cos_dh, "abs_dh": abs_dh, "heading_valid": hv.astype(float),
        "u": u, "dist_to_start": u * g.length[a], "dist_to_end": (1 - u) * g.length[a],
        "log_arc_len": np.log1p(g.length[a]),
        "out_deg_to": g.out_deg[g.arcs[a, 1]], "in_deg_from": g.in_deg[g.arcs[a, 0]],
        "is_two_way": g.is_two_way[a].astype(float), "n_cand": n_cand[o], "n_within_10m": within[o],
    }
    ctx = []
    for k in CONTEXT_OFFSETS:
        j = np.clip(o + k, 0, n - 1)
        dk, _, _ = project_point_segment(P[j], g.A[a], g.B[a])
        dk = np.minimum(dk, 200.0)
        f[f"ctx_d_{'m' if k < 0 else 'p'}{abs(k)}"] = dk
        ctx.append(dk)
    f["ctx_d_mean"] = np.mean(ctx, axis=0)

    conn = np.zeros(len(o))
    for k in CONN_WINDOW:
        j = np.clip(o + k, 0, n - 1)
        nk = nearest[j]
        ok = nk >= 0
        nks = np.where(ok, nk, 0)
        m = ok & ((nks == a) | (g.arcs[a, 1] == g.arcs[nks, 0]) | (g.arcs[nks, 1] == g.arcs[a, 0]))
        conn += m
    f["ctx_conn_frac"] = conn / len(CONN_WINDOW)
    f["is_nearest_prev"] = (nearest[np.clip(o - 1, 0, n - 1)] == a).astype(float)
    f["is_nearest_next"] = (nearest[np.clip(o + 1, 0, n - 1)] == a).astype(float)
    f["speed"] = np.minimum(speed[o], 70.0)
    f["dt_prev"] = np.minimum(dt_prev[o], 300.0)
    f["dt_next"] = np.minimum(dt_next[o], 300.0)
    f["turn_rate"] = np.minimum(turn[o], 180.0)

    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})[FEATURE_COLS]
    df.insert(0, "arc", a.astype(np.int64))
    df.insert(0, "obs", o.astype(np.int64))
    return df
