"""Derive one reference arc per GPS point from the reference route.

The `.route` file lists the arcs travelled *in order* but has ~5x fewer rows than GPS points, so
there is no direct point -> arc label. We align the two sequences with a monotone dynamic programme:

    choose r(0) <= r(1) <= ... <= r(N-1)  (indices into the route)
    minimising  sum_i cost(i, r(i))
    cost(i, j) = min( dist(point i, route arc j) + heading_penalty(i, j),  CAP )

Monotonicity encodes "the vehicle moves forward along its route", which disambiguates places the
route passes twice (loops, out-and-back on the twin arc). Forward skips are free because short
arcs can be traversed between two samples. The CAP keeps far-away points (route not covering that
part of the trip, GPS outliers) from dragging the alignment; after decoding, any point farther
than `max_match_dist_m` from its assigned arc is labelled -1 (unmatched).

The recurrence  D[i, j] = cost(i, j) + min_{j' <= j} D[i-1, j']  is computed with a running
prefix-minimum, so the whole DP is O(N * M) vectorised over j.
"""
from __future__ import annotations

import numpy as np

from .geo import angle_diff_deg, project_point_segment
from .graph import RoadGraph


def point_headings(P: np.ndarray, t: np.ndarray, w: int = 2, min_disp: float = 5.0):
    """Smoothed heading (deg), speed (m/s) and validity flag for each point.

    Heading uses the displacement between points i-w and i+w; if the vehicle moved less than
    `min_disp` metres the heading is unreliable (stationary) and is filled from the nearest valid one."""
    n = len(P)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    d = P[hi] - P[lo]
    disp = np.hypot(d[:, 0], d[:, 1])
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    speed = disp / dt
    heading = (np.degrees(np.arctan2(d[:, 0], d[:, 1])) + 360) % 360
    valid = disp >= min_disp
    if valid.any() and not valid.all():
        idx = np.where(valid, np.arange(n), -1)
        ffill = np.maximum.accumulate(idx)
        bidx = np.where(valid, np.arange(n), n)
        bfill = np.minimum.accumulate(bidx[::-1])[::-1]
        src = np.where(ffill >= 0, ffill, bfill)
        heading = heading[src]
    return heading, speed, valid


def derive_labels(P: np.ndarray, t: np.ndarray, g: RoadGraph, route: np.ndarray,
                  max_match_dist_m: float = 40.0, heading_lambda_m: float = 5.0,
                  min_speed_for_heading: float = 2.0):
    """Return (labels, route_index, dist) for each GPS point. labels[i] = arc id or -1."""
    route = np.asarray(route, dtype=np.int64)
    N, M = len(P), len(route)
    A, B = g.A[route], g.B[route]
    rb = g.bearing[route]
    heading, speed, hvalid = point_headings(P, t)
    use_h = hvalid & (speed >= min_speed_for_heading)
    cap = max_match_dist_m * 1.5

    bp_dtype = np.int16 if M < np.iinfo(np.int16).max else np.int32
    bp = np.empty((N, M), dtype=bp_dtype)
    idx = np.arange(M)

    def cost_row(i):
        d, _, _ = project_point_segment(P[i][None, :], A, B)
        if use_h[i]:
            d = d + heading_lambda_m * (1 - np.cos(np.radians(angle_diff_deg(heading[i], rb)))) / 2
        return np.minimum(d, cap)

    D = cost_row(0)
    bp[0] = idx
    for i in range(1, N):
        pm = np.minimum.accumulate(D)
        am = np.maximum.accumulate(np.where(D == pm, idx, 0))
        D = cost_row(i) + pm
        bp[i] = am

    r = np.empty(N, dtype=np.int64)
    r[-1] = int(np.argmin(D))
    for i in range(N - 1, 0, -1):
        r[i - 1] = bp[i, r[i]]

    dist, _, _ = project_point_segment(P, A[r], B[r])
    labels = route[r].copy()
    labels[dist > max_match_dist_m] = -1
    return labels, r, dist


def label_stats(labels, r, route) -> dict:
    matched = labels >= 0
    covered = np.unique(r[matched]) if matched.any() else np.array([], dtype=int)
    return dict(n_points=len(labels), n_matched=int(matched.sum()), frac_matched=float(matched.mean()),
                n_route_arcs=len(route), route_arcs_with_points=len(covered),
                frac_route_arcs_covered=len(covered) / max(len(route), 1),
                monotone=bool(np.all(np.diff(r) >= 0)))
