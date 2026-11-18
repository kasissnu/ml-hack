"""Geodesy helpers.

Longitude/latitude are angular, so distances in degrees are meaningless (1 deg of longitude is
~111 km at the equator but ~63 km at 55 N). Each trip spans at most ~75 km (median ~8 km), so we use
a *local equirectangular projection* centred on the trip:

    x = R * cos(lat0) * (lon - lon0)      (metres east)
    y = R * (lat - lat0)                  (metres north)

The relative scale error grows with the distance from the trip centre (below ~1 % at the edge of the
largest trip), but every distance the pipeline uses is local (GPS point to an arc within 50 m, or
network paths of a few hundred metres), so absolute errors stay below 1 m - the largest single-step
error versus haversine over all train trips is 0.72 m (results/eda/trip_stats.csv), far below GPS
noise. Checked in tests/test_geo_labels_candidates.py. Fast, vectorised, no extra dependency.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6371008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = R_EARTH * np.cos(np.deg2rad(self.lat0)) * np.pi / 180.0
        self.ky = R_EARTH * np.pi / 180.0

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjection":
        return cls(np.mean(lonlat[:, 0]), np.mean(lonlat[:, 1]))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=np.float64)
        return np.column_stack(((lonlat[:, 0] - self.lon0) * self.kx,
                                (lonlat[:, 1] - self.lat0) * self.ky))

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack((xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0))


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(dx, dy):
    """Compass bearing of a vector in projected metres: 0 = north, 90 = east."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def project_point_segment(P, A, B):
    """Project points P onto segments AB (all (n,2), broadcastable).

    Returns (dist, u, Q) where u in [0,1] is the fractional position along AB and Q the projection."""
    AB = B - A
    L2 = np.einsum("...i,...i->...", AB, AB)
    L2 = np.where(L2 > 0, L2, 1e-12)
    u = np.clip(np.einsum("...i,...i->...", P - A, AB) / L2, 0.0, 1.0)
    Q = A + u[..., None] * AB
    d = np.sqrt(np.einsum("...i,...i->...", P - Q, P - Q))
    return d, u, Q
