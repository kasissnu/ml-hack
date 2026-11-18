"""Candidate road arcs for each GPS point.

Each arc is densified into sample points every `step` metres and indexed in a KD-tree. Any point of
a segment lies within step/2 of a sample, so a ball query of radius R + step/2 returns a superset
of the arcs within R; exact point-to-segment distances then filter and rank them.

For every GPS point we keep the K nearest arcs within R. Arcs tied with the K-th distance are also
kept, so both directions of a two-way road always enter together. If nothing lies within R we
retry with 2R and 4R; if still empty the point has no candidates (the HMM then splits the chain).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geo import project_point_segment
from .graph import RoadGraph


@dataclass
class Candidates:
    obs: np.ndarray    # (C,) GPS point index of each candidate row
    arc: np.ndarray    # (C,) arc id
    dist: np.ndarray   # (C,) perpendicular distance (m)
    u: np.ndarray      # (C,) fractional position of the projection along the arc
    rank: np.ndarray   # (C,) 0 = nearest
    n_obs: int

    def counts(self) -> np.ndarray:
        return np.bincount(self.obs, minlength=self.n_obs)


class ArcIndex:
    def __init__(self, g: RoadGraph, P: np.ndarray, max_radius: float, step: float = 10.0):
        self.g = g
        self.step = step
        # only index arcs whose bounding box comes near the trajectory
        lo = P.min(0) - max_radius
        hi = P.max(0) + max_radius
        amin = np.minimum(g.A, g.B); amax = np.maximum(g.A, g.B)
        near = np.flatnonzero(((amax >= lo) & (amin <= hi)).all(1))
        n_seg = np.maximum(np.ceil(g.length[near] / step).astype(int), 1)
        arc_rep = np.repeat(near, n_seg + 1)
        start = np.repeat(np.cumsum(n_seg + 1) - (n_seg + 1), n_seg + 1)
        frac = (np.arange(len(arc_rep)) - start) / np.repeat(n_seg, n_seg + 1)
        pts = g.A[arc_rep] + frac[:, None] * (g.B[arc_rep] - g.A[arc_rep])
        self.sample_arc = arc_rep
        self.tree = cKDTree(pts) if len(pts) else None

    def query(self, P: np.ndarray, radius: float, k: int) -> Candidates:
        n = len(P)
        obs_l, arc_l, d_l, u_l, rk_l = [], [], [], [], []
        todo = np.arange(n)
        for mult in (1, 2, 4):
            if len(todo) == 0 or self.tree is None:
                break
            R = radius * mult
            hits = self.tree.query_ball_point(P[todo], R + self.step / 2)
            lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
            if lens.sum() == 0:
                continue
            o = np.repeat(todo, lens)
            a = self.sample_arc[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])]
            # unique (obs, arc) pairs
            key = np.unique(o * (self.g.n_arcs + 1) + a)
            o = key // (self.g.n_arcs + 1); a = key % (self.g.n_arcs + 1)
            d, u, _ = project_point_segment(P[o], self.g.A[a], self.g.B[a])
            keep = d <= R
            o, a, d, u = o[keep], a[keep], d[keep], u[keep]
            # sort by obs then distance, compute rank within obs
            order = np.lexsort((a, d, o))
            o, a, d, u = o[order], a[order], d[order], u[order]
            first = np.r_[0, np.flatnonzero(np.diff(o)) + 1]
            grp_start = np.repeat(first, np.diff(np.r_[first, len(o)]))
            rank = np.arange(len(o)) - grp_start
            # distance of the k-th candidate of each obs; keep everything tied with it
            kth_pos = np.minimum(first + k - 1, np.r_[first[1:], len(o)] - 1)
            kth_d = np.repeat(d[kth_pos], np.diff(np.r_[first, len(o)]))
            sel = (rank < k) | (d <= kth_d + 1e-6)
            obs_l.append(o[sel]); arc_l.append(a[sel]); d_l.append(d[sel]); u_l.append(u[sel]); rk_l.append(rank[sel])
            found = np.unique(o)
            todo = np.setdiff1d(todo, found)
        if obs_l:
            obs = np.concatenate(obs_l); arc = np.concatenate(arc_l); dist = np.concatenate(d_l)
            u = np.concatenate(u_l); rank = np.concatenate(rk_l)
            order = np.lexsort((rank, obs))
            return Candidates(obs[order], arc[order], dist[order], u[order], rank[order], n)
        e = np.array([], dtype=np.int64)
        return Candidates(e, e, e.astype(float), e.astype(float), e, n)


def true_arc_rank(P, g: RoadGraph, labels: np.ndarray, index: ArcIndex, max_radius: float):
    """For miss-rate analysis: distance of the labelled arc and how many arcs are strictly closer."""
    big = index.query(P, max_radius, k=10_000)
    d_true = np.full(len(P), np.inf)
    n_closer = np.full(len(P), np.iinfo(np.int64).max)
    m = labels >= 0
    dt, _, _ = project_point_segment(P[m], g.A[labels[m]], g.B[labels[m]])
    d_true[m] = dt
    closer = big.dist < d_true[big.obs] - 1e-6
    n_closer[m] = 0
    np.add.at(n_closer, big.obs[closer & m[big.obs]], 1)
    return d_true, n_closer
