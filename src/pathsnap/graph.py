"""Road network representation for one trip.

Nodes are points, arcs are *directed* straight segments between two nodes (row index = arc id).
Two-way roads appear as two arcs (u->v and v->u) with identical geometry; only the direction of
travel distinguishes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .geo import LocalProjection, bearing_deg


class RoadGraph:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.nodes_xy = proj.to_xy(nodes_lonlat)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        fr, to = self.arcs[:, 0], self.arcs[:, 1]
        self.A = self.nodes_xy[fr]
        self.B = self.nodes_xy[to]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_deg(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(fr, minlength=self.n_nodes)
        self.in_deg = np.bincount(to, minlength=self.n_nodes)

        # reverse twin: arc v->u for arc u->v (or -1)
        key = fr * self.n_nodes + to
        rkey = to * self.n_nodes + fr
        s = pd.Series(np.arange(self.n_arcs), index=key)
        s = s[~s.index.duplicated()]
        self.reverse_twin = s.reindex(rkey).fillna(-1).to_numpy(np.int64)
        self.is_two_way = self.reverse_twin >= 0

        # node graph for shortest paths: keep the shortest of parallel arcs, avoid zero weights
        df = pd.DataFrame({"f": fr, "t": to, "w": np.maximum(self.length, 1e-3), "a": np.arange(self.n_arcs)})
        df = df[df.f != df.t].sort_values("w").drop_duplicates(["f", "t"])
        self.csr = csr_matrix((df.w.to_numpy(), (df.f.to_numpy(), df.t.to_numpy())),
                              shape=(self.n_nodes, self.n_nodes))
        self._edge_arc = {(f, t): a for f, t, a in zip(df.f.to_numpy(), df.t.to_numpy(), df.a.to_numpy())}
        self._sp_cache: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ connectivity
    def components(self):
        weak_n, weak_lab = connected_components(self.csr, directed=True, connection="weak")
        strong_n, strong_lab = connected_components(self.csr, directed=True, connection="strong")
        return weak_n, weak_lab, strong_n, strong_lab

    def isolated_nodes(self) -> int:
        return int(((self.out_deg + self.in_deg) == 0).sum())

    def arcs_connected(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """True where arc b can directly follow arc a (or they are the same arc)."""
        a = np.asarray(a); b = np.asarray(b)
        return (a == b) | (self.arcs[a, 1] == self.arcs[b, 0])

    # ------------------------------------------------------------------ shortest paths
    def node_distances(self, src: int, dst: np.ndarray, limit: float) -> np.ndarray:
        """Network distance (m) from node src to each node in dst, inf if farther than `limit`.

        Results are cached per source node; a cached run is reused when its limit is large enough."""
        cached = self._sp_cache.get(int(src))
        if cached is None or cached[0] < limit:
            run_limit = max(limit, cached[0] * 2 if cached else limit)
            dist = dijkstra(self.csr, directed=True, indices=int(src), limit=run_limit)
            reached = np.flatnonzero(np.isfinite(dist))
            cached = (run_limit, reached, dist[reached])
            self._sp_cache[int(src)] = cached
        _, reached, rdist = cached
        dst = np.asarray(dst)
        pos = np.searchsorted(reached, dst)
        pos = np.clip(pos, 0, max(len(reached) - 1, 0))
        out = np.full(dst.shape, np.inf)
        if len(reached):
            hit = reached[pos] == dst
            out[hit] = rdist[pos[hit]]
        out[out > limit] = np.inf
        return out

    def shortest_arc_path(self, src_node: int, dst_node: int, limit: float = np.inf) -> list[int] | None:
        """Sequence of arc ids forming the shortest path src_node -> dst_node, or None."""
        if src_node == dst_node:
            return []
        dist, pred = dijkstra(self.csr, directed=True, indices=int(src_node),
                              limit=limit, return_predecessors=True)
        if not np.isfinite(dist[dst_node]):
            return None
        nodes = [int(dst_node)]
        while nodes[-1] != src_node:
            p = pred[nodes[-1]]
            if p < 0:
                return None
            nodes.append(int(p))
        nodes.reverse()
        return [int(self._edge_arc[(u, v)]) for u, v in zip(nodes[:-1], nodes[1:])]

    def clear_cache(self):
        self._sp_cache.clear()
