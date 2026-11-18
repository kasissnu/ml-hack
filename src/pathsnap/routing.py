"""Bounded shortest-path search on a trip's road network.

The HMM needs network distances between candidate arcs of consecutive GPS points. Those distances
are short (tens to hundreds of metres), so a plain Dijkstra with a distance limit explores only a
handful of nodes. A pure-Python heap implementation on CSR adjacency lists is much faster here than
calling scipy's Dijkstra once per source (which re-validates the whole graph on every call).
Results are cached per source node and reused whenever the cached search radius is large enough.
"""
from __future__ import annotations

import heapq
import math

from .graph import RoadGraph


class Router:
    def __init__(self, g: RoadGraph):
        csr = g.csr.tocsr()
        self.indptr = csr.indptr.tolist()
        self.indices = csr.indices.tolist()
        self.weights = csr.data.tolist()
        self.edge_arc = g._edge_arc
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def search(self, src: int, limit: float):
        """(dist, pred) dicts for all nodes reachable from src within `limit` metres."""
        c = self._cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        ip, ix, w = self.indptr, self.indices, self.weights
        dist = {src: 0.0}
        pred = {src: -1}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ip[u], ip[u + 1]):
                v = ix[k]
                nd = d + w[k]
                if nd <= limit and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist, pred)
        return dist, pred

    def distance(self, src: int, dst: int, limit: float) -> float:
        dist, _ = self.search(src, limit)
        d = dist.get(dst, math.inf)
        return d if d <= limit else math.inf

    def arc_path(self, src: int, dst: int, limit: float) -> list[int] | None:
        """Arc ids of the shortest path src -> dst (empty if src == dst, None if unreachable)."""
        if src == dst:
            return []
        dist, pred = self.search(src, limit)
        if dist.get(dst, math.inf) > limit:   # cached searches may extend beyond `limit`
            return None
        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(pred[nodes[-1]])
        nodes.reverse()
        return [int(self.edge_arc[(u, v)]) for u, v in zip(nodes[:-1], nodes[1:])]
