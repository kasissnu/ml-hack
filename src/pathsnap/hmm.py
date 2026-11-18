"""HMM map-matching with Viterbi decoding (Newson & Krumm style, classifier emissions).

Hidden state at GPS observation t : one of its candidate arcs c, together with the projected
                                    position u_c of the point on that arc.
Emission  log P(o_t | c)          : w_e * log p_c, where p_c is the classifier probability
                                    re-normalised over the candidates of observation t.
Transition log P(c_{t-1}=a -> c_t=b): -( |d_net(a,b) - d_gc| + U-turn penalty ) / beta
    d_gc  : straight-line distance between the two GPS points (projected metres)
    d_net : distance driven along the network from the projection on a to the projection on b
            same arc, forward : (u_b - u_a) * len
            same arc, backward: |u_b - u_a| * len, flagged as a U-turn if more than `tol` metres
            otherwise         : (1 - u_a) * len_a + shortest_path(to(a), from(b)) + u_b * len_b
    b == reverse twin of a, or a backward move on the same arc, adds `uturn_m` metres.
    No path within the search limit -> transition impossible (-inf).

Chains are broken (decoded separately and restarted from the emission) when
    * the time gap dt > gap_s,
    * an observation has no candidates,
    * no transition from any previous state is possible.
Expensive network distances are computed once per trip (`build_lattice`); decoding with different
parameters (`viterbi`) is then pure numpy, which makes decoder tuning cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import RoadGraph
from .routing import Router

DEFAULT_PARAMS = dict(beta=10.0, uturn_m=50.0, gap_s=60.0, w_e=1.0)
SAME_ARC_TOL_M = 10.0
MAX_GAP_COMPUTE_S = 300.0


@dataclass
class Lattice:
    starts: np.ndarray          # (n+1,) candidate-row offsets per observation (rows sorted by obs)
    arc: np.ndarray             # (C,)
    dt: np.ndarray              # (n,) time since previous observation (dt[0] = inf)
    dgc: np.ndarray             # (n,)
    dnet: list                  # dnet[t]: (k_{t-1}, k_t) array or None
    uturn: list                 # uturn[t]: bool array or None

    @property
    def n_obs(self):
        return len(self.starts) - 1


def build_lattice(g: RoadGraph, router: Router, P: np.ndarray, t: np.ndarray,
                  obs: np.ndarray, arc: np.ndarray, u: np.ndarray) -> Lattice:
    n = len(P)
    counts = np.bincount(obs, minlength=n)
    starts = np.r_[0, np.cumsum(counts)]
    dt = np.r_[np.inf, np.diff(t)]
    dgc = np.r_[0.0, np.hypot(*np.diff(P, axis=0).T)]
    to_node = g.arcs[:, 1]; from_node = g.arcs[:, 0]
    length = g.length; twin = g.reverse_twin
    dnet_l: list = [None] * n
    uturn_l: list = [None] * n
    for i in range(1, n):
        s0, e0, s1, e1 = starts[i - 1], starts[i], starts[i], starts[i + 1]
        if e0 == s0 or e1 == s1 or dt[i] > MAX_GAP_COMPUTE_S:
            continue
        ap, up = arc[s0:e0], u[s0:e0]
        ac, uc = arc[s1:e1], u[s1:e1]
        limit = float(min(max(250.0, 3.0 * dgc[i] + 50.0), 5000.0))
        D = np.full((len(ap), len(ac)), np.inf)
        U = np.zeros((len(ap), len(ac)), dtype=bool)
        fb = from_node[ac]
        for r in range(len(ap)):
            a = int(ap[r])
            same = ac == a
            if same.any():
                d = (uc[same] - up[r]) * length[a]
                D[r, same] = np.abs(d)
                U[r, same] = d < -SAME_ARC_TOL_M
            other = ~same
            if other.any():
                base = (1.0 - up[r]) * length[a]
                dist, _ = router.search(int(to_node[a]), limit)
                sp = np.fromiter((dist.get(int(x), np.inf) for x in fb[other]), float, count=int(other.sum()))
                D[r, other] = base + sp + uc[other] * length[ac[other]]
                U[r, other] = ac[other] == twin[a]
        D[D > limit + length[ap].max() + length[ac].max()] = np.inf
        dnet_l[i] = D
        uturn_l[i] = U
    return Lattice(starts, arc, dt, dgc, dnet_l, uturn_l)


def emission_logp(p: np.ndarray, starts: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    p = np.clip(p.astype(float), floor, None)
    n = len(starts) - 1
    seg = np.repeat(np.arange(n), np.diff(starts))
    tot = np.bincount(seg, weights=p, minlength=n)
    return np.log(np.maximum(p / tot[seg], floor))


def viterbi(lat: Lattice, logp_emis: np.ndarray, params: dict | None = None):
    """Return (chosen candidate row per observation or -1, number of chain breaks)."""
    prm = {**DEFAULT_PARAMS, **(params or {})}
    beta, uturn_m, gap_s, w_e = prm["beta"], prm["uturn_m"], prm["gap_s"], prm["w_e"]
    n = lat.n_obs
    st = lat.starts
    choice = np.full(n, -1, dtype=np.int64)
    bps: list = [None] * n
    score = None
    chain_start = 0
    breaks = 0

    def backtrack(end, sc):
        j = int(np.argmax(sc))
        for k in range(end, chain_start - 1, -1):
            choice[k] = st[k] + j
            if k > chain_start:
                j = int(bps[k][j])

    for i in range(n):
        k = st[i + 1] - st[i]
        if k == 0:
            if score is not None:
                backtrack(i - 1, score); breaks += 1
            score = None
            continue
        em = w_e * logp_emis[st[i]:st[i + 1]]
        D = lat.dnet[i]
        if score is not None and D is not None and lat.dt[i] <= gap_s:
            cost = np.abs(D - lat.dgc[i]) + uturn_m * lat.uturn[i]
            tr = score[:, None] - cost / beta
            best = np.argmax(tr, axis=0)
            val = tr[best, np.arange(k)]
            if np.isfinite(val).any():
                bps[i] = best
                score = val + em
                continue
        if score is not None:
            backtrack(i - 1, score); breaks += 1
        chain_start = i
        score = em.copy()
    if score is not None:
        backtrack(n - 1, score)
    return choice, breaks


def transition_stats(lat: Lattice, choice: np.ndarray, g: RoadGraph, slack_m: float = 50.0):
    """Continuity of a per-point arc sequence.

    For every pair of consecutive observations (dt <= MAX_GAP_COMPUTE_S, both assigned) whose arcs
    differ we count:
      direct    : the second arc starts where the first ends (physically adjacent)
      plausible : the second arc is reachable from the first by driving at most 2 * d_gc + slack
                  metres along the network (no jumps to disconnected / parallel roads)
    Returns (n_direct, n_plausible, n_changes)."""
    n_dir = n_pl = n_ch = 0
    st = lat.starts
    for i in range(1, lat.n_obs):
        a, b = choice[i - 1], choice[i]
        if a < 0 or b < 0 or lat.dnet[i] is None:
            continue
        aa, bb = lat.arc[a], lat.arc[b]
        if aa == bb:
            continue
        n_ch += 1
        n_dir += int(g.arcs[aa, 1] == g.arcs[bb, 0])
        d = lat.dnet[i][a - st[i - 1], b - st[i]]
        n_pl += int(np.isfinite(d) and d <= 2 * lat.dgc[i] + slack_m)
    return n_dir, n_pl, n_ch


def stitch_route(point_arcs: np.ndarray, g: RoadGraph, router: Router, max_bridge_m: float = 3000.0):
    """Turn per-point arcs into a continuous arc sequence by inserting shortest paths between
    consecutive arcs that are not directly connected. Returns (route arcs, n_unbridged gaps)."""
    seq = [int(a) for a in point_arcs if a >= 0]
    if not seq:
        return [], 0
    out = [seq[0]]
    unbridged = 0
    for b in seq[1:]:
        a = out[-1]
        if b == a:
            continue
        if g.arcs[a, 1] == g.arcs[b, 0]:
            out.append(b)
            continue
        path = router.arc_path(int(g.arcs[a, 1]), int(g.arcs[b, 0]), max_bridge_m)
        if path is None:
            unbridged += 1
        else:
            out.extend(path)
        out.append(b)
    return out, unbridged
