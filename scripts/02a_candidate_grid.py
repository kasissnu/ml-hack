"""Phase 2a - Derive per-point labels and choose candidate-generation parameters (radius R, K).

Uses the TRAIN partition only. For each (R, K) we measure
  candidate miss rate  = share of matched GPS points whose reference arc is NOT in the candidate set
  candidates / point   = average candidate-set size (cost for the classifier and the HMM)
and pick the cheapest setting whose miss rate is within 0.5 percentage points of the best.

Outputs:
  cache/labels_<partition>.pkl         per-trip label arrays (train and student_test)
  results/labels/label_stats.csv       matched fraction, route coverage, monotonicity per trip
  results/candidates/grid.csv, grid.png, true_rank_hist.png, chosen.json
  results/labels/maps/labels_<trip>.html   label validation maps
"""
import _bootstrap  # noqa: F401

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import viz
from pathsnap.candidates import ArcIndex, true_arc_rank
from pathsnap.config import load_config, partition_dir, path
from pathsnap.geo import LocalProjection
from pathsnap.graph import RoadGraph
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import derive_labels, label_stats

CFG = load_config()
LC = CFG["labels"]
RADII = [30, 50, 75, 100]
KS = [4, 6, 8, 12, 16]
OUT_C = path("results") / "candidates"; OUT_C.mkdir(parents=True, exist_ok=True)
OUT_L = path("results") / "labels"; (OUT_L / "maps").mkdir(parents=True, exist_ok=True)


def process(partition, tid, want_grid):
    d = partition_dir(partition) / tid
    trip = load_trip(d); route = load_route(d)
    proj = LocalProjection.for_points(trip.track[:, :2])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    P = proj.to_xy(trip.track[:, :2])
    labels, r, dist = derive_labels(P, trip.track[:, 2], g, route, LC["max_match_dist_m"],
                                    LC["heading_lambda_m"], LC["min_speed_for_heading"])
    st = dict(partition=partition, trip_id=tid, **label_stats(labels, r, route),
              median_label_dist=float(np.median(dist[labels >= 0])) if (labels >= 0).any() else np.nan)
    grid = None
    if want_grid:
        idx = ArcIndex(g, P, max_radius=max(RADII), step=CFG["candidates"]["densify_step_m"])
        d_true, n_closer = true_arc_rank(P, g, labels, idx, max(RADII))
        big = idx.query(P, max(RADII), k=10_000)
        rows = []
        m = labels >= 0
        for R in RADII:
            n_in = np.bincount(big.obs[big.dist <= R], minlength=len(P))
            for K in KS:
                hit = m & (d_true <= R) & (n_closer < K)
                rows.append(dict(trip_id=tid, R=R, K=K, n_matched=int(m.sum()), n_hit=int(hit.sum()),
                                 sum_cands=float(np.minimum(n_in, K)[m].sum())))
        grid = (rows, n_closer[m & (d_true <= max(RADII))])
    return tid, labels, st, grid


def label_map(partition, tid, labels):
    d = partition_dir(partition) / tid
    trip = load_trip(d); route = load_route(d)
    m = viz.base_map(trip.track)
    viz.add_route(m, trip.nodes, trip.arcs, route, "Reference route", viz.COLORS["reference"])
    for name, mask, color in [("Matched points", labels >= 0, viz.COLORS["gps"]),
                              ("Unmatched points (> 40 m)", labels < 0, viz.COLORS["wrong"])]:
        fg = folium.FeatureGroup(name=f"{name} ({int(mask.sum())})")
        for i in np.flatnonzero(mask)[::2]:
            folium.CircleMarker([trip.track[i, 1], trip.track[i, 0]], radius=2.5, color=color, fill=True,
                                tooltip=f"#{i} label arc {labels[i]}").add_to(fg)
        fg.add_to(m)
    viz.add_legend(m, [("Reference route", viz.COLORS["reference"]), ("Matched GPS", viz.COLORS["gps"]),
                       ("Unmatched GPS", viz.COLORS["wrong"])], title=f"Derived labels: {partition} {tid}")
    viz.finish(m).save(str(OUT_L / "maps" / f"labels_{partition}_{tid}.html"))


def main():
    t0 = time.time()
    all_stats = []
    grid_rows, ranks = [], []
    for partition in ("train", "student_test"):
        tids = list_trips(partition_dir(partition))
        res = Parallel(n_jobs=8)(delayed(process)(partition, t, partition == "train") for t in tids)
        labels = {tid: lab for tid, lab, _, _ in res}
        with open(path("cache") / f"labels_{partition}.pkl", "wb") as f:
            pickle.dump(labels, f)
        all_stats += [st for _, _, st, _ in res]
        for _, _, _, gr in res:
            if gr is not None:
                grid_rows += gr[0]; ranks.append(gr[1])
        print(f"{partition}: labels for {len(tids)} trips ({time.time() - t0:.0f}s)")
    stats = pd.DataFrame(all_stats)
    stats.to_csv(OUT_L / "label_stats.csv", index=False)
    print(stats.groupby("partition")[["frac_matched", "frac_route_arcs_covered", "median_label_dist"]].describe().T)
    assert stats.monotone.all()

    grid = pd.DataFrame(grid_rows).groupby(["R", "K"]).sum(numeric_only=True).reset_index()
    grid["miss_rate"] = 1 - grid.n_hit / grid.n_matched
    grid["cands_per_point"] = grid.sum_cands / grid.n_matched
    grid = grid[["R", "K", "n_matched", "miss_rate", "cands_per_point"]]
    grid.to_csv(OUT_C / "grid.csv", index=False)
    best = grid.miss_rate.min()
    ok = grid[grid.miss_rate <= best + 0.005].sort_values(["cands_per_point", "miss_rate"])
    chosen = ok.iloc[0]
    json.dump({"radius_m": float(chosen.R), "k": int(chosen.K), "miss_rate_train": float(chosen.miss_rate),
               "cands_per_point": float(chosen.cands_per_point), "rule": "cheapest within 0.5pp of best miss rate"},
              open(OUT_C / "chosen.json", "w"), indent=2)
    print(grid.to_string(index=False)); print("chosen:", dict(chosen))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for R in RADII:
        s = grid[grid.R == R]
        ax[0].plot(s.K, 100 * s.miss_rate, "o-", label=f"R={R} m")
        ax[1].plot(s.K, s.cands_per_point, "o-", label=f"R={R} m")
    ax[0].set(xlabel="K", ylabel="candidate miss rate (%)", title="Miss rate vs K (train)")
    ax[1].set(xlabel="K", ylabel="candidates per point", title="Candidate-set size vs K (train)")
    for a_ in ax: a_.legend(); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_C / "grid.png", dpi=130); plt.close(fig)

    rk = np.concatenate(ranks)
    plt.figure(figsize=(6.4, 4))
    plt.hist(np.minimum(rk, 15), bins=np.arange(17) - 0.5, color="#3b6fb6", edgecolor="white")
    plt.xlabel("number of arcs strictly closer than the reference arc (15 = 15+)"); plt.ylabel("GPS points")
    plt.title("Rank of the reference arc by distance (train)"); plt.yscale("log")
    plt.tight_layout(); plt.savefig(OUT_C / "true_rank_hist.png", dpi=130); plt.close()

    lab_train = pickle.load(open(path("cache") / "labels_train.pkl", "rb"))
    for tid in ["00000001", "00000033"]:
        label_map("train", tid, lab_train[tid])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
