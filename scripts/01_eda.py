"""Phase 1 - Dataset inventory, data-quality checks and exploratory analysis.

Partitions analysed: train (80 trips) for all EDA statistics and plots; student_test (10 trips) for
inventory / sanity checks only. The evaluator partition is never touched.

Outputs (results/eda/):
  inventory.csv            files, sizes and row counts per trip
  trip_stats.csv           per-trip trajectory, network and route statistics
  partition_summary.csv    aggregated statistics per partition
  missing_values.csv       missing / invalid values per file type and partition
  data_quality_summary.md  automatically generated list of issues found
  *.png                    plots used in the report
  maps/eda_<trip>.html     Folium maps (GPS + road network + reference route)
"""
import _bootstrap  # noqa: F401

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection

from pathsnap import viz
from pathsnap.config import load_config, partition_dir, path
from pathsnap.geo import LocalProjection, haversine, project_point_segment
from pathsnap.graph import RoadGraph
from pathsnap.io import list_trips, load_route, load_trip, read_raw_track

CFG = load_config()
OUT = path("results") / "eda"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "maps").mkdir(exist_ok=True)
EXTS = ["track", "nodes", "arcs", "route"]


def gps_to_route_distance(P, g: RoadGraph, route, chunk=2000):
    A, B = g.A[route], g.B[route]
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        d, _, _ = project_point_segment(P[s:s + chunk, None, :], A[None], B[None])
        out[s:s + chunk] = d.min(1)
    return out


def missing_counts(trip_dir: Path):
    rows = []
    for ext, ncol in [("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)]:
        p = trip_dir / f"{trip_dir.name}.{ext}"
        if not p.exists():
            rows.append(dict(file=ext, present=False, rows=0, missing_cells=np.nan, bad_rows=np.nan))
            continue
        df = pd.read_csv(p, sep=r"\s+", header=None)
        rows.append(dict(file=ext, present=True, rows=len(df), n_cols=df.shape[1],
                         missing_cells=int(df.isna().sum().sum()),
                         bad_rows=int((df.shape[1] != ncol) or 0)))
    return rows


def analyse_trip(partition: str, tid: str):
    d = partition_dir(partition) / tid
    inv = {"partition": partition, "trip_id": tid}
    for ext in EXTS:
        p = d / f"{tid}.{ext}"
        inv[f"{ext}_present"] = p.exists()
        inv[f"{ext}_bytes"] = p.stat().st_size if p.exists() else 0
    miss = [dict(partition=partition, trip_id=tid, **r) for r in missing_counts(d)]

    raw = read_raw_track(d)
    trip = load_trip(d)
    route = load_route(d)
    tr = trip.track
    proj = LocalProjection.for_points(tr[:, :2])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    P = proj.to_xy(tr[:, :2])

    dt = np.diff(tr[:, 2])
    step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    speed = step / np.maximum(dt, 1e-6)
    # cross-check: projected distance vs haversine
    step_xy = np.hypot(*np.diff(P, axis=0).T)
    proj_err = np.abs(step_xy - step).max() if len(step) else 0.0

    weak_n, weak_lab, strong_n, strong_lab = g.components()
    scc_sizes = np.bincount(strong_lab)
    arcs = trip.arcs
    s = dict(partition=partition, trip_id=tid,
             raw_points=len(raw), clean_points=len(tr), **{f"clean_{k}": v for k, v in trip.cleaning.items()},
             t_start=tr[0, 2], duration_s=tr[-1, 2] - tr[0, 2],
             dt_min=dt.min(), dt_median=np.median(dt), dt_max=dt.max(), frac_dt_eq_1=np.mean(np.isclose(dt, 1.0)),
             **{f"gaps_gt_{g_}s": int((dt > g_).sum()) for g_ in CFG["cleaning"]["gap_flags_s"]},
             path_length_km=step.sum() / 1000, speed_median=np.median(speed), speed_p95=np.percentile(speed, 95),
             speed_max=speed.max(), n_speed_outliers=int((speed > CFG["cleaning"]["max_speed_mps"]).sum()),
             frac_stationary=np.mean(step < 0.5),
             lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
             extent_x_km=np.ptp(P[:, 0]) / 1000, extent_y_km=np.ptp(P[:, 1]) / 1000,
             max_proj_error_m=proj_err,
             n_nodes=g.n_nodes, n_arcs=g.n_arcs, n_self_loops=int((arcs[:, 0] == arcs[:, 1]).sum()),
             n_duplicate_arcs=int(len(arcs) - len(np.unique(arcs, axis=0))),
             n_zero_length_arcs=int((g.length < 0.01).sum()),
             arc_length_median=np.median(g.length), frac_two_way=g.is_two_way.mean(),
             isolated_nodes=g.isolated_nodes(), weak_components=weak_n, strong_components=strong_n,
             largest_scc_frac=scc_sizes.max() / g.n_nodes,
             node_bbox_contains_track=bool((trip.nodes[:, 0].min() <= tr[:, 0].min()) and (trip.nodes[:, 0].max() >= tr[:, 0].max())
                                           and (trip.nodes[:, 1].min() <= tr[:, 1].min()) and (trip.nodes[:, 1].max() >= tr[:, 1].max())))
    dist = None
    if route is not None:
        valid = (route >= 0) & (route < g.n_arcs)
        s.update(route_len=len(route), route_unique=len(np.unique(route)), route_invalid_ids=int((~valid).sum()))
        r = route[valid]
        s["route_contiguity"] = float(np.mean(arcs[r[:-1], 1] == arcs[r[1:], 0])) if len(r) > 1 else 1.0
        s["route_length_km"] = g.length[r].sum() / 1000
        s["points_per_route_arc"] = len(tr) / max(len(r), 1)
        dist = gps_to_route_distance(P, g, r)
        s.update(gps_route_dist_median=np.median(dist), gps_route_dist_p95=np.percentile(dist, 95),
                 gps_route_dist_max=dist.max(),
                 frac_points_gt_40m=np.mean(dist > CFG["labels"]["max_match_dist_m"]))
    return inv, s, miss, (dt, speed, dist)


def plot_hist(data, title, xlabel, fname, bins=60, log=False, logx=False):
    plt.figure(figsize=(6.4, 4))
    if logx:
        data = data[data > 0]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), bins)
        plt.xscale("log")
    plt.hist(data, bins=bins, color="#3b6fb6", edgecolor="white")
    if log:
        plt.yscale("log")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=130); plt.close()


def plot_trajectories(trip_ids):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, tid in zip(axes.ravel(), trip_ids):
        trip = load_trip(partition_dir("train") / tid)
        proj = LocalProjection.for_points(trip.track[:, :2])
        P = proj.to_xy(trip.track[:, :2])
        sc = ax.scatter(P[:, 0], P[:, 1], c=trip.track[:, 2] - trip.track[0, 2], s=3, cmap="viridis")
        ax.set_title(f"train {tid} ({len(P)} pts)"); ax.set_aspect("equal")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        fig.colorbar(sc, ax=ax, label="time (s)")
    fig.suptitle("GPS trajectories in local projected metres (partition: train)")
    fig.tight_layout(); fig.savefig(OUT / "trajectories_train.png", dpi=120); plt.close(fig)


def plot_network(tid, margin=300):
    d = partition_dir("train") / tid
    trip = load_trip(d); route = load_route(d)
    proj = LocalProjection.for_points(trip.track[:, :2])
    g = RoadGraph(trip.nodes, trip.arcs, proj)
    P = proj.to_xy(trip.track[:, :2])
    lo, hi = P.min(0) - margin, P.max(0) + margin
    inside = (((g.A >= lo) & (g.A <= hi)).all(1)) | (((g.B >= lo) & (g.B <= hi)).all(1))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(np.stack([g.A[inside], g.B[inside]], 1), colors="#b0b0b0", linewidths=0.7,
                                     label="road network"))
    ax.add_collection(LineCollection(np.stack([g.A[route], g.B[route]], 1), colors="#1a9e3a", linewidths=2.5,
                                     label="reference route"))
    ax.scatter(P[:, 0], P[:, 1], s=2, c="#1f4fd1", label="GPS", zorder=3)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.legend(loc="upper right"); ax.set_title(f"Road network around train trip {tid} "
                                               f"({inside.sum()} of {g.n_arcs} arcs shown)")
    fig.tight_layout(); fig.savefig(OUT / f"network_{tid}.png", dpi=130); plt.close(fig)


def folium_map(partition, tid):
    d = partition_dir(partition) / tid
    trip = load_trip(d); route = load_route(d)
    m = viz.base_map(trip.track)
    viz.add_network(m, trip.nodes, trip.arcs, trip.track)
    if route is not None:
        viz.add_route(m, trip.nodes, trip.arcs, route, "Reference route", viz.COLORS["reference"])
    viz.add_gps(m, trip.track)
    viz.add_legend(m, [("GPS observations", viz.COLORS["gps"]), ("Road network", viz.COLORS["network"]),
                       ("Reference route", viz.COLORS["reference"])], title=f"{partition} trip {tid}")
    viz.finish(m).save(str(OUT / "maps" / f"eda_{partition}_{tid}.html"))


def main():
    t0 = time.time()
    jobs = [(p, t) for p in ("train", "student_test") for t in list_trips(partition_dir(p))]
    print(f"Analysing {len(jobs)} trips ...")
    res = Parallel(n_jobs=8)(delayed(analyse_trip)(p, t) for p, t in jobs)
    inv = pd.DataFrame([r[0] for r in res]); stats = pd.DataFrame([r[1] for r in res])
    miss = pd.DataFrame([m for r in res for m in r[2]])
    inv.to_csv(OUT / "inventory.csv", index=False)
    stats.to_csv(OUT / "trip_stats.csv", index=False)

    miss_summary = miss.groupby(["partition", "file"]).agg(
        trips_present=("present", "sum"), total_rows=("rows", "sum"),
        missing_cells=("missing_cells", "sum"), bad_shape_files=("bad_rows", "sum")).reset_index()
    miss_summary.to_csv(OUT / "missing_values.csv", index=False)

    num = stats.select_dtypes("number").columns
    summ = stats.groupby("partition")[num].agg(["count", "mean", "median", "min", "max"]).T
    summ.to_csv(OUT / "partition_summary.csv")

    tr_mask = (stats.partition == "train").to_numpy()
    tr_res = [r[3] for r, m in zip(res, tr_mask) if m]
    dts = np.concatenate([x[0] for x in tr_res]); spd = np.concatenate([x[1] for x in tr_res])
    dist = np.concatenate([x[2] for x in tr_res if x[2] is not None])
    st = stats[tr_mask]
    plot_hist(dts, "Sampling interval between GPS points (train)", "dt (s)", "dt_hist_train.png", log=True, logx=True)
    plot_hist(spd[spd < 60], "Implied speed between consecutive points (train)", "speed (m/s)", "speed_hist_train.png", log=True)
    plot_hist(st.clean_points.to_numpy(), "GPS observations per trip (train)", "points", "points_per_trip_train.png", bins=30)
    plot_hist(np.minimum(dist, 100), "Distance GPS point -> nearest reference-route arc (train, clipped at 100 m)",
              "distance (m)", "gps_route_dist_train.png", log=True)
    plot_hist(np.log10(st.n_arcs.to_numpy()), "Road-network size per trip (train)", "log10(#arcs)", "network_size_train.png", bins=30)
    plot_hist(st.largest_scc_frac.to_numpy(), "Fraction of nodes in largest strongly-connected component (train)",
              "fraction", "scc_frac_train.png", bins=30)

    ids = st.sort_values("clean_points").trip_id.to_numpy()
    sample = list(ids[np.linspace(0, len(ids) - 1, 6).astype(int)])
    plot_trajectories(sample)
    for tid in sample[:2]:
        plot_network(tid)
    worst = st.sort_values("gps_route_dist_p95").trip_id.iloc[-1]
    for p, tid in [("train", sample[2]), ("train", worst), ("student_test", stats[~tr_mask].trip_id.iloc[0])]:
        folium_map(p, tid)

    # ---------------------------------------------------------------- data quality summary
    lines = ["# Data-quality summary (auto-generated by scripts/01_eda.py)", ""]
    for part, g in stats.groupby("partition"):
        lines += [f"## {part} ({len(g)} trips)",
                  f"- GPS points: total {g.clean_points.sum()}, per trip min {g.clean_points.min()} / median {g.clean_points.median():.0f} / max {g.clean_points.max()}",
                  f"- Rows dropped while cleaning: non-finite {g.clean_dropped_nonfinite.sum()}, out-of-range {g.clean_dropped_out_of_range.sum()}, duplicate timestamps {g.clean_dropped_duplicate_time.sum()}; unsorted tracks: {(~g.clean_was_sorted).sum()}",
                  f"- Sampling: median dt {g.dt_median.median():.2f}s; trips with a gap > 30 s: {(g.gaps_gt_30s > 0).sum()}; largest gap {g.dt_max.max():.0f}s",
                  f"- Speed outliers (>{CFG['cleaning']['max_speed_mps']} m/s): {g.n_speed_outliers.sum()} steps in {(g.n_speed_outliers > 0).sum()} trips",
                  f"- Road network: nodes {g.n_nodes.min()}-{g.n_nodes.max()}, arcs {g.n_arcs.min()}-{g.n_arcs.max()}; two-way fraction median {g.frac_two_way.median():.2f}",
                  f"- Self-loops {g.n_self_loops.sum()}, duplicate arcs {g.n_duplicate_arcs.sum()}, zero-length arcs {g.n_zero_length_arcs.sum()}, isolated nodes {g.isolated_nodes.sum()}",
                  f"- Weakly connected components per network: median {g.weak_components.median():.0f}, max {g.weak_components.max()}; largest SCC fraction min {g.largest_scc_frac.min():.3f}",
                  f"- Track inside node bounding box: {g.node_bbox_contains_track.sum()}/{len(g)} trips",
                  f"- Max projection error vs haversine (single step): {g.max_proj_error_m.max():.4f} m"]
        if "route_len" in g:
            lines += [f"- Reference route: arcs per trip {g.route_len.min()}-{g.route_len.max()}, invalid ids {g.route_invalid_ids.sum()}, contiguity min {g.route_contiguity.min():.3f}",
                      f"- GPS points per route arc: median {g.points_per_route_arc.median():.1f} (route is NOT one row per GPS point)",
                      f"- GPS->route distance: median of medians {g.gps_route_dist_median.median():.1f} m, worst p95 {g.gps_route_dist_p95.max():.1f} m; trips with >5% points beyond {CFG['labels']['max_match_dist_m']} m: {(g.frac_points_gt_40m > 0.05).sum()}"]
        lines.append("")
    (OUT / "data_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"EDA done in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
