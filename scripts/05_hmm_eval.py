"""Phase 5 - HMM / Viterbi route reconstruction: decoder tuning and classifier-only vs HMM comparison.

Probabilities used
  train        : out-of-fold probabilities of the selected (tuned) XGBoost from scripts/04_bayes_opt.py
                 (each trip scored by a model that never saw it) -> decoder tuning without leakage.
  student_test : probabilities of the tuned XGBoost refit on all 80 train trips.

Step 1  grid search of decoder parameters (beta, U-turn penalty, emission weight, gap threshold) on
        all 80 train trips, objective = pooled observation-level Top-1 on matched points.
Step 2  full evaluation (train OOF and student_test) of
          classifier-only : argmax of the classifier for each point
          classifier+HMM  : Viterbi with default parameters, and with tuned parameters
        metrics: Top-1, strict accuracy, direct / plausible continuity, chain breaks,
                 route precision / recall / F1 (arc set and length weighted) on the stitched route.

Outputs: results/hmm/decoder_grid.csv, best_params.json, comparison.csv, per_trip_<partition>.csv,
         plots; cache/hmm_predictions_<partition>.pkl (used for the maps)
"""
import _bootstrap  # noqa: F401

import argparse
import itertools
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import load_config, partition_dir, path
from pathsnap.features import prepare_trip
from pathsnap.hmm import DEFAULT_PARAMS, build_lattice, emission_logp, stitch_route, transition_stats, viterbi
from pathsnap.io import load_route, load_trip
from pathsnap.metrics import argmax_per_group, route_prf
from pathsnap.routing import Router

CFG = load_config()
OUT = path("results") / "hmm"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = path("cache")
CHOSEN = json.load(open(path("results") / "candidates" / "chosen.json"))
GRID = dict(beta=[3, 5, 8, 12, 20, 40], uturn_m=[0, 25, 75, 200], w_e=[0.5, 1.0, 2.0], gap_s=[30, 60, 120])


def trip_setup(partition, tid, rows_arc, p):
    td = prepare_trip(load_trip(partition_dir(partition) / tid), CHOSEN["radius_m"], CHOSEN["k"],
                      CFG["candidates"]["densify_step_m"])
    assert np.array_equal(td.cands.arc, rows_arc), f"candidate order mismatch in {tid}"
    router = Router(td.g)
    lat = build_lattice(td.g, router, td.P, td.t, td.cands.obs, td.cands.arc, td.cands.u)
    em = emission_logp(p, lat.starts)
    return td, router, lat, em


def grid_worker(partition, tid, rows_arc, p, labels, combos):
    td, _, lat, em = trip_setup(partition, tid, rows_arc, p)
    m = labels >= 0
    out = []
    for c in combos:
        ch, br = viterbi(lat, em, c)
        arcs = np.where(ch >= 0, lat.arc[np.maximum(ch, 0)], -1)
        out.append(int((arcs[m] == labels[m]).sum()))
    return tid, int(m.sum()), out


def full_worker(partition, tid, rows_arc, p, labels, params_list):
    td, router, lat, em = trip_setup(partition, tid, rows_arc, p)
    route = load_route(partition_dir(partition) / tid)   # only for scoring, after decoding
    m = labels >= 0
    n = lat.n_obs
    res = []
    sel = argmax_per_group(td.cands.obs, p)
    clf_choice = np.full(n, -1, dtype=np.int64); clf_choice[td.cands.obs[sel]] = sel
    methods = [("classifier_only", clf_choice, 0, 0.0)]
    for name, prm in params_list:
        t0 = time.perf_counter(); ch, br = viterbi(lat, em, prm); dt = time.perf_counter() - t0
        methods.append((name, ch, br, dt))
    preds = {}
    for name, ch, br, dec_s in methods:
        arcs = np.where(ch >= 0, lat.arc[np.maximum(ch, 0)], -1)
        n_dir, n_pl, n_ch = transition_stats(lat, ch, td.g)
        stitched, unbridged = stitch_route(arcs, td.g, router)
        prf = route_prf(stitched, route, td.g.length)
        res.append(dict(partition=partition, trip_id=tid, method=name, n_points=n, n_matched=int(m.sum()),
                        correct=int((arcs[m] == labels[m]).sum()), n_assigned=int((arcs >= 0).sum()),
                        direct_ok=n_dir, plausible_ok=n_pl, changes=n_ch, chain_breaks=br,
                        unbridged_gaps=unbridged, stitched_len_km=td.g.length[stitched].sum() / 1000 if stitched else 0,
                        ref_len_km=td.g.length[route].sum() / 1000, decode_s=dec_s, **prf))
        preds[name] = dict(point_arcs=arcs, route=stitched)
    return res, (tid, preds)


def load(partition, model):
    df = pd.read_pickle(CACHE / f"features_{partition}.pkl")
    labels = pickle.load(open(CACHE / f"labels_{partition}.pkl", "rb"))
    src = "oof" if partition == "train" else "st_proba"
    p = np.load(CACHE / src / f"{model}.npy").astype(float)
    tid = df.trip_id.astype(str).to_numpy()
    groups = {t: np.flatnonzero(tid == t) for t in sorted(labels)}
    return df, labels, p, groups


def summarise(df):
    g = df.groupby(["partition", "method"])
    s = g[["n_points", "n_matched", "correct", "direct_ok", "plausible_ok", "changes", "chain_breaks", "unbridged_gaps"]].sum()
    s["top1"] = s.correct / s.n_matched
    s["strict_accuracy"] = s.correct / s.n_points
    s["direct_continuity"] = s.direct_ok / s.changes
    s["plausible_continuity"] = s.plausible_ok / s.changes
    for c in ["route_precision", "route_recall", "route_f1", "route_len_precision", "route_len_recall", "route_len_f1"]:
        s[c + "_mean"] = g[c].mean()
    s["trips_fully_continuous"] = g.unbridged_gaps.apply(lambda x: (x == 0).mean())
    s["decode_ms_per_1k_points"] = 1e6 * g.decode_s.sum() / s.n_points
    return s.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost_tuned")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default="hmm", help="output folder under results/ (and cache suffix)")
    args = ap.parse_args()
    global OUT
    OUT = path("results") / args.out; OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.out == "hmm" else f"_{args.out}"
    if not (CACHE / "oof" / f"{args.model}.npy").exists():
        raise SystemExit(f"missing OOF probabilities for {args.model}; run scripts/04_bayes_opt.py first")
    t0 = time.time()

    # ---------------------------------------------------------------- step 1: decoder grid on train
    df, labels, p, groups = load("train", args.model)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = Parallel(n_jobs=args.n_jobs)(delayed(grid_worker)("train", t, df.arc.to_numpy()[r], p[r], labels[t], combos)
                                       for t, r in groups.items())
    tot_m = sum(r[1] for r in res)
    acc = np.sum([r[2] for r in res], axis=0) / tot_m
    grid = pd.DataFrame(combos); grid["train_oof_top1"] = acc
    grid.sort_values("train_oof_top1", ascending=False).to_csv(OUT / "decoder_grid.csv", index=False)
    best = combos[int(np.argmax(acc))]
    json.dump({"params": best, "train_oof_top1": float(acc.max()), "default_params": DEFAULT_PARAMS,
               "grid": GRID, "objective": "pooled Top-1 on matched points, train OOF probabilities",
               "model": args.model}, open(OUT / "best_params.json", "w"), indent=2)
    print(f"grid done ({time.time() - t0:.0f}s) best={best} top1={acc.max():.4f}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a_, (par, other) in zip(ax, [("beta", "w_e"), ("uturn_m", "w_e"), ("gap_s", "w_e")]):
        fixed = {k: v for k, v in best.items() if k not in (par, other)}
        sub = grid.copy()
        for k, v in fixed.items():
            sub = sub[sub[k] == v]
        for w, s in sub.groupby(other):
            a_.plot(s[par], s.train_oof_top1, "o-", label=f"{other}={w}")
        a_.set(xlabel=par, ylabel="train OOF Top-1", title=f"Sensitivity to {par}")
        if par == "beta": a_.set_xscale("log")
        a_.legend(); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "decoder_sensitivity.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- step 2: full comparison
    params_list = [("hmm_default", DEFAULT_PARAMS), ("hmm_tuned", best)]
    rows = []
    for partition in ("train", "student_test"):
        df, labels, p, groups = load(partition, args.model)
        out = Parallel(n_jobs=args.n_jobs)(delayed(full_worker)(partition, t, df.arc.to_numpy()[r], p[r], labels[t], params_list)
                                           for t, r in groups.items())
        rows += [x for o in out for x in o[0]]
        pickle.dump(dict(o[1] for o in out), open(CACHE / f"hmm_predictions_{partition}{suffix}.pkl", "wb"))
        print(f"{partition} evaluated ({time.time() - t0:.0f}s)", flush=True)
    per = pd.DataFrame(rows)
    per["top1"] = per.correct / per.n_matched.clip(lower=1)
    per["plausible_continuity"] = per.plausible_ok / per.changes.clip(lower=1)
    for part in per.partition.unique():
        per[per.partition == part].to_csv(OUT / f"per_trip_{part}.csv", index=False)
    comp = summarise(per)
    comp.to_csv(OUT / "comparison.csv", index=False)
    show = ["partition", "method", "top1", "strict_accuracy", "direct_continuity", "plausible_continuity",
            "chain_breaks", "unbridged_gaps", "route_f1_mean", "route_len_f1_mean", "decode_ms_per_1k_points"]
    print(comp[show].round(4).to_string(index=False))

    # plots
    st = comp[comp.partition == "student_test"].set_index("method").loc[["classifier_only", "hmm_default", "hmm_tuned"]]
    metrics = ["top1", "direct_continuity", "plausible_continuity", "route_f1_mean", "route_len_f1_mean"]
    fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(metrics))
    for i, (mth, r) in enumerate(st.iterrows()):
        ax.bar(x + i * 0.27, [r[c] for c in metrics], 0.27, label=mth)
    ax.set_xticks(x + 0.27); ax.set_xticklabels(["Top-1", "direct continuity", "plausible continuity", "route F1", "route F1 (length)"])
    ax.set_ylim(0.5, 1.0); ax.legend(loc="lower right", framealpha=0.95); ax.grid(axis="y", alpha=.3)
    ax.set_title("Classifier-only vs classifier + HMM/Viterbi (student_test, 10 trips)")
    fig.tight_layout(); fig.savefig(OUT / "clf_vs_hmm_student_test.png", dpi=130); plt.close(fig)

    for part in per.partition.unique():
        pv = per[per.partition == part].pivot(index="trip_id", columns="method", values="top1")
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(pv.classifier_only, pv.hmm_tuned, s=20)
        lo = min(pv.min().min(), 0.5); ax.plot([lo, 1], [lo, 1], "k--", lw=1)
        ax.set(xlabel="classifier-only Top-1", ylabel="classifier + HMM Top-1",
               title=f"Per-trip Top-1 ({part}); above the line = HMM helps")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / f"per_trip_scatter_{part}.png", dpi=130); plt.close(fig)
        pv["delta"] = pv.hmm_tuned - pv.classifier_only
        pv.sort_values("delta").to_csv(OUT / f"per_trip_delta_{part}.csv")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
