"""Phase 2b - Build the candidate feature dataset for train and student_test.

Requires scripts/02a_candidate_grid.py (labels cache + chosen R, K).

Each row = (trip, GPS point, candidate arc) with the features of pathsnap.features.FEATURE_COLS,
plus:
  label_arc : derived reference arc of the GPS point (-1 = unmatched)
  y         : 1 if the candidate is the reference arc else 0

Outputs:
  cache/features_<partition>.pkl
  results/candidates/miss_rate_by_partition.csv
  results/features/feature_summary_train.csv, feature_corr_train.png
"""
import _bootstrap  # noqa: F401

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
from pathsnap.features import FEATURE_COLS, build_features, prepare_trip
from pathsnap.io import list_trips, load_trip

CFG = load_config()
CHOSEN = json.load(open(path("results") / "candidates" / "chosen.json"))
OUT_F = path("results") / "features"; OUT_F.mkdir(parents=True, exist_ok=True)


def process(partition, tid, labels):
    trip = load_trip(partition_dir(partition) / tid)
    td = prepare_trip(trip, CHOSEN["radius_m"], CHOSEN["k"], CFG["candidates"]["densify_step_m"])
    df = build_features(td)
    df.insert(0, "trip_id", tid)
    df["label_arc"] = labels[df.obs.to_numpy()]
    df["y"] = (df.arc.to_numpy() == df.label_arc.to_numpy()).astype(np.int8)
    m = labels >= 0
    hit = np.zeros(len(labels), bool)
    hit[df.obs.to_numpy()[df.y.to_numpy() == 1]] = True
    stats = dict(trip_id=tid, n_points=len(labels), n_matched=int(m.sum()), n_hit=int(hit[m].sum()),
                 n_no_cand=int((np.bincount(df.obs, minlength=len(labels)) == 0).sum()))
    return df, stats


def main():
    t0 = time.time()
    miss_rows = []
    for partition in ("train", "student_test"):
        labels = pickle.load(open(path("cache") / f"labels_{partition}.pkl", "rb"))
        tids = list_trips(partition_dir(partition))
        res = Parallel(n_jobs=8)(delayed(process)(partition, t, labels[t]) for t in tids)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        df["trip_id"] = df.trip_id.astype("category")
        df.to_pickle(path("cache") / f"features_{partition}.pkl")
        st = pd.DataFrame([r[1] for r in res])
        tot = st.sum(numeric_only=True)
        miss_rows.append(dict(partition=partition, trips=len(st), points=int(tot.n_points), matched=int(tot.n_matched),
                              candidate_rows=len(df), cands_per_point=len(df) / tot.n_points,
                              miss_rate=1 - tot.n_hit / tot.n_matched, points_without_candidates=int(tot.n_no_cand),
                              positive_rate=float(df.y.mean())))
        print(f"{partition}: {len(df)} rows, {len(FEATURE_COLS)} features ({time.time() - t0:.0f}s)")
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(path("results") / "candidates" / "miss_rate_by_partition.csv", index=False)
    print(miss.to_string(index=False))

    tr = pd.read_pickle(path("cache") / "features_train.pkl")
    tr = tr[tr.label_arc >= 0]
    summ = tr.groupby("y")[FEATURE_COLS].mean().T
    summ.columns = ["mean_negative", "mean_positive"]
    summ.to_csv(OUT_F / "feature_summary_train.csv")
    corr = tr[FEATURE_COLS + ["y"]].sample(min(len(tr), 300_000), random_state=CFG["seed"]).corr()
    corr["y"].drop("y").sort_values().to_csv(OUT_F / "feature_target_corr_train.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im); ax.set_title("Feature correlation (train candidates, 300k-row sample)")
    fig.tight_layout(); fig.savefig(OUT_F / "feature_corr_train.png", dpi=120); plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
