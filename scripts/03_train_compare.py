"""Phase 3 - Train and compare candidate classifiers.

Protocol
  * 5-fold GroupKFold on the 80 TRAIN trips (a trip never appears in both fit and validation folds).
    Fold assignment saved to results/models/folds.csv.
  * Models are fit only on rows of matched GPS points (label != -1) of the training folds.
    Predictions are made for ALL candidate rows of the held-out trips (out-of-fold probabilities are
    reused later to tune the HMM without leakage); metrics use matched points only.
  * Each model is then refit on all 80 train trips and evaluated on the 10 STUDENT_TEST trips.
  * Two non-learned baselines (nearest arc; distance + heading rule) are reported for reference.

Usage
  python scripts/03_train_compare.py                 # all models
  python scripts/03_train_compare.py --models lightgbm mlp
  python scripts/03_train_compare.py --report-only   # rebuild comparison table / plots

Outputs
  results/models/<model>/metrics.json, cv_folds.csv, per_trip_student_test.csv, confusion_matrix.png,
                         training_history.png (only for models with real epoch/iteration histories)
  results/models/comparison.csv and comparison plots
  cache/oof/<model>.npy, cache/st_proba/<model>.npy, cache/models/<model>.joblib
"""
import _bootstrap  # noqa: F401

import argparse
import json
import pickle
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config, partition_dir, path
from pathsnap.features import FEATURE_COLS
from pathsnap.io import _read
from pathsnap.metrics import argmax_per_group, candidate_metrics, obs_key, route_continuity
from pathsnap.models import get_models

CFG = load_config()
SEED = CFG["seed"]
OUT = path("results") / "models"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = path("cache")
for sub in ("oof", "st_proba", "models"):
    (CACHE / sub).mkdir(exist_ok=True)
BASELINES = {
    "baseline_nearest": ("Non-learned baseline", "pick the nearest candidate arc (ties by arc id)"),
    "baseline_dist_heading": ("Non-learned baseline", "score = -distance + 10 * cos(heading difference)"),
}

# Keep the default experiment focused on the five selected model families.
# Baselines and any other model can still be requested explicitly with --models.
SELECTED_MODELS = ("lightgbm", "xgboost", "random_forest", "hist_gb", "mlp")


# --------------------------------------------------------------------------- data
def load_partition(partition):
    df = pd.read_pickle(CACHE / f"features_{partition}.pkl")
    labels = pickle.load(open(CACHE / f"labels_{partition}.pkl", "rb"))
    trips = sorted(labels)
    df["trip_id"] = pd.Categorical(df.trip_id.astype(str), categories=trips)
    tcode = df.trip_id.cat.codes.to_numpy()
    key = obs_key(tcode, df.obs.to_numpy())
    arcs = {t: _read(partition_dir(partition) / t / f"{t}.arcs", np.int64) for t in trips}
    return df, labels, trips, tcode, key, arcs


def evaluate(df, key, p, labels, arcs, trip_subset, yhat=None):
    """All metrics for predictions p on the rows of df that belong to trip_subset."""
    rows = np.flatnonzero(df.trip_id.isin(trip_subset).to_numpy())
    sub = df.iloc[rows]
    m = (sub.label_arc >= 0).to_numpy()
    y = sub.y.to_numpy()
    pp = p[rows]
    cm = candidate_metrics(y[m], pp[m] if yhat is None else None, None if yhat is None else yhat[rows][m])
    sel = argmax_per_group(key[rows], pp)
    ch = sub.iloc[sel]
    n_matched = sum(int((labels[t] >= 0).sum()) for t in trip_subset)
    n_points = sum(len(labels[t]) for t in trip_subset)
    correct = int(ch.y[(ch.label_arc >= 0).to_numpy()].sum())
    valid = total = 0
    per_trip = []
    for t, g in ch.groupby("trip_id", observed=True):
        g = g.sort_values("obs")
        v, tt = route_continuity(g.arc.to_numpy(), arcs[t])
        valid += v; total += tt
        nm = int((labels[t] >= 0).sum())
        per_trip.append(dict(trip_id=t, n_points=len(labels[t]), n_matched=nm,
                             top1=int(g.y[(g.label_arc >= 0).to_numpy()].sum()) / max(nm, 1),
                             continuity=v / tt if tt else 1.0, transitions=tt))
    cm.update(top1=correct / max(n_matched, 1), strict_accuracy=correct / max(n_points, 1),
              route_continuity=valid / total if total else 1.0, n_matched_points=n_matched)
    return cm, pd.DataFrame(per_trip)


def subsample(df_fit, max_obs, rng):
    if max_obs is None:
        return df_fit
    keys = np.unique(df_fit["_key"].to_numpy())
    if len(keys) <= max_obs:
        return df_fit
    keep = rng.choice(keys, max_obs, replace=False)
    return df_fit[df_fit["_key"].isin(keep)]


# --------------------------------------------------------------------------- training
def run_model(name, spec, tr, tr_key, tr_lab, tr_arcs, st, st_key, st_lab, st_arcs, folds):
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tr = tr.assign(_key=tr_key)
    oof = np.full(len(tr), np.nan)
    fold_rows = []
    for k in range(CFG["cv"]["n_folds"]):
        val_trips = folds[folds.fold == k].trip_id.tolist()
        is_val = tr.trip_id.isin(val_trips).to_numpy()
        fit_df = subsample(tr[~is_val & (tr.label_arc >= 0).to_numpy()], spec.train_subsample_obs, rng)
        est = spec.factory()
        t0 = time.perf_counter(); est.fit(fit_df[FEATURE_COLS].to_numpy(), fit_df.y.to_numpy()); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); oof[is_val] = est.predict_proba(tr.loc[is_val, FEATURE_COLS].to_numpy())[:, 1]
        t_pred = time.perf_counter() - t0
        met, _ = evaluate(tr, tr_key, oof, tr_lab, tr_arcs, val_trips)
        met.pop("confusion")
        n_val_pts = sum(len(tr_lab[t]) for t in val_trips)
        # training-fold score (on a sample) to diagnose over/under-fitting
        fit_trips = sorted(set(fit_df.trip_id.astype(str)))
        fit_trips = sorted(rng.choice(fit_trips, min(12, len(fit_trips)), replace=False))  # 12-trip sample
        p_fit = np.full(len(tr), np.nan)
        is_fit = tr.trip_id.isin(fit_trips).to_numpy()
        p_fit[is_fit] = est.predict_proba(tr.loc[is_fit, FEATURE_COLS].to_numpy())[:, 1]
        met_fit, _ = evaluate(tr, tr_key, p_fit, tr_lab, tr_arcs, fit_trips)
        fold_rows.append(dict(fold=k, n_val_trips=len(val_trips), fit_rows=len(fit_df), fit_time_s=t_fit,
                              pred_ms_per_1k_points=1000 * t_pred / n_val_pts * 1000,
                              train_top1=met_fit["top1"], train_macro_f1=met_fit["macro_f1"],
                              train_log_loss=met_fit["log_loss"], **met))
        print(f"  [{name}] fold {k}: top1={met['top1']:.4f} macroF1={met['macro_f1']:.4f} "
              f"logloss={met['log_loss']:.4f} fit={t_fit:.1f}s", flush=True)
    cv = pd.DataFrame(fold_rows); cv.to_csv(out / "cv_folds.csv", index=False)
    np.save(CACHE / "oof" / f"{name}.npy", oof.astype(np.float32))

    # refit on all train trips, evaluate on student_test
    fit_df = subsample(tr[(tr.label_arc >= 0).to_numpy()], spec.train_subsample_obs, rng)
    est = spec.factory()
    X, y = fit_df[FEATURE_COLS].to_numpy(), fit_df.y.to_numpy()
    st_m = (st.label_arc >= 0).to_numpy()
    Xs, ys = st.loc[st_m, FEATURE_COLS].to_numpy(), st.loc[st_m, "y"].to_numpy()
    t0 = time.perf_counter()
    if name == "lightgbm":
        est.fit(X, y, eval_set=[(X, y), (Xs, ys)], eval_names=["train", "student_test"], eval_metric="binary_logloss")
    elif name == "xgboost":
        est.fit(X, y, eval_set=[(X, y), (Xs, ys)], verbose=False)
    else:
        est.fit(X, y)
    t_fit_full = time.perf_counter() - t0
    t0 = time.perf_counter(); p_st = est.predict_proba(st[FEATURE_COLS].to_numpy())[:, 1]; t_pred = time.perf_counter() - t0
    np.save(CACHE / "st_proba" / f"{name}.npy", p_st.astype(np.float32))
    joblib.dump(est, CACHE / "models" / f"{name}.joblib")
    st_trips = sorted(st_lab)
    met_st, per_trip = evaluate(st, st_key, p_st, st_lab, st_arcs, st_trips)
    per_trip.to_csv(out / "per_trip_student_test.csv", index=False)
    plot_confusion(met_st["confusion"], name, out)
    plot_history(name, est, out)

    n_st_pts = sum(len(v) for v in st_lab.values())
    res = dict(model=name, family=spec.family, config=spec.config, notes=spec.notes,
               train_subsample_obs=spec.train_subsample_obs,
               cv={c: [float(cv[c].mean()), float(cv[c].std())] for c in
                   ["top1", "macro_f1", "log_loss", "precision", "recall", "accuracy", "route_continuity",
                    "train_top1", "train_macro_f1", "train_log_loss", "fit_time_s"]},
               student_test=met_st, fit_time_full_s=t_fit_full,
               inference_ms_per_1k_points=1000 * t_pred / n_st_pts * 1000, seed=SEED)
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    print(f"  [{name}] student_test: top1={met_st['top1']:.4f} macroF1={met_st['macro_f1']:.4f} "
          f"logloss={met_st['log_loss']:.4f} continuity={met_st['route_continuity']:.4f}", flush=True)


def run_baseline(name, tr, tr_key, tr_lab, tr_arcs, st, st_key, st_lab, st_arcs, folds):
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)

    def score(df):
        if name == "baseline_nearest":
            return -df["d"].to_numpy() - 1e-6 * df["rank"].to_numpy()
        return -df["d"].to_numpy() + 10.0 * df["cos_dh"].to_numpy()

    def onehot(df, key):
        s = score(df); yhat = np.zeros(len(df), int); yhat[argmax_per_group(key, s)] = 1
        return s, yhat

    s_tr, yhat_tr = onehot(tr, tr_key)
    fold_rows = []
    for k in range(CFG["cv"]["n_folds"]):
        val = folds[folds.fold == k].trip_id.tolist()
        met, _ = evaluate(tr, tr_key, s_tr, tr_lab, tr_arcs, val, yhat=yhat_tr)
        met.pop("confusion"); fold_rows.append(dict(fold=k, **met))
    cv = pd.DataFrame(fold_rows); cv.to_csv(out / "cv_folds.csv", index=False)
    t0 = time.perf_counter(); s_st, yhat_st = onehot(st, st_key); t_pred = time.perf_counter() - t0
    met_st, per_trip = evaluate(st, st_key, s_st, st_lab, st_arcs, sorted(st_lab), yhat=yhat_st)
    per_trip.to_csv(out / "per_trip_student_test.csv", index=False)
    n_st_pts = sum(len(v) for v in st_lab.values())
    res = dict(model=name, family=BASELINES[name][0], config={"rule": BASELINES[name][1]},
               notes="no training; probability not defined -> log loss N/A",
               cv={c: [float(cv[c].mean()), float(cv[c].std())] for c in
                   ["top1", "macro_f1", "precision", "recall", "accuracy", "route_continuity"]},
               student_test=met_st, fit_time_full_s=0.0,
               inference_ms_per_1k_points=1000 * t_pred / n_st_pts * 1000, seed=SEED)
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    print(f"  [{name}] student_test top1={met_st['top1']:.4f}")


# --------------------------------------------------------------------------- plots
def plot_confusion(cm, name, out):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: other arc", "pred: true arc"]); ax.set_yticklabels(["other arc", "true arc"])
    ax.set_title(f"{name} - student_test candidates")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)


def plot_history(name, est, out):
    """Only for models that really record an iteration/epoch history."""
    if name == "lightgbm":
        h = est.evals_result_
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for k in h: ax.plot(h[k]["binary_logloss"], label=k)
        ax.set(xlabel="boosting iteration", ylabel="log loss", title="LightGBM training history")
    elif name == "xgboost":
        h = est.evals_result()
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for k, lab in zip(h, ["train", "student_test"]): ax.plot(h[k]["logloss"], label=lab)
        ax.set(xlabel="boosting iteration", ylabel="log loss", title="XGBoost training history")
    elif name == "mlp":
        mlp = est[-1]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(mlp.loss_curve_, label="training loss (log loss)")
        ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="C1", label="validation accuracy (10% hold-out)")
        ax2.set_ylabel("validation accuracy"); ax2.legend(loc="center right")
        ax.set(xlabel="epoch", ylabel="training loss", title="MLP training history")
    else:
        return
    ax.legend(loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "training_history.png", dpi=130); plt.close(fig)


def build_report():
    rows = []
    for p in sorted(OUT.glob("*/metrics.json")):
        r = json.load(open(p))
        st = r["student_test"]; cv = r["cv"]
        rows.append(dict(model=r["model"], family=r["family"],
                         key_config="; ".join(f"{k}={v}" for k, v in r["config"].items()),
                         cv_macro_f1=cv["macro_f1"][0], cv_macro_f1_std=cv["macro_f1"][1],
                         cv_top1=cv["top1"][0], cv_top1_std=cv["top1"][1],
                         cv_log_loss=cv.get("log_loss", [np.nan])[0],
                         cv_route_continuity=cv["route_continuity"][0],
                         train_top1=cv.get("train_top1", [np.nan])[0],
                         st_top1=st["top1"], st_strict_accuracy=st["strict_accuracy"],
                         st_macro_f1=st["macro_f1"], st_precision=st["precision"], st_recall=st["recall"],
                         st_accuracy=st["accuracy"], st_log_loss=st["log_loss"],
                         st_route_continuity=st["route_continuity"],
                         fit_time_s=r["fit_time_full_s"], inference_ms_per_1k_points=r["inference_ms_per_1k_points"],
                         notes=r.get("notes", "")))
    comp = pd.DataFrame(rows).sort_values("cv_top1", ascending=False)
    comp.to_csv(OUT / "comparison.csv", index=False)
    print(comp[["model", "cv_top1", "cv_macro_f1", "cv_log_loss", "st_top1", "st_macro_f1", "st_log_loss",
                "st_route_continuity", "fit_time_s", "inference_ms_per_1k_points"]].to_string(index=False))

    learned = comp[~comp.model.str.startswith("baseline")]
    allm = comp
    def bar(df, cols, labels, fname, title, ylabel, err=None, ylim=None):
        x = np.arange(len(df)); w = 0.8 / len(cols)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, (c, lab) in enumerate(zip(cols, labels)):
            ax.bar(x + i * w, df[c], w, label=lab, yerr=df[err[i]] if err else None, capsize=3)
        ax.set_xticks(x + w * (len(cols) - 1) / 2); ax.set_xticklabels(df.model, rotation=25, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.3)
        if ylim: ax.set_ylim(*ylim)
        if len(cols) > 1: ax.legend()
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    bar(allm, ["cv_macro_f1"], ["CV macro-F1"], "cv_macro_f1.png",
        "5-fold GroupKFold macro-F1 (train, mean +- std over folds)", "macro-F1", err=["cv_macro_f1_std"], ylim=(0.4, 1))
    bar(allm, ["cv_top1"], ["CV Top-1"], "cv_top1.png",
        "5-fold GroupKFold observation-level Top-1 (train)", "Top-1 accuracy", err=["cv_top1_std"], ylim=(0.4, 1))
    bar(allm, ["st_top1", "st_macro_f1"], ["Top-1 accuracy", "macro-F1"], "student_test_top1_f1.png",
        "Student-test performance (10 trips)", "score", ylim=(0.4, 1))
    bar(learned, ["cv_log_loss", "st_log_loss"], ["CV", "student_test"], "log_loss.png",
        "Log loss (lower is better; N/A for baselines)", "log loss")
    bar(allm, ["inference_ms_per_1k_points"], ["inference"], "inference_time.png",
        "Classifier inference time on student_test", "ms per 1000 GPS points")
    bar(allm, ["train_top1", "cv_top1", "st_top1"], ["train folds", "CV held-out", "student_test"],
        "generalisation_top1.png", "Top-1: training fit vs held-out (over/under-fitting check)", "Top-1", ylim=(0.4, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        tr, tr_lab, tr_trips, _, tr_key, tr_arcs = load_partition("train")
        st, st_lab, _, _, st_key, st_arcs = load_partition("student_test")
        m = (tr.label_arc >= 0).to_numpy()
        gkf = GroupKFold(n_splits=CFG["cv"]["n_folds"])
        fold_of = {}
        for k, (_, vi) in enumerate(gkf.split(tr[m], groups=tr.trip_id[m].astype(str))):
            for t in tr.trip_id[m].iloc[vi].astype(str).unique():
                fold_of[t] = k
        folds = pd.DataFrame({"trip_id": tr_trips, "fold": [fold_of[t] for t in tr_trips]})
        folds.to_csv(OUT / "folds.csv", index=False)
        specs = get_models(SEED)
        names = args.models or list(SELECTED_MODELS)
        for name in names:
            t0 = time.time(); print(f"=== {name}", flush=True)
            if name in BASELINES:
                run_baseline(name, tr, tr_key, tr_lab, tr_arcs, st, st_key, st_lab, st_arcs, folds)
            else:
                run_model(name, specs[name], tr, tr_key, tr_lab, tr_arcs, st, st_key, st_lab, st_arcs, folds)
            print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    build_report()


if __name__ == "__main__":
    main()
