"""Phase 4 - Bayesian hyperparameter optimisation of the selected model (XGBoost).

Why XGBoost: in scripts/03_train_compare.py it tied with LightGBM on 5-fold CV Top-1 (0.888 vs 0.889),
was best on every student-test metric (Top-1 0.899, macro-F1 0.937, log loss 0.082) and was ~4x faster
at inference than LightGBM, which matters for the API.

Search
  * Optuna TPE sampler (seed 42) + MedianPruner (prunes after >=2 folds once 8 trials are complete).
  * Objective: mean observation-level Top-1 accuracy over the SAME 5 GroupKFold folds of the 80 train
    trips used in Phase 3 (results/models/folds.csv). Macro-F1 and log loss are stored per trial.
  * The student_test and evaluator partitions are never used inside the objective.

After the search the best configuration is (1) re-run through CV to obtain out-of-fold
probabilities (used to tune the HMM without leakage), (2) refit on all 80 train trips and
(3) compared with the untuned XGBoost on identical metrics and partitions.

Outputs (results/optuna/): trials.csv, study.db, best_params.json, search_space.json,
optimisation_history.png, param_importance.png, tuned_vs_untuned.csv/.png, run_info.json
Cache: cache/models/xgboost_tuned.joblib, cache/oof/xgboost_tuned.npy, cache/st_proba/xgboost_tuned.npy
"""
import _bootstrap  # noqa: F401

import argparse
import importlib
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from pathsnap.config import load_config, path
from pathsnap.features import FEATURE_COLS
from pathsnap.metrics import argmax_per_group

tc = importlib.import_module("03_train_compare")   # reuse data loading + evaluation of Phase 3
CFG = load_config()
SEED = CFG["seed"]
OUT = path("results") / "optuna"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = path("cache")

SEARCH_SPACE = {
    "n_estimators": "int [150, 900] step 50",
    "learning_rate": "float [0.01, 0.3] log",
    "max_depth": "int [3, 11]",
    "min_child_weight": "float [1, 50] log",
    "subsample": "float [0.5, 1.0]",
    "colsample_bytree": "float [0.4, 1.0]",
    "gamma": "float [0, 5]",
    "reg_alpha": "float [1e-8, 10] log",
    "reg_lambda": "float [1e-3, 20] log",
    "scale_pos_weight": "float [1, 6]",
}
UNTUNED = dict(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8)


def make_model(params):
    return xgb.XGBClassifier(**params, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss")


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 900, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 11),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 6.0),
    )


def fast_fold_scores(y, label_arc, key, p, n_matched):
    from sklearn.metrics import f1_score, log_loss
    m = label_arc >= 0
    sel = argmax_per_group(key, p)
    top1 = y[sel][m[sel]].sum() / n_matched
    return top1, f1_score(y[m], (p[m] >= 0.5).astype(int), average="macro"), log_loss(y[m], np.clip(p[m], 1e-7, 1 - 1e-7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout-min", type=float, default=150)
    args = ap.parse_args()

    tr, tr_lab, tr_trips, _, tr_key, tr_arcs = tc.load_partition("train")
    st, st_lab, _, _, st_key, st_arcs = tc.load_partition("student_test")
    folds = pd.read_csv(path("results") / "models" / "folds.csv", dtype={"trip_id": str})
    X = tr[FEATURE_COLS].to_numpy(np.float32); y = tr.y.to_numpy(); lab = tr.label_arc.to_numpy()
    tid = tr.trip_id.astype(str).to_numpy()
    fold_masks = []
    for k in range(CFG["cv"]["n_folds"]):
        val = np.isin(tid, folds[folds.fold == k].trip_id)
        n_matched = sum(int((tr_lab[t] >= 0).sum()) for t in folds[folds.fold == k].trip_id)
        fold_masks.append((val, n_matched))

    def objective(trial):
        params = suggest(trial)
        tops, f1s, lls = [], [], []
        t0 = time.time()
        for k, (val, nm) in enumerate(fold_masks):
            fit = ~val & (lab >= 0)
            est = make_model(params).fit(X[fit], y[fit])
            p = est.predict_proba(X[val])[:, 1]
            t1, f1, ll = fast_fold_scores(y[val], lab[val], tr_key[val], p, nm)
            tops.append(t1); f1s.append(f1); lls.append(ll)
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(tops)))
        trial.set_user_attr("folds_done", len(tops))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(tops))

    storage = f"sqlite:///{(OUT / 'study.db').as_posix()}"
    study = optuna.create_study(study_name="xgboost_top1", direction="maximize", storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if not any(t.state == optuna.trial.TrialState.COMPLETE and t.params == {**UNTUNED} for t in study.trials):
        study.enqueue_trial({**UNTUNED, "min_child_weight": 1.0, "gamma": 0.0, "reg_alpha": 1e-8,
                             "reg_lambda": 1.0, "scale_pos_weight": 1.0})
    t_search = time.time()
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    remaining = max(args.trials - len(done), 0)
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=args.timeout_min * 60, gc_after_trial=True)
    search_s = time.time() - t_search

    df = study.trials_dataframe()
    df = df[df.state.isin(["COMPLETE", "PRUNED"])]   # drop a trial interrupted when the search was stopped
    df.to_csv(OUT / "trials.csv", index=False)
    if search_s < 1:   # search finished in an earlier run: wall time from the trial timestamps
        search_s = (pd.to_datetime(df.datetime_complete).max() - pd.to_datetime(df.datetime_start).min()).total_seconds()
    json.dump(SEARCH_SPACE, open(OUT / "search_space.json", "w"), indent=2)
    best = study.best_trial
    best_params = dict(best.params)
    json.dump({"best_cv_top1": best.value, **{f"user_{k}": v for k, v in best.user_attrs.items()},
               "params": best_params}, open(OUT / "best_params.json", "w"), indent=2)
    print("best trial", best.number, best.value, best_params, flush=True)

    # plots
    complete = df[df.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(complete.number, complete.value, s=18, label="completed trial")
    ax.plot(complete.number, complete.value.cummax(), color="C3", label="best so far")
    pr = df[df.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, [complete.value.min()] * len(pr), marker="x", color="grey", s=14, label="pruned")
    ax.set(xlabel="trial", ylabel="CV Top-1 (5-fold GroupKFold)", title="Optuna TPE optimisation history (XGBoost)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "optimisation_history.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#3b6fb6")
        ax.set(xlabel="importance (fANOVA)", title="Hyperparameter importance for CV Top-1")
        fig.tight_layout(); fig.savefig(OUT / "param_importance.png", dpi=130); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("importance failed:", e)

    # tuned model: OOF probabilities (same folds) and student_test evaluation
    oof = np.full(len(tr), np.nan, dtype=np.float32)
    cv_rows = []
    for k, (val, nm) in enumerate(fold_masks):
        fit = ~val & (lab >= 0)
        est = make_model(best_params).fit(X[fit], y[fit])
        oof[val] = est.predict_proba(X[val])[:, 1]
        met, _ = tc.evaluate(tr, tr_key, oof.astype(float), tr_lab, tr_arcs, folds[folds.fold == k].trip_id.tolist())
        met.pop("confusion"); cv_rows.append(met)
    np.save(CACHE / "oof" / "xgboost_tuned.npy", oof)
    cvt = pd.DataFrame(cv_rows)
    t0 = time.perf_counter()
    final = make_model(best_params).fit(X[lab >= 0], y[lab >= 0])
    fit_s = time.perf_counter() - t0
    Xs = st[FEATURE_COLS].to_numpy(np.float32)
    t0 = time.perf_counter(); p_st = final.predict_proba(Xs)[:, 1]; pred_s = time.perf_counter() - t0
    np.save(CACHE / "st_proba" / "xgboost_tuned.npy", p_st.astype(np.float32))
    joblib.dump(final, CACHE / "models" / "xgboost_tuned.joblib")
    met_st, per_trip = tc.evaluate(st, st_key, p_st.astype(float), st_lab, st_arcs, sorted(st_lab))
    per_trip.to_csv(OUT / "per_trip_student_test_tuned.csv", index=False)
    tc.plot_confusion(met_st["confusion"], "xgboost_tuned", OUT)
    n_st = sum(len(v) for v in st_lab.values())

    untuned = json.load(open(path("results") / "models" / "xgboost" / "metrics.json"))
    rows = [
        dict(model="XGBoost untuned", cv_top1=untuned["cv"]["top1"][0], cv_macro_f1=untuned["cv"]["macro_f1"][0],
             cv_log_loss=untuned["cv"]["log_loss"][0], st_top1=untuned["student_test"]["top1"],
             st_macro_f1=untuned["student_test"]["macro_f1"], st_log_loss=untuned["student_test"]["log_loss"],
             st_route_continuity=untuned["student_test"]["route_continuity"],
             fit_time_s=untuned["fit_time_full_s"], inference_ms_per_1k_points=untuned["inference_ms_per_1k_points"]),
        dict(model="XGBoost tuned (Optuna)", cv_top1=cvt.top1.mean(), cv_macro_f1=cvt.macro_f1.mean(),
             cv_log_loss=cvt.log_loss.mean(), st_top1=met_st["top1"], st_macro_f1=met_st["macro_f1"],
             st_log_loss=met_st["log_loss"], st_route_continuity=met_st["route_continuity"],
             fit_time_s=fit_s, inference_ms_per_1k_points=1000 * pred_s / n_st * 1000),
    ]
    comp = pd.DataFrame(rows); comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["cv_top1", "cv_macro_f1", "st_top1", "st_macro_f1"]
    x = np.arange(len(cols))
    for i, r in comp.iterrows():
        ax[0].bar(x + i * 0.38, [r[c] for c in cols], 0.38, label=r.model)
        ax[1].bar(np.arange(2) + i * 0.38, [r.cv_log_loss, r.st_log_loss], 0.38, label=r.model)
    ax[0].set_xticks(x + 0.19); ax[0].set_xticklabels(["CV Top-1", "CV macro-F1", "ST Top-1", "ST macro-F1"])
    ax[0].set_ylim(0.8, 0.96); ax[0].legend(); ax[0].set_title("Accuracy metrics (higher is better)")
    ax[1].set_xticks([0.19, 1.19]); ax[1].set_xticklabels(["CV log loss", "ST log loss"]); ax[1].set_title("Log loss (lower is better)")
    for a in ax: a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "tuned_vs_untuned.png", dpi=130); plt.close(fig)
    json.dump({"n_trials_total": len(df), "n_complete": int((df.state == "COMPLETE").sum()),
               "n_pruned": int((df.state == "PRUNED").sum()), "search_wall_time_s": search_s,
               "sampler": "TPE(seed=42, n_startup_trials=10)", "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=1)",
               "objective": "mean 5-fold GroupKFold observation-level Top-1 on 80 train trips",
               "tuned_student_test": {k: v for k, v in met_st.items() if k != "confusion"}},
              open(OUT / "run_info.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
