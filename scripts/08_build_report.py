"""Build the technical report (report/PathSnap_Technical_Report.pdf) from the saved results.

Every number and figure in the report is read from results/ (produced by scripts 01-07), so the
PDF can be regenerated and verified against the repository at any time:

    python scripts/08_build_report.py

The HTML is rendered to PDF with a headless Edge/Chrome (print-to-pdf). Folium maps are captured
as PNG screenshots with the same browser.
"""
import _bootstrap  # noqa: F401

import html
import importlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.config import ROOT, load_config, path

RES = path("results")
REP = ROOT / "report"; (REP / "img").mkdir(parents=True, exist_ok=True)
CFG = load_config()
maps07 = importlib.import_module("07_make_maps")
TEXT = json.load(open(REP / "report_text.json", encoding="utf-8"))


def rel(p: Path) -> str:
    return os.path.relpath(p, REP).replace("\\", "/")


def fig(p, caption, width=90):
    p = Path(p)
    if not p.exists():
        return f'<p class="missing">[missing figure: {html.escape(str(p))}]</p>'
    return (f'<figure style="width:{width}%"><img src="{rel(p)}"><figcaption>{caption}</figcaption></figure>')


def table(df, fmt=None, index=False):
    df = df.copy()
    for c, f in (fmt or {}).items():
        if c in df:
            df[c] = df[c].map(lambda v: "N/A" if pd.isna(v) else f.format(v))
    return df.to_html(index=index, escape=True, border=0, classes="tbl", na_rep="N/A")


def pct(x):
    return f"{100 * x:.1f}%"


def P(text):
    return f"<p>{text}</p>"


def section_eda():
    st = pd.read_csv(RES / "eda" / "trip_stats.csv", dtype={"trip_id": str})
    miss = pd.read_csv(RES / "eda" / "missing_values.csv")
    lab = pd.read_csv(RES / "labels" / "label_stats.csv", dtype={"trip_id": str})
    grid = pd.read_csv(RES / "candidates" / "grid.csv")
    mr = pd.read_csv(RES / "candidates" / "miss_rate_by_partition.csv")
    corr = pd.read_csv(RES / "features" / "feature_target_corr_train.csv", index_col=0).iloc[:, 0]
    tr = st[st.partition == "train"]; sv = st[st.partition == "student_test"]
    inv = pd.DataFrame([
        dict(partition=p, trips=len(g), gps_points=int(g.clean_points.sum()),
             points_per_trip_median=int(g.clean_points.median()), nodes_range=f"{g.n_nodes.min()} - {g.n_nodes.max()}",
             arcs_range=f"{g.n_arcs.min()} - {g.n_arcs.max()}", route_arcs_range=f"{g.route_len.min()} - {g.route_len.max()}",
             trips_with_gap_over_30s=int((g.gaps_gt_30s > 0).sum()), max_gap_s=int(g.dt_max.max()))
        for p, g in st.groupby("partition")])
    lab_s = lab.groupby("partition").agg(points=("n_points", "sum"), matched=("n_matched", "sum"),
                                         route_arc_coverage=("frac_route_arcs_covered", "mean"),
                                         median_label_dist_m=("median_label_dist", "median")).reset_index()
    lab_s["matched_frac"] = lab_s.matched / lab_s.points
    worst = lab[lab.partition == "train"].sort_values("frac_matched").head(4)[["trip_id", "n_points", "frac_matched"]]
    sel = grid[(grid.R == 50) & (grid.K == 8)].iloc[0]
    eda_map = maps07.screenshot(RES / "eda" / "maps" / "eda_student_test_00000000.html", REP / "img" / "eda_map_student_test_00000000.png")
    lab_map = maps07.screenshot(RES / "labels" / "maps" / "labels_train_00000033.html", REP / "img" / "labels_train_00000033.png")
    feat_rows = [
        ("Distance", "d, log_d, d_minus_min, d_ratio, rank, softmin, gap12",
         "GPS noise is ~5 m, so the true arc is usually among the closest; relative versions (vs. the nearest candidate) make the feature comparable across dense and sparse areas."),
        ("Heading", "cos_dh, abs_dh, heading_valid",
         "~95 % of roads are two-way twins with identical geometry: only the agreement between direction of travel and arc bearing separates them."),
        ("Position on arc", "u, dist_to_start, dist_to_end, log_arc_len",
         "Points projecting onto the clipped end of a segment are often on the next arc; long arcs collect more points."),
        ("Topology", "out_deg_to, in_deg_from, is_two_way, n_cand, n_within_10m",
         "Junction structure and how crowded / ambiguous the candidate set is."),
        ("Context (+-1, +-3, +-5 points)", "ctx_d_*, ctx_d_mean, ctx_conn_frac, is_nearest_prev, is_nearest_next",
         "A single noisy fix is unreliable; neighbouring fixes that lie on (or connect to) the same arc give strong evidence."),
        ("Kinematics", "speed, dt_prev, dt_next, turn_rate",
         "Low speed makes heading unreliable; time gaps and turns indicate where a change of arc is likely."),
    ]
    h = ["<h2>2. Dataset Understanding and Exploratory Data Analysis</h2>",
         P("Script: <code>scripts/01_eda.py</code> (inventory, quality checks, plots, Folium maps), "
           "<code>scripts/02a_candidate_grid.py</code> (labels, candidates), <code>scripts/02_build_dataset.py</code> (features). "
           "<b>Partitions:</b> all statistics, plots and design decisions use the <b>train</b> partition (80 trips); the "
           "<b>student_test</b> partition (10 trips) was only inventoried and sanity-checked; the evaluator partition was never opened."),
         "<h3>2.1 Data organisation</h3>",
         P("Each trip folder holds four header-less, whitespace-separated files: <code>.track</code> (lon, lat, t in seconds), "
           "<code>.nodes</code> (lon, lat; row index = node id), <code>.arcs</code> (from_node, to_node; row index = arc id, directed) and "
           "<code>.route</code> (chronological arc ids of the reference route). GPS rows do <b>not</b> correspond to route rows: "
           f"train trips contain {tr.points_per_route_arc.median():.1f} GPS points per route arc (median)."),
         table(inv),
         P("Data types (verified when loading): .track float64 x 3, .nodes float64 x 2, .arcs int64 x 2, .route int64 x 1; no header rows. "
           "Coordinate order is longitude, latitude (checked: every track lies inside its node bounding box and latitudes are plausible)."),
         table(pd.DataFrame([dict(partition=p_, **{f"{e}_MB_total": round(g_[f'{e}_bytes'].sum() / 1e6, 1) for e in ("track", "nodes", "arcs", "route")},
                                  files_missing=int((~g_[[f'{e}_present' for e in ('track', 'nodes', 'arcs', 'route')]]).sum().sum()))
                             for p_, g_ in pd.read_csv(RES / "eda" / "inventory.csv").groupby("partition")])),
         P(f"Geographical extent of the trajectories (train): east-west median {tr.extent_x_km.median():.1f} km (max {tr.extent_x_km.max():.1f} km), "
           f"north-south median {tr.extent_y_km.median():.1f} km (max {tr.extent_y_km.max():.1f} km); path length median "
           f"{tr.path_length_km.median():.1f} km; median speed {tr.speed_median.median():.1f} m/s. Isolated nodes: {int(tr.isolated_nodes.sum())}. "
           "Per-trip values are in results/eda/trip_stats.csv."),
         "<h3>2.2 Missing-value and data-quality analysis</h3>",
         table(miss.rename(columns={"bad_shape_files": "files_with_wrong_columns"})),
         P(f"No file is missing, no value is NaN/inf, every track is time-sorted, no timestamps are duplicated and every track lies inside "
           f"its network's bounding box. Real issues found on train: {int((tr.gaps_gt_30s > 0).sum())} trips with a sampling gap above 30 s "
           f"(largest {int(tr.dt_max.max())} s), {int(tr.n_speed_outliers.sum())} implied-speed outliers above "
           f"{CFG['cleaning']['max_speed_mps']:.0f} m/s in {int((tr.n_speed_outliers > 0).sum())} trips, {int(tr.n_self_loops.sum())} self-loop arcs, "
           f"{int(tr.n_duplicate_arcs.sum())} duplicate arcs, {int(tr.n_zero_length_arcs.sum())} zero-length arcs, networks split into up to "
           f"{int(tr.weak_components.max())} weakly connected components (largest strongly connected component as small as "
           f"{pct(tr.largest_scc_frac.min())} of nodes), and reference routes that cover only part of the trip."),
         "<h3>2.3 How we pre-processed the data</h3>", "<ol>",
         "<li><b>Cleaning</b> (<code>io.clean_track</code>): drop non-finite / out-of-range coordinates, stable-sort by time, drop duplicate "
         "timestamps (compared on a 1 ms grid because some timestamps are 0.99999 instead of 1.0). Kept for robustness on unseen trips.</li>",
         f"<li><b>Projection</b>: longitude/latitude are angles, so Euclidean distance in degrees is wrong (1 deg of longitude is 111 km at the "
         f"equator but 63 km at 55 N). Each trip is projected with a local equirectangular projection centred on the trip; the largest "
         f"single-step error versus the haversine distance on train is {tr.max_proj_error_m.max():.2f} m, far below GPS noise.</li>",
         f"<li><b>Per-point labels</b> (<code>labels.derive_labels</code>): the route is an ordered arc list, so each GPS point is aligned to "
         f"a route position with a monotone dynamic programme (positions may only move forward, which resolves loops and roads driven twice) "
         f"minimising point-to-arc distance plus a small heading penalty. Points still farther than {CFG['labels']['max_match_dist_m']:.0f} m "
         f"from their arc are labelled <i>unmatched</i> (-1) and excluded from training positives and from the Top-1 denominator; "
         f"strict accuracy counts them as errors.</li>",
         f"<li><b>Candidate generation</b> (<code>candidates.ArcIndex</code>): arcs are densified every 10 m into a KD-tree; for each GPS "
         f"point the K nearest arcs within R metres are kept (ties kept, so both directions of a two-way road enter together). R and K were "
         f"chosen on train only (Figure 4): the cheapest setting within 0.5 pp of the lowest miss rate is <b>R = 50 m, K = 8</b> "
         f"(miss rate {pct(sel.miss_rate)}, {sel.cands_per_point:.2f} candidates per point).</li>",
         "<li><b>Feature engineering</b> (<code>features.build_features</code>): one row per (GPS point, candidate arc); 33 features, none of "
         "which uses absolute coordinates (trips are in different countries) or the reference route (a unit test deletes the .route file "
         "and checks that all features are unchanged).</li></ol>",
         table(lab_s, {"route_arc_coverage": "{:.3f}", "median_label_dist_m": "{:.2f}", "matched_frac": "{:.4f}"}),
         P("Trips where the reference route covers only part of the drive (train):"),
         table(worst, {"frac_matched": "{:.3f}"}),
         table(mr, {"cands_per_point": "{:.2f}", "miss_rate": "{:.4f}", "positive_rate": "{:.3f}"}),
         "<h3>2.4 Patterns observed and features extracted</h3>",
         P(TEXT["eda_patterns"]),
         table(pd.DataFrame(feat_rows, columns=["Group", "Features", "Why we chose them"])),
         P("Correlation of each feature with the label on train candidates (strongest five each way): " +
           ", ".join(f"{k} {v:+.2f}" for k, v in list(corr.sort_values().items())[:5]) + "; " +
           ", ".join(f"{k} {v:+.2f}" for k, v in list(corr.sort_values(ascending=False).items())[:5]) + "."),
         fig(RES / "eda" / "trajectories_train.png", "Figure 1 - GPS trajectory plots of six train trips in projected metres (colour = time)."),
         fig(RES / "eda" / "network_00000026.png", "Figure 2 - Road-network visualisation around train trip 00000026 with the reference route and GPS.", 70),
         fig(RES / "eda" / "dt_hist_train.png", "Figure 3a - Sampling interval distribution (train).", 48) +
         fig(RES / "eda" / "gps_route_dist_train.png", "Figure 3b - Distance of GPS points to the reference route (train).", 48),
         fig(RES / "candidates" / "grid.png", "Figure 4 - Candidate miss rate and candidate-set size versus R and K (train)."),
         fig(RES / "eda" / "network_size_train.png", "Figure 5a - Road-network sizes (train).", 48) +
         fig(RES / "eda" / "scc_frac_train.png", "Figure 5b - Share of nodes in the largest strongly connected component (train).", 48),
         fig(RES / "features" / "feature_corr_train.png", "Figure 6 - Feature correlation matrix (train candidates, 300k-row sample).", 75),
         fig(REP / "img" / "eda_map_student_test_00000000.png",
             "Figure 7 - Folium map (student_test 00000000): raw GPS (blue), road network (grey), reference route (green); "
             "interactive version: results/eda/maps/eda_student_test_00000000.html.", 95) if eda_map else "",
         fig(REP / "img" / "labels_train_00000033.png",
             "Figure 8 - Folium label-validation map (train 00000033): the reference route covers only part of the drive; points more "
             "than 40 m away (purple) are labelled unmatched.", 95) if lab_map else ""]
    return "\n".join(h)


MODEL_NAMES = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours",
               "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM",
               "xgboost": "XGBoost", "mlp": "Multi-layer Perceptron",
               "baseline_nearest": "Baseline: nearest arc", "baseline_dist_heading": "Baseline: distance + heading rule"}


def section_models():
    comp = pd.read_csv(RES / "models" / "comparison.csv")
    t = comp.copy()
    t["Model"] = t.model.map(MODEL_NAMES)
    t["Route continuity"] = t.st_route_continuity
    tab = t[["Model", "family", "key_config", "cv_macro_f1", "cv_top1", "st_top1", "Route continuity", "st_macro_f1",
             "st_log_loss", "inference_ms_per_1k_points"]].rename(columns={
        "family": "Algorithm family", "key_config": "Key configuration", "cv_macro_f1": "CV macro-F1",
        "cv_top1": "CV Top-1", "st_top1": "Top-1 accuracy (student-test)", "st_macro_f1": "Student-test macro-F1",
        "st_log_loss": "Log loss (student-test)", "inference_ms_per_1k_points": "Inference time (ms / 1000 points)"})
    tab["Key configuration"] = tab["Key configuration"].str.slice(0, 110)
    h = ["<h2>3. Model Experimentation and Comparison</h2>",
         P("Script: <code>scripts/03_train_compare.py</code>; models defined in <code>src/pathsnap/models.py</code>. "
           "<b>Task:</b> binary classification of every (GPS point, candidate arc) row - is this the reference arc? For each GPS point the "
           "highest-probability candidate is the prediction. <b>Protocol:</b> 5-fold GroupKFold by trip on the 80 train trips "
           "(a trip never appears on both sides; fold assignment in results/models/folds.csv), then refit on all 80 trips and "
           "evaluation on the 10 student-test trips. Models are fit on rows of matched points only. Seed 42 everywhere."),
         P("<b>Metrics.</b> <i>Top-1 accuracy</i> (observation level): share of matched GPS points whose highest-scoring candidate is the "
           "reference arc - points whose reference arc is not among the candidates count as errors. <i>Macro-F1, precision, recall, log loss</i>: "
           "candidate-level binary metrics at threshold 0.5. <i>Route continuity</i>: share of consecutive changes of predicted arc where the "
           "second arc starts at the end node of the first. <b>Primary selection metric: CV Top-1</b>, because it is exactly the task "
           "(one arc per GPS point) and it is computed on trips the model never saw; log loss is the tie-breaker because the HMM consumes "
           "the probabilities."),
         "<h3>3.1 Overall model comparison</h3>",
         table(tab, {"CV macro-F1": "{:.4f}", "CV Top-1": "{:.4f}", "Top-1 accuracy (student-test)": "{:.4f}",
                     "Route continuity": "{:.4f}", "Student-test macro-F1": "{:.4f}", "Log loss (student-test)": "{:.4f}",
                     "Inference time (ms / 1000 points)": "{:.1f}"}),
         P("N/A: the two baselines are deterministic rules without probabilities, so log loss is not defined. Route continuity here is "
           "measured on classifier-only predictions (before HMM decoding). Inference time = classifier prediction time on student-test, "
           "excluding candidate generation and features (identical for all models)."),
         P("Additional metrics (student-test candidate level, positive class = reference arc; training time = refit on all train rows):"),
         table(t[["Model", "st_precision", "st_recall", "st_accuracy", "st_strict_accuracy", "cv_top1_std", "cv_log_loss",
                  "fit_time_s"]].rename(columns={
             "st_precision": "Precision", "st_recall": "Recall", "st_accuracy": "Candidate accuracy",
             "st_strict_accuracy": "Strict point accuracy", "cv_top1_std": "CV Top-1 std (5 folds)",
             "cv_log_loss": "CV log loss", "fit_time_s": "Training time (s)"}),
               {"Precision": "{:.4f}", "Recall": "{:.4f}", "Candidate accuracy": "{:.4f}", "Strict point accuracy": "{:.4f}",
                "CV Top-1 std (5 folds)": "{:.4f}", "CV log loss": "{:.4f}", "Training time (s)": "{:.1f}"}),
         "<h3>3.2 Per-model analysis</h3>"]
    for _, r in comp.iterrows():
        name = r.model
        if name.startswith("baseline"):
            continue
        mj = json.load(open(RES / "models" / name / "metrics.json"))
        cvf = pd.read_csv(RES / "models" / name / "cv_folds.csv")
        tx = TEXT["models"][name]
        stt = mj["student_test"]
        h += [f"<h4>3.2.{len([x for x in h if x.startswith('<h4>')]) + 1} {MODEL_NAMES[name]} ({mj['family']})</h4>",
              "<p><b>A. Model understanding</b></p>",
              P(f"<b>1. Why suitable:</b> {tx['why']}"), P(f"<b>2. Strengths and limitations:</b> {tx['strengths']}"),
              "<p><b>B. Training and configuration</b></p>",
              P(f"<b>5. Important hyperparameters (initial values):</b> " +
                ", ".join(f"{k} = {v}" for k, v in mj["config"].items()) + (f". {mj['notes']}." if mj.get("notes") else ".")),
              P(f"<b>6. Loss / objective:</b> {tx['loss']}"),
              "<p><b>C. Experimental results</b></p>",
              P(f"<b>7. Student-test performance:</b> Top-1 {stt['top1']:.4f}, strict accuracy {stt['strict_accuracy']:.4f}, "
                f"route continuity {stt['route_continuity']:.4f}; CV Top-1 {mj['cv']['top1'][0]:.4f} +- {mj['cv']['top1'][1]:.4f} "
                f"over 5 folds; fit time on all train rows {mj['fit_time_full_s']:.1f} s; inference "
                f"{mj['inference_ms_per_1k_points']:.1f} ms per 1000 GPS points."),
              P(f"<b>8. Observation-level metrics (student-test):</b> Top-1 accuracy {stt['top1']:.4f}, macro-F1 {stt['macro_f1']:.4f} "
                f"(precision {stt['precision']:.4f}, recall {stt['recall']:.4f} for the positive class), log loss {stt['log_loss']:.4f}."),
              P(f"<b>9. Over- / under-fitting:</b> Top-1 on 12 sampled training-fold trips {mj['cv']['train_top1'][0]:.4f} vs held-out CV "
                f"{mj['cv']['top1'][0]:.4f} vs student-test {stt['top1']:.4f}; log loss train {cvf.train_log_loss.mean():.4f} vs CV "
                f"{cvf.log_loss.mean():.4f}. {tx['fit']}")]
    h += ["<h3>3.3 Model visualisations</h3>",
          fig(RES / "models" / "cv_macro_f1.png", "Figure 9 - Cross-validation macro-F1 (mean and std over 5 GroupKFold folds), all models."),
          fig(RES / "models" / "cv_top1.png", "Figure 10 - Cross-validation observation-level Top-1, all models."),
          fig(RES / "models" / "student_test_top1_f1.png", "Figure 11 - Student-test Top-1 accuracy and macro-F1."),
          fig(RES / "models" / "log_loss.png", "Figure 12 - Log loss on CV and student-test (learned models only).", 75),
          fig(RES / "models" / "generalisation_top1.png", "Figure 13 - Top-1 on training-fold trips vs held-out CV vs student-test (over-fitting check)."),
          fig(RES / "models" / "inference_time.png", "Figure 14 - Classifier inference time on student-test.", 75),
          fig(RES / "models" / "xgboost" / "training_history.png", "Figure 15a - XGBoost: real per-iteration log loss (train vs student-test).", 48) +
          fig(RES / "models" / "lightgbm" / "training_history.png", "Figure 15b - LightGBM: real per-iteration log loss.", 48),
          fig(RES / "models" / "mlp" / "training_history.png", "Figure 15c - MLP (Phase 3 run): sklearn's epoch-wise training loss and "
              "validation accuracy on a random 10 % row split.", 48) +
          fig(RES / "models" / "mlp" / "loss_curve_grouped.png", "Figure 15d - MLP: epoch-wise training vs validation log loss on a "
              "trip-grouped split (64 / 16 train trips).", 48),
          P("Only XGBoost, LightGBM (boosting iterations) and the MLP (epochs) produce real training histories, so only they have "
            "training curves. For the boosting curves the student-test trips were passed as an evaluation set purely to record the loss "
            "per iteration - they were not used for early stopping or any other decision. Confusion matrices for every model are in results/models/&lt;model&gt;/confusion_matrix.png; two examples:"),
          fig(RES / "models" / "xgboost" / "confusion_matrix.png", "Figure 16a - XGBoost confusion matrix (student-test candidates).", 40) +
          fig(RES / "models" / "gaussian_nb" / "confusion_matrix.png", "Figure 16b - Gaussian NB confusion matrix (student-test candidates).", 40),
          "<h3>3.4 Model selection</h3>", P(TEXT["selection"])]
    return "\n".join(h)


def section_bayes():
    best = json.load(open(RES / "optuna" / "best_params.json"))
    info = json.load(open(RES / "optuna" / "run_info.json"))
    tv = pd.read_csv(RES / "optuna" / "tuned_vs_untuned.csv")
    ss = json.load(open(RES / "optuna" / "search_space.json"))
    trials = pd.read_csv(RES / "optuna" / "trials.csv")
    top = trials[trials.state == "COMPLETE"].sort_values("value", ascending=False).head(5)
    pcols = [c for c in top.columns if c.startswith("params_")]
    top = top[["number", "value"] + pcols].rename(columns=lambda c: c.replace("params_", ""))
    h = ["<h2>4. Bayesian Hyperparameter Optimisation (XGBoost)</h2>",
         P("Script: <code>scripts/04_bayes_opt.py</code>; full trial record in results/optuna/trials.csv and study.db (Optuna SQLite)."),
         P(f"<b>1. Hyperparameters searched and 2. ranges:</b>"),
         table(pd.DataFrame(list(ss.items()), columns=["hyperparameter", "range / distribution"])),
         P(f"<b>3. Objective:</b> {info['objective']} (same folds as Phase 3; student-test never used inside the search). "
           f"Sampler {info['sampler']}; pruner {info['pruner']} - a trial is stopped after a fold if its running mean is below the median "
           f"of earlier trials at the same fold. The first trial was the untuned configuration, so the search starts from the baseline."),
         P(f"<b>4. Trials:</b> {info['n_trials_total']} ({info['n_complete']} completed, {info['n_pruned']} pruned); search wall time "
           f"{info['search_wall_time_s'] / 60:.1f} min on a 22-thread laptop CPU."),
         P("<b>5. Best hyperparameters:</b> " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                                    for k, v in best["params"].items()) + f" (CV Top-1 {best['best_cv_top1']:.4f})."),
         table(top, {"value": "{:.4f}", **{c: "{:.4g}" for c in top.columns if c not in ("number", "value")}}),
         P("<b>6. Tuned vs untuned</b> (same metrics, same partitions):"),
         table(tv, {c: "{:.4f}" for c in tv.columns if c not in ("model", "fit_time_s", "inference_ms_per_1k_points")} |
               {"fit_time_s": "{:.1f}", "inference_ms_per_1k_points": "{:.1f}"}),
         P(TEXT["bayes_discussion"]),
         fig(RES / "optuna" / "optimisation_history.png", "Figure 17 - Optuna optimisation history (CV Top-1 per trial, best so far)."),
         fig(RES / "optuna" / "param_importance.png", "Figure 18 - Hyperparameter importance (fANOVA).", 65),
         fig(RES / "optuna" / "tuned_vs_untuned.png", "Figure 19 - Untuned vs tuned XGBoost.")]
    return "\n".join(h)


def section_hmm():
    bp = json.load(open(RES / "hmm" / "best_params.json"))
    comp = pd.read_csv(RES / "hmm" / "comparison.csv")
    cols = ["partition", "method", "top1", "strict_accuracy", "direct_continuity", "plausible_continuity", "chain_breaks",
            "unbridged_gaps", "route_precision_mean", "route_recall_mean", "route_f1_mean", "route_len_f1_mean",
            "trips_fully_continuous", "decode_ms_per_1k_points"]
    grid = pd.read_csv(RES / "hmm" / "decoder_grid.csv").head(8)
    delta_st = pd.read_csv(RES / "hmm" / "per_trip_delta_student_test.csv", dtype={"trip_id": str})
    delta_tr = pd.read_csv(RES / "hmm" / "per_trip_delta_train.csv", dtype={"trip_id": str})
    h = ["<h2>5. HMM and Viterbi Route Reconstruction</h2>",
         P("Code: <code>src/pathsnap/hmm.py</code>, <code>src/pathsnap/routing.py</code>; experiment: <code>scripts/05_hmm_eval.py</code>."),
         P(f"<b>1. Hidden states.</b> {TEXT['hmm']['states']}"),
         P(f"<b>2. Emission probabilities.</b> {TEXT['hmm']['emission']}"),
         P(f"<b>3. Transition probabilities.</b> {TEXT['hmm']['transition']}"),
         P(f"<b>Viterbi recurrence.</b> {TEXT['hmm']['viterbi']}"),
         P(f"<b>4. Missing candidates and large GPS gaps.</b> {TEXT['hmm']['gaps']}"),
         P(f"<b>5. Disconnected road segments.</b> {TEXT['hmm']['disconnected']}"),
         P(f"<b>Decoder tuning.</b> Grid over {', '.join(f'{k} in {v}' for k, v in bp['grid'].items())} "
           f"({int(np.prod([len(v) for v in bp['grid'].values()]))} combinations), objective {bp['objective']}. Best: "
           + ", ".join(f"{k} = {v}" for k, v in bp["params"].items()) + f" (train OOF Top-1 {bp['train_oof_top1']:.4f}). "
           f"Defaults before tuning: " + ", ".join(f"{k} = {v}" for k, v in bp["default_params"].items()) + "."),
         table(grid, {"train_oof_top1": "{:.4f}"}),
         P("<b>6. Did HMM decoding improve the route?</b> Classifier-only vs classifier + HMM (train = out-of-fold probabilities, "
           "student_test = model refit on all train trips):"),
         table(comp[cols], {c: "{:.4f}" for c in cols if c not in ("partition", "method", "chain_breaks", "unbridged_gaps")}),
         P("Definitions: <i>direct continuity</i> = consecutive different arcs physically adjacent; <i>plausible continuity</i> = the second arc "
           "is reachable from the first within 2 x GPS displacement + 50 m of driving; <i>route P/R/F1</i> = overlap between the arcs of "
           "the stitched predicted route and the reference route (mean over trips), also length-weighted; <i>unbridged gaps</i> = places "
           "where stitching found no path within 3 km."),
         P(TEXT["hmm_discussion"]),
         P("Per-trip change in Top-1 (HMM tuned - classifier-only), student-test:"),
         table(delta_st.round(4)),
         P("Train trips where HMM helped least (OOF):"),
         table(delta_tr.head(6).round(4)),
         fig(RES / "hmm" / "clf_vs_hmm_student_test.png", "Figure 20 - Classifier-only vs classifier + HMM on student-test."),
         fig(RES / "hmm" / "per_trip_scatter_student_test.png", "Figure 21a - Per-trip Top-1, student-test.", 45) +
         fig(RES / "hmm" / "per_trip_scatter_train.png", "Figure 21b - Per-trip Top-1, train (OOF).", 45),
         fig(RES / "hmm" / "decoder_sensitivity.png", "Figure 22 - Decoder parameter sensitivity on train OOF probabilities.")]
    return "\n".join(h)


def section_maps():
    idx = pd.read_csv(RES / "maps" / "index.csv", dtype={"trip_id": str})
    h = ["<h2>6. Geographical Visualisation Using Folium</h2>",
         P("Script: <code>scripts/07_make_maps.py</code> (map code in <code>src/pathsnap/viz.py</code>); interactive HTML maps in "
           "results/maps/. Each map has toggleable layers (LayerControl) and a legend: road network (grey), reference route (green, wide "
           "translucent), classifier-only arcs (orange, hidden by default), predicted HMM route (red), GPS observations (blue, start/end "
           "markers) and GPS points whose HMM arc differs from the reference (purple). The extent is fitted to the trip."),
         table(idx[["which", "trip_id", "top1", "route_f1", "wrong_points", "html"]], {"top1": "{:.4f}", "route_f1": "{:.4f}"})]
    for k, (_, r) in enumerate(idx.iterrows()):
        if isinstance(r.png, str) and r.png:
            h.append(fig(RES / "maps" / "png" / r.png, f"Figure {23 + k} - {r.which} student-test trip {r.trip_id}: Top-1 {pct(r.top1)}, "
                                                        f"route F1 {pct(r.route_f1)} (results/maps/{r.html}).", 95))
    h.append(P(TEXT["maps_discussion"]))
    return "\n".join(h)


def section_deploy():
    chk = pd.read_csv(RES / "pipeline_check.csv", dtype={"trip_id": str})
    log_excerpt = ""
    lp = RES / "logs" / "api_rehearsal_student_test.log"   # copy of logs/api.log from the HTTP rehearsal
    if lp.exists():
        lines = lp.read_text(encoding="utf-8").splitlines()
        log_excerpt = "\n".join(lines[:9] + ["..."] + lines[-12:])
    ui_png = REP / "img" / "ui.png"
    h = ["<h2>7. API, UI and Deployment</h2>",
         P(TEXT["deploy"]),
         table(pd.DataFrame([
             ("GET", "/health", "status, model name, candidate and HMM parameters, number of trips"),
             ("GET", "/trips", "ids and GPS-point counts of the trips in PATHSNAP_TRIPS_DIR (no ground truth)"),
             ("POST", "/predict {trip_id}", "full pipeline -> GPS points, snapped points, per-point arc ids, route polylines, "
                                           "server-side metrics, per-stage timings")], columns=["Method", "Endpoint", "Returns"])),
         P("End-to-end verification (<code>scripts/06_export_artifacts.py</code>): the exported artifacts were run through exactly the "
           "API code path on all 10 student-test trips; Top-1 matches the offline evaluation for every trip:"),
         table(chk, {"pipeline_top1": "{:.4f}", "offline_top1": "{:.4f}", "route_f1": "{:.4f}",
                     "plausible_continuity": "{:.4f}", "total_s": "{:.2f}"}),
         fig(ui_png, "Figure 26 - Evaluator UI (Streamlit) after reconstructing a student-test trip in a local rehearsal.", 95)
         if ui_png.exists() else P("<span class='missing'>[Insert screenshot of the UI on Laptop 2]</span>"),
         P("<b>API log excerpt</b> - local two-process rehearsal (API with PATHSNAP_TRIPS_DIR=data/student_test, requests over HTTP "
           "for all 10 trips); full log in results/logs/api_rehearsal_student_test.log:"),
         f"<pre>{html.escape(log_excerpt)}</pre>" if log_excerpt else "",
         *[fig(REP / "img" / f, cap, 95) for f, cap in [
             ("eval_ui.png", "Figure 27 - UI on Laptop 2 during the final evaluation."),
             ("eval_api_console.png", "Figure 28 - API console on Laptop 1 (receiving laptop) showing successful /predict requests "
                                      "during the final evaluation.")] if (REP / "img" / f).exists()],
         "" if (REP / "img" / "eval_api_console.png").exists() else
         P("<span class='missing'>[Evaluation-day screenshots: save them as report/img/eval_ui.png and "
           "report/img/eval_api_console.png and re-run scripts/08_build_report.py.]</span>")]
    return "\n".join(h)


CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#1d1d1d;margin:0 6mm}
h1{font-size:22pt;margin:0 0 4px} h2{font-size:15pt;border-bottom:2px solid #2b4c7e;padding-bottom:3px;margin-top:22px;
page-break-before:always} h3{font-size:12pt;color:#2b4c7e;margin:16px 0 6px} h4{font-size:11pt;margin:14px 0 4px}
.tbl{border-collapse:collapse;font-size:8.3pt;margin:6px 0 10px;width:100%} .tbl th{background:#e8eef7;text-align:left}
.tbl th,.tbl td{border:1px solid #c9d3e2;padding:3px 5px;vertical-align:top} figure{display:inline-block;text-align:center;
margin:6px 0.5%;vertical-align:top;page-break-inside:avoid} figure img{width:100%} figcaption{font-size:8.5pt;color:#444} code{font-size:9pt;background:#f2f2f2;
padding:0 2px} pre{font-size:7.5pt;background:#f6f6f6;padding:6px;white-space:pre-wrap} .missing{color:#b00;font-weight:600}
.cover{text-align:center;margin-top:45mm;margin-bottom:30mm} .cover p{font-size:12pt} h2.first{page-break-before:avoid}
"""


def main():
    team = TEXT["team"]
    parts = [f"<html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>",
             "<div class='cover'><h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>",
             "<p>UE24CS352A Machine Learning - Hackathon technical report</p>",
             f"<p><b>Team {html.escape(team['id'])}</b><br>{'<br>'.join(html.escape(m) for m in team['members'])}</p>",
             f"<p>Repository: {html.escape(team['repo'])}</p></div>",
             "<h2 class='first'>1. Overview</h2>", P(TEXT["overview"]),
             section_eda(), section_models(), section_bayes(), section_hmm(), section_maps(), section_deploy(),
             "<h2>8. Limitations and conclusions</h2>", P(TEXT["conclusion"]),
             "<h3>Reproducibility</h3>", "<pre>" + html.escape(TEXT["reproduce"]) + "</pre>",
             "<h3>AI assistance disclosure</h3>", P(TEXT["ai_disclosure"]),
             "</body></html>"]
    out_html = REP / "PathSnap_Technical_Report.html"
    out_html.write_text("\n".join(parts), encoding="utf-8")
    exe = next((b for b in maps07.BROWSERS if b and os.path.exists(b)), None)
    out_pdf = REP / "PathSnap_Technical_Report.pdf"
    if exe:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out_pdf}", out_html.resolve().as_uri()], timeout=180, capture_output=True)
    print("report:", out_html, out_pdf if out_pdf.exists() else "(PDF not produced - open the HTML and print to PDF)")


if __name__ == "__main__":
    main()
