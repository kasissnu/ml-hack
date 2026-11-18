"""Phase 6 - Export the inference artifacts and verify the end-to-end pipeline.

Writes artifacts/:
  model.joblib            tuned XGBoost (refit on all 80 train trips, scripts/04_bayes_opt.py)
  feature_cols.json       ordered feature list expected by the model
  candidate_params.json   radius / K / densify step (scripts/02a_candidate_grid.py)
  hmm_params.json         tuned decoder parameters (scripts/05_hmm_eval.py)
  model_card.json         provenance: versions, seed, metrics, git commit

Verification: pathsnap.pipeline.predict_trip is run on every student_test trip, exactly as the API
does, and its Top-1 must equal the offline evaluation in results/hmm/per_trip_student_test.csv.
Result: results/pipeline_check.csv
"""
import _bootstrap  # noqa: F401

import json
import platform
import shutil
import subprocess
import time

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost

from pathsnap.config import ROOT, load_config, partition_dir, path
from pathsnap.features import FEATURE_COLS
from pathsnap.io import list_trips
from pathsnap.pipeline import Artifacts, evaluate_prediction, predict_trip

CFG = load_config()
ART = path("artifacts")
RES = path("results")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    shutil.copy(path("cache") / "models" / "xgboost_tuned.joblib", ART / "model.joblib")
    json.dump(FEATURE_COLS, open(ART / "feature_cols.json", "w"), indent=1)
    cand = json.load(open(RES / "candidates" / "chosen.json"))
    json.dump({"radius_m": cand["radius_m"], "k": cand["k"], "densify_step_m": CFG["candidates"]["densify_step_m"]},
              open(ART / "candidate_params.json", "w"), indent=2)
    hmm = json.load(open(RES / "hmm" / "best_params.json"))
    json.dump(hmm["params"], open(ART / "hmm_params.json", "w"), indent=2)
    tuned = pd.read_csv(RES / "optuna" / "tuned_vs_untuned.csv").set_index("model").iloc[-1].to_dict()
    hmm_comp = pd.read_csv(RES / "hmm" / "comparison.csv")
    hmm_st = hmm_comp[(hmm_comp.partition == "student_test") & (hmm_comp.method == "hmm_tuned")].iloc[0].to_dict()
    card = dict(name="PathSnap map-matcher", classifier="XGBoost (Optuna-tuned)",
                best_params=json.load(open(RES / "optuna" / "best_params.json"))["params"],
                trained_on="80 train trips (matched GPS points only)", seed=CFG["seed"],
                created=time.strftime("%Y-%m-%d %H:%M"), git_commit=git_commit(),
                metrics_classifier=tuned, metrics_hmm_student_test=hmm_st,
                versions=dict(python=platform.python_version(), xgboost=xgboost.__version__,
                              sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, optuna=optuna.__version__,
                              numpy=np.__version__, pandas=pd.__version__))
    json.dump(card, open(ART / "model_card.json", "w"), indent=2, default=float)

    art = Artifacts.load(ART)
    offline = pd.read_csv(RES / "hmm" / "per_trip_student_test.csv", dtype={"trip_id": str})
    offline = offline[offline.method == "hmm_tuned"].set_index("trip_id")
    rows = []
    for tid in list_trips(partition_dir("student_test")):
        d = partition_dir("student_test") / tid
        pred = predict_trip(d, art)
        met = evaluate_prediction(d, pred)
        rows.append(dict(trip_id=tid, n_points=pred["n_points"], pipeline_top1=met["top1_accuracy"],
                         offline_top1=offline.loc[tid, "top1"], route_f1=met["route_f1"],
                         plausible_continuity=met["plausible_continuity"], total_s=pred["timings"]["total_s"]))
        print(rows[-1], flush=True)
    chk = pd.DataFrame(rows)
    chk["match"] = np.isclose(chk.pipeline_top1, chk.offline_top1, atol=1e-9)
    chk.to_csv(RES / "pipeline_check.csv", index=False)
    print(chk.to_string(index=False))
    assert chk.match.all(), "pipeline output differs from the offline evaluation"
    print("artifacts exported and verified")


if __name__ == "__main__":
    main()
