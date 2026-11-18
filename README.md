<<<<<<< HEAD
# PathSnap — Map-Matching & Route Reconstruction (Team 32)

Noisy GPS trajectories are matched to a road network:

1. **Candidate generation** – for every GPS point, the road arcs within 50 m (8 nearest) are found with a KD-tree.
2. **Classifier** – an Optuna-tuned **XGBoost** model scores each (GPS point, candidate arc) pair using 33 engineered features
   (distance, heading agreement, position on arc, topology, neighbouring-point context, kinematics).
3. **HMM + Viterbi** – classifier probabilities are the emissions, road-network distances give the transitions;
   Viterbi decoding picks the most likely *connected* arc sequence, which is stitched into a continuous route.
4. **Deployment** – a FastAPI service (Laptop 1) runs the full pipeline and scores it server-side; a Streamlit UI (Laptop 2)
   lets the evaluator pick one of the reserved trips and shows the predicted route on a Folium map.

The technical report is generated under `report/` after the Team 32 experiment pipeline completes.

---

## 1. Installation

```bash
conda create -n pathsnap python=3.11 -y      # or use any existing Python 3.11 env
conda activate pathsnap
pip install -r requirements.txt
```

### Data layout (not committed)
```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}       80 trips
data/student_test/<trip_id>/...                                 10 trips
data/eval/<trip_id>/...                                         evaluator trips (copied in on evaluation day)
```

For the supplied local materials in this workspace:

```bash
export PATHSNAP_DATA_ROOT="../Student folder"
```

## 2. Reproducing the experiments

Run from the repository root, in order. Every script writes its outputs (CSV, JSON, PNG, HTML) into `results/`;
large intermediate files go to `cache/` (git-ignored).

| Step | Command | What it does | Main outputs |
|---|---|---|---|
| 1 | `python scripts/01_eda.py` | inventory, data-quality checks, EDA plots, Folium maps | `results/eda/` |
| 2a | `python scripts/02a_candidate_grid.py` | per-point labels (monotone DP alignment), candidate R/K grid | `results/labels/`, `results/candidates/` |
| 2b | `python scripts/02_build_dataset.py` | candidate feature dataset (33 features) | `cache/features_*.pkl`, `results/features/` |
| 3 | `python scripts/03_train_compare.py` | selected five models, 5-fold GroupKFold CV + student-test | `results/models/` |
| 3b | `python scripts/03b_mlp_loss_curve.py` | MLP epoch-wise training vs validation loss on a trip-grouped split | `results/models/mlp/loss_curve_grouped.*` |
| 4 | `python scripts/04_bayes_opt.py --trials 30` | Optuna (TPE) Bayesian optimisation of XGBoost, tuned vs untuned | `results/optuna/` |
| 5 | `python scripts/05_hmm_eval.py` (tuned) and `python scripts/05_hmm_eval.py --model xgboost --out hmm_untuned` | HMM decoder tuning, classifier-only vs HMM | `results/hmm/`, `results/hmm_untuned/` |
| 6 | `python scripts/06_export_artifacts.py` | export model + configs, verify the API pipeline | `artifacts/`, `results/pipeline_check.csv` |
| 7 | `python scripts/07_make_maps.py` | Folium maps of predictions (best/median/worst trip) | `results/maps/` |
| 8 | `python scripts/08_build_report.py` | builds the PDF report from `results/` | `report/` |

Tests: `python -m pytest -q tests` (geometry, labels, candidates vs brute force, leakage check, router, Viterbi, API).

Approximate run times on a 22-thread laptop: steps 1–2 < 1 min each, step 3 ≈ 45 min, step 3b ≈ 1 min,
step 4 ≈ 1.6 h (30 trials), step 5 ≈ 5 min per model, steps 6–8 ≈ 2 min each.

Experiment logs of the runs reported in the PDF are in `results/logs/`; Optuna's full trial record is `results/optuna/study.db`
and `trials.csv`. Seed 42 is used everywhere (`config.yaml`).

**Model weights:** `artifacts/model.joblib` is the deployed Team 32 model. The weights of the other trained models are
in `artifacts/experiment_models/` — except Random Forest (332 MB, above GitHub's 100 MB file limit), which
`scripts/03_train_compare.py --models random_forest` regenerates deterministically.

## 3. Deployment (two laptops)

**Laptop 1 – API** (needs the repo, `artifacts/` and the trips):
```powershell
conda activate pathsnap
cd team32
$env:PATHSNAP_TRIPS_DIR = "data\eval"          # folder with the 10 evaluator trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* Find the laptop's IP with `ipconfig` (e.g. `192.168.1.20`).
* Allow inbound TCP 8000 in Windows Defender Firewall (once, as administrator):
  `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
* The console (and `logs/api.log`) shows every request: client IP, trip, points, candidates, stage timings and metrics.

**Laptop 2 – UI** (needs only the repo code and requirements, no data):
```powershell
conda activate pathsnap
cd team32
$env:PATHSNAP_API = "http://192.168.1.20:8000"   # Laptop 1's IP
streamlit run ui/app.py --server.port 8501
```
Open http://localhost:8501, pick a trip in the sidebar and press **Reconstruct route**.

Both laptops must be on the same network (same Wi-Fi or a phone hotspot). Quick check from Laptop 2:
`curl http://192.168.1.20:8000/health`.

**Rehearsal** with the student-test trips: set `PATHSNAP_TRIPS_DIR=data\student_test` on Laptop 1.

### API
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | status, model, candidate and HMM parameters |
| GET | `/trips` | available trip ids and GPS-point counts (no ground truth) |
| POST | `/predict` `{"trip_id": "..."}` | runs the full pipeline; returns GPS points, snapped points, per-point arc ids, route polylines, metrics, timings |

Ground truth (`.route`) is read only by `pathsnap.pipeline.evaluate_prediction`, after the prediction, and only aggregate
metrics leave the server.

## 4. Repository layout
```
config.yaml          all parameters (paths, label threshold, candidate settings, CV folds, seed)
src/pathsnap/        library: io, geo, graph, routing, labels, candidates, features, models, metrics, hmm, pipeline, viz
scripts/             numbered, reproducible experiment scripts (01–08)
api/main.py          FastAPI backend (Laptop 1)
ui/app.py            Streamlit evaluator UI (Laptop 2)
artifacts/           exported model and configuration used by the API
results/             all experiment outputs: CSV, JSON, plots, Folium HTML maps
report/              technical report (PDF + HTML) and its text source
tests/               pytest suite
```

## 5. Key results (student-test, 10 trips)

| Stage | Top-1 accuracy | Plausible route continuity | Route F1 |
|---|---|---|---|
| Nearest-arc baseline | 0.538 | – | – |
| XGBoost (untuned), classifier only | 0.899 | 0.82 | 0.80 |
| XGBoost (Optuna-tuned), classifier only | 0.894 | 0.83 | 0.81 |
| **Tuned XGBoost + HMM/Viterbi (deployed)** | **0.951** | **0.99** | **0.96** |

* Candidate miss rate (true arc not among candidates): 0.22 % (student-test), 0.46 % (train).
* 5-fold GroupKFold CV Top-1: XGBoost 0.888 untuned → 0.891 tuned (30 Optuna TPE trials).
* API latency: 0.1–1.9 s per student-test trip (largest train trip, 19 k points: ~9 s).

Sources: `results/models/comparison.csv`, `results/optuna/tuned_vs_untuned.csv`, `results/hmm/comparison.csv`
(the untuned-model HMM run is kept in `results/hmm_untuned/`). All report tables are generated from these files.

## 6. Team collaboration

This repository includes the completed Team 32 pipeline and the teammate frontend/ML work merged with full Git history.
The teammate repository is available as the `teammate` remote for historical reference.
