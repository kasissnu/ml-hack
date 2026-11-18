# Team 32 PathSnap — Complete Presentation Guide

## 1. One-sentence explanation

PathSnap converts noisy GPS trajectories into accurate, connected road routes by combining machine-learning candidate scoring with HMM/Viterbi sequence decoding.

## 2. The problem

GPS is noisy. A GPS point may be several metres away from the actual road, and nearby roads can look similar. Choosing the nearest road independently for every point can create incorrect jumps and disconnected routes.

Our system answers two questions:

1. Which nearby road segment is most likely for each GPS point?
2. Which sequence of road segments forms the most realistic continuous route?

## 3. Complete data flow

```text
GPS tracks + road network
          |
          v
Data loading and quality checks
          |
          v
Nearby road candidate generation
          |
          v
33 features for every GPS-candidate pair
          |
          v
Machine-learning probability scores
          |
          v
HMM + Viterbi route decoding
          |
          v
Continuous route reconstruction
          |
          v
FastAPI backend -> Streamlit frontend and map
```

## 4. Main implementation stages

### Stage 1 — Data loading and validation

`src/pathsnap/io.py` reads GPS tracks, road nodes, road arcs, and reference routes. The data contains 80 training trips and 10 student-test trips.

### Stage 2 — Candidate generation

`src/pathsnap/candidates.py` finds nearby road segments for each GPS point using a KD-tree.

```text
Search radius: 50 metres
Maximum candidates per GPS point: 8
Road geometry densification: 10 metres
```

The true road was absent from the candidate list for only 0.46% of training points and 0.22% of student-test points.

### Stage 3 — Label generation

`src/pathsnap/labels.py` uses monotone dynamic-programming alignment to assign the correct road arc to each GPS observation. The monotone constraint prevents labels from moving backwards arbitrarily through the route.

### Stage 4 — Feature engineering

`src/pathsnap/features.py` creates 33 features for every GPS point and candidate road pair:

- GPS-to-road distance
- Heading agreement
- Position along the road segment
- Candidate rank
- Road segment length
- Speed and acceleration
- Previous and next GPS-point context
- Connectivity to neighbouring road segments

### Stage 5 — Machine-learning scoring

`src/pathsnap/models.py` contains five model families. Each model outputs the probability that a candidate road is correct.

### Stage 6 — HMM and Viterbi decoding

The classifier scores candidates independently. That can still produce road jumps between consecutive GPS points. The HMM adds route-level reasoning:

- Emission probability: classifier confidence for a candidate road
- Transition probability: how realistic movement is between candidates
- Gap handling: support for missing observations
- U-turn control: discourages implausible immediate reversals

Viterbi selects the most likely complete sequence of connected road segments.

### Stage 7 — API and frontend

`api/main.py` is the FastAPI backend. It exposes:

```text
GET  /health
GET  /trips
POST /predict
```

`ui/app.py` is the Streamlit frontend. It connects to the API, lets the evaluator select a trip, and displays GPS observations, snapped points, the predicted route, metrics, and timings on a Folium map.

## 5. Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | NumPy, Pandas, SciPy |
| Machine learning | Scikit-learn, XGBoost, LightGBM |
| Hyperparameter tuning | Optuna |
| Backend | FastAPI and Uvicorn |
| Frontend | Streamlit |
| Maps | Folium |
| Model storage | Joblib and JSON |
| Configuration | YAML |
| Testing | Pytest |
| Version control | Git and GitHub |

## 6. Why these five models?

We selected five models to compare different approaches instead of relying on one model family.

| Model | Reason for selection |
|---|---|
| LightGBM | Fast gradient boosting with very strong validation performance |
| XGBoost | Strong overall performance and effective Optuna tuning |
| Random Forest | Robust bagging baseline and non-boosting comparison |
| HistGradientBoosting | Efficient histogram-based boosting model |
| MLP | Neural-network comparison against tree-based models |

## 7. Model comparison results

Student-test classifier-only results:

| Model | Top-1 accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| XGBoost | 89.7% | 93.7% | 0.082 |
| MLP | 89.0% | 93.0% | 0.093 |
| LightGBM | 88.9% | 93.5% | 0.083 |
| HistGradientBoosting | 88.8% | 93.3% | 0.097 |
| Random Forest | 88.5% | 93.3% | 0.086 |

### Why XGBoost was deployed

XGBoost was selected because it had the strongest overall classifier performance, had already been tuned with Optuna, and integrated well with the HMM decoder. LightGBM was very competitive and faster in some measurements, but XGBoost gave the strongest overall accuracy for the final pipeline.

## 8. Effect of HMM/Viterbi

The HMM produced the largest improvement because it converts independent candidate predictions into a connected route.

| System | Top-1 accuracy | Plausible continuity | Route F1 |
|---|---:|---:|---:|
| XGBoost classifier only | 89.4% | 82.7% | 80.7% |
| XGBoost + default HMM | 93.7% | 99.1% | 95.1% |
| XGBoost + tuned HMM | **95.1%** | **99.4%** | **95.7%** |

Final tuned HMM parameters:

```yaml
beta: 3
uturn_m: 25
w_e: 0.5
gap_s: 60
```

## 9. Final results

On the 10 student-test trips, the deployed pipeline achieved:

```text
Top-1 accuracy:        95.1%
Strict accuracy:       94.9%
Plausible continuity:  99.4%
Route F1:              95.7%
```

## 10. Distributed deployment

### Person 1 — macOS backend

Person 1 runs the model, HMM decoder, trip data, and FastAPI service.

```bash
cd "/Users/kasissnu/Downloads/ml hackathon/team32"
source .venv/bin/activate
export PATHSNAP_TRIPS_DIR="../Student folder/student_test"
export PATHSNAP_ARTIFACTS="$PWD/artifacts"
export PYTHONPATH="$PWD/src"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Person 2 — Windows frontend

Person 2 runs only the Streamlit UI and connects to Person 1's IP address.

```powershell
cd "C:\path\to\team32"
.\.venv\Scripts\Activate.ps1
$env:PATHSNAP_API = "http://PERSON_1_MAC_IP:8000"
python -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Both laptops must be on the same Wi-Fi network.

## 11. API testing commands

From Person 2's Windows laptop:

```powershell
curl.exe http://PERSON_1_MAC_IP:8000/health
curl.exe http://PERSON_1_MAC_IP:8000/trips
curl.exe -X POST http://PERSON_1_MAC_IP:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"trip_id":"00000000"}'
```

Expected `/health` output includes:

```json
{
  "status": "ok",
  "model": "XGBoost (Optuna-tuned)"
}
```

## 12. Short presentation script

> Our project solves map matching for noisy GPS data. First, we generate nearby road candidates using a KD-tree. Then we calculate 33 spatial, directional, kinematic, and connectivity features for every GPS-candidate pair. We compare five machine-learning models and use XGBoost as the final classifier because it provides the strongest overall performance and can be tuned effectively with Optuna. Finally, we pass the classifier probabilities to an HMM and use Viterbi decoding to select a globally connected road sequence. This improves the result from about 89.4% classifier-only accuracy to 95.1% final accuracy, with 99.4% plausible route continuity and 95.7% route F1. The FastAPI backend runs on one laptop, while the Streamlit frontend runs on the second laptop and visualizes the reconstructed route.

## 13. Transparency note

To save time, Team 32 carried forward the completed PT-044 model comparison values and final artifacts. The Team 32 code, API, frontend, and local student-test pipeline were then verified with the available data.

The repository test suite passes all 11 tests.
