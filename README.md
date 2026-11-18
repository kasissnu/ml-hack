# PathSnap - Team 32

Two-person implementation of the PathSnap map-matching and route-reconstruction system.

## Work split

- **Person 1:** ML pipeline, features, model experiments, tuning, HMM/Viterbi. See `src/data`, `src/features`, `src/models`, and `src/decoder`.
- **Person 2:** API, UI, visualisation, and deployment. See `src/api`, `src/ui`, and `deployment`.
- **Shared boundary:** `contracts/prediction_contract.py`.

Read `CONTRIBUTING.md` before starting work.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the supplied dataset locally under `data/raw/`. Do not commit raw or evaluator data.

## Data partitions

- `train/`: training and group-aware validation
- `student_test/`: development comparison only
- evaluator-testing data: never used for training, feature selection, or tuning

## Current status

Repository skeleton created. Person 1 should begin with dataset inventory and validation. Person 2 can begin with the API/UI skeleton using the prediction contract and mock responses.
