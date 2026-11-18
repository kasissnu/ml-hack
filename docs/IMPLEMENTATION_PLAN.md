# Team 32 PathSnap Implementation Plan

This is the working plan for Person 1's ML pipeline. Update the status and record decisions as the implementation progresses.

## Shared rules

- Use `data/raw/train` for fitting and group-aware cross-validation.
- Use `data/raw/student_test` only for development comparison.
- Never use evaluator-testing observations or labels for training, feature selection, tuning, or threshold selection.
- Keep all experiments reproducible with saved configurations, random seeds, metrics, and timings.
- Keep the public prediction boundary in `contracts/prediction_contract.py` stable for Person 2.

## Phase 1: Data loading, validation, and exploration

Status: complete

1. Load `.track`, `.nodes`, `.arcs`, and `.route` files with explicit schemas.
2. Inventory trip counts, row counts, file availability, and file sizes.
3. Validate coordinates, timestamps, duplicate points, missing values, and time gaps.
4. Validate node and arc indices and basic road-network connectivity.
5. Inspect route length and route arc validity on representative training trips.
6. Determine and document how route arcs align with GPS observations.
7. Save EDA tables and representative maps under `reports/eda/`.

Exit criteria: the data schema and route-label alignment method are documented and reproducible. Completed in the Phase 1 audit.

## Phase 2: Candidate generation

1. Convert coordinates to a suitable metric coordinate system.
2. Generate the nearest `K` road-arc candidates per GPS observation.
3. Measure candidate miss rate using training reference routes.
4. Save candidate tables in a format that supports model training and inference.

Exit criteria: candidate generation works on every training trip and has an explicit miss-rate report.

## Phase 3: Feature engineering

Create leakage-free features: point-to-segment distance, along-segment position, GPS speed, heading difference, segment length, time gaps, and previous-candidate connectivity.

Exit criteria: one feature schema is shared by training and prediction code.

## Phase 4: Model comparison

Train and compare at least five models using trip-level grouped cross-validation:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. HistGradientBoosting
5. MLP

Record Top-1 accuracy, Macro F1, log loss where applicable, candidate miss rate, route continuity, training time, inference time, configuration, and seed.

Exit criteria: comparison table and actual-result plots justify the selected model.

## Phase 5: Bayesian optimisation

1. Select the strongest model from Phase 4.
2. Define a grouped cross-validation objective and search space.
3. Run and retain every Bayesian optimisation trial.
4. Refit the best configuration using permitted training data.
5. Compare tuned and untuned models on `student_test`.

Exit criteria: best parameters, trial log, computational cost, and tuned-versus-untuned results are saved.

## Phase 6: HMM and Viterbi decoding

1. Treat candidate arcs as HMM states.
2. Use model probabilities as emission scores.
3. Use road connectivity, distance, heading, and time plausibility as transition scores.
4. Decode the best connected route with Viterbi.
5. Compare independent predictions with decoded routes.

Exit criteria: route continuity improves or the failure cases are measured and explained.

## Phase 7: Integration and artifacts

1. Export the model, preprocessor, candidate configuration, road network, and decoder settings.
2. Implement `predict_route(trip_id)` behind the shared contract.
3. Give Person 2 a stable artifact-loading and prediction interface.
4. Prepare reproducible commands for training and inference.

## Phase 8: Report evidence

Generate all report tables and plots from saved experiment outputs. Include EDA, assumptions, feature decisions, five-model comparison, Bayesian optimisation, HMM results, limitations, and reproducibility instructions.

## Decision log

Record decisions here when the data requires a choice that affects downstream work.

| Date | Decision | Evidence | Impact |
|---|---|---|---|
| 2026-11-18 | Initial repository and phase plan created | Team 32 scaffold and supplied dataset inventory | Starts Phase 1 |
