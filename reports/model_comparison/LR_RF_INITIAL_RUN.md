# Initial Logistic Regression and Random Forest Runs

The two models are trained as binary candidate rankers. Each candidate row is positive when its arc matches the provisional aligned reference arc. Rows from observations whose reference arc is absent from the candidate set remain part of the miss-rate report and do not create a false positive label.

## Logistic Regression

- Candidate count: `K=24`
- Negative sampling: 3 negatives per available positive
- Grouped cross-validation: 5 folds by trip
- Training matrix cap: 600,000 rows
- StandardScaler followed by balanced Logistic Regression

The full run completed and wrote `experiments/results/logistic_regression_results.csv` locally. The student-test metrics were:

- Top-1 accuracy: 0.597
- Candidate Macro F1: 0.599
- Candidate log loss: 0.361
- Candidate miss rate: 0.069

## Random Forest

- Candidate count: `K=24`
- Negative sampling: 3 negatives per available positive
- Random Forest: 20 trees, maximum depth 12, minimum leaf size 5
- Single-process training to limit memory use
- Training matrix cap: 600,000 rows

The full RF run completed across all 80 training trips with five grouped folds. The student-test metrics were:

- Top-1 accuracy: 0.648
- Candidate Macro F1: 0.723
- Candidate log loss: 0.218
- Candidate miss rate: 0.069

The RF artifact is an initial resource-bounded baseline, not the final tuned RF configuration.

## Interpretation

The current LR and RF results are reproducible full training-partition baselines under the same candidate and row-cap settings. RF is stronger on these initial metrics, but route continuity and HMM decoding still need to be evaluated before final selection.
