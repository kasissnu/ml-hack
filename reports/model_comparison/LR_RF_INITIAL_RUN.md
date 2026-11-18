# Initial Logistic Regression and Random Forest Runs

The two models are trained as binary candidate rankers. Each candidate row is positive when its arc matches the provisional aligned reference arc. Rows from observations whose reference arc is absent from the candidate set remain part of the miss-rate report and do not create a false positive label.

## Logistic Regression

- Candidate count: `K=24`
- Negative sampling: 3 negatives per available positive
- Grouped cross-validation: 5 folds by trip
- Training matrix cap: 600,000 rows
- StandardScaler followed by balanced Logistic Regression

The full run completed and wrote `experiments/results/logistic_regression_results.csv` locally. The student-test metrics in that run were approximately:

- Top-1 accuracy: 0.592
- Candidate Macro F1: 0.571
- Candidate log loss: 0.411
- Candidate miss rate: 0.069

## Random Forest

- Candidate count: `K=24`
- Negative sampling: 3 negatives per available positive
- Random Forest: 20 trees, maximum depth 12, minimum leaf size 5
- Single-process training to limit memory use
- Bounded training run: 20,000 rows

The RF artifact and local result file were produced successfully. Its result file must be retained with the exact run configuration before using the numbers in the final report. This is an initial resource-bounded baseline, not the final tuned RF configuration.

## Interpretation

The current LR result is a reproducible full training-partition baseline. RF needs a larger controlled run or a more memory-efficient training setup before selecting between the models. Neither model should be chosen from these initial runs alone; route continuity and HMM decoding still need to be evaluated.
