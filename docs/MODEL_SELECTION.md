# Team 32 selected five models

The PT-044 model comparison already contains the completed five fold grouped-CV and student-test results. We are carrying forward the five strongest model families and their recorded configurations so the Team 32 pipeline can continue from the completed experiment stage.

| Selected model | Why selected | Recorded student-test Top-1 | Recorded student-test Macro F1 | Recorded log loss |
|---|---|---:|---:|---:|
| LightGBM | Highest CV Macro F1 and fast inference | 0.889 | 0.935 | 0.083 |
| XGBoost | Strongest overall boosted model and final deployable classifier | 0.897 | 0.937 | 0.082 |
| Random Forest | Strong bagging baseline with high validation performance | 0.885 | 0.933 | 0.086 |
| HistGradientBoosting | Fast, strong histogram boosting family | 0.888 | 0.933 | 0.097 |
| MLP | Neural network family with competitive accuracy and fast inference | 0.890 | 0.930 | 0.093 |

The final classifier is the Optuna-tuned XGBoost model. Its recorded parameters are:

```yaml
n_estimators: 800
learning_rate: 0.032784293445229554
max_depth: 11
min_child_weight: 4.4591431531478065
subsample: 0.778630780417708
colsample_bytree: 0.9803051453707186
gamma: 0.9443303958965052
reg_alpha: 9.109852884180164e-06
reg_lambda: 0.002019781049572206
scale_pos_weight: 2.3103591511849535
```

The tuned decoder parameters are:

```yaml
beta: 3
uturn_m: 25
w_e: 0.5
gap_s: 60
```

The source for the carried-forward values is recorded in `results/models/comparison.csv`, `results/optuna/best_params.json`, and `results/hmm/best_params.json`. No new model comparison is required before continuing to decoder integration, artifact export, API, UI, and report generation.
