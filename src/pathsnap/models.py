"""Model zoo for the candidate classifier.

Each entry: family, a factory returning an unfitted estimator (scalers inside a Pipeline so they
are fit on training folds only), the key configuration (for the report), and an optional
training-row subsample (whole GPS observations are kept together) for algorithms whose cost grows
super-linearly with the number of rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    family: str
    factory: Callable
    config: dict
    train_subsample_obs: int | None = None   # max number of GPS observations used for fitting
    notes: str = ""
    extra: dict = field(default_factory=dict)


def lgbm_params(seed: int) -> dict:
    return dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, verbose=-1)


def get_models(seed: int = 42) -> dict[str, ModelSpec]:
    specs = [
        ModelSpec("logreg", "Linear / probabilistic",
                  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
                  {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "scaling": "StandardScaler"}),
        ModelSpec("gaussian_nb", "Linear / probabilistic (generative)",
                  lambda: make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-9)),
                  {"var_smoothing": 1e-9}),
        ModelSpec("knn", "Instance-based",
                  lambda: make_pipeline(StandardScaler(),
                                        KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
                  {"n_neighbors": 25, "weights": "distance", "metric": "euclidean", "scaling": "StandardScaler"},
                  train_subsample_obs=20_000,
                  notes="fit on 20k observations (~115k rows): prediction cost grows with the reference set"),
        ModelSpec("random_forest", "Ensemble (bagging)",
                  lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                                 n_jobs=-1, random_state=seed),
                  {"n_estimators": 150, "min_samples_leaf": 5, "max_features": "sqrt"}),
        ModelSpec("hist_gb", "Boosting",
                  lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                         early_stopping=False, random_state=seed),
                  {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31}),
        ModelSpec("lightgbm", "Boosting",
                  lambda: lgb.LGBMClassifier(**lgbm_params(seed)),
                  {k: v for k, v in lgbm_params(seed).items() if k not in ("n_jobs", "verbose", "random_state")}),
        ModelSpec("xgboost", "Boosting",
                  lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                                            colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                            random_state=seed, eval_metric="logloss"),
                  {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
                   "colsample_bytree": 0.8, "tree_method": "hist"}),
        ModelSpec("mlp", "Neural network",
                  lambda: make_pipeline(StandardScaler(),
                                        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
                                                      learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                                      validation_fraction=0.1, n_iter_no_change=6,
                                                      random_state=seed)),
                  {"hidden_layers": [128, 64], "activation": "relu", "alpha": 1e-4, "batch_size": 512,
                   "lr": 1e-3, "max_iter": 60, "early_stopping": True},
                  train_subsample_obs=80_000,
                  notes="fit on 80k observations (~450k rows) to keep epoch time manageable"),
    ]
    return {s.name: s for s in specs}
