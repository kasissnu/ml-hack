"""Training and evaluation utilities for Logistic Regression and Random Forest."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.loader import load_trip, list_trip_ids, resolve_data_root
from features.build_features import FEATURE_COLUMNS, add_features, split_features_and_label
from features.candidates import generate_candidates


@dataclass(frozen=True)
class TrainingConfig:
    k: int = 24
    negative_ratio: int = 3
    cv_folds: int = 5
    random_state: int = 32
    max_training_rows: int | None = 600_000


def make_model(name: str, random_state: int = 32):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=20,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {name}")


def _sample_training_candidates(candidates: pd.DataFrame, negative_ratio: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, group in candidates.groupby("observation_index", sort=False):
        positives = group[group["reference_in_candidates"]]
        if positives.empty:
            continue
        negatives = group[~group["reference_in_candidates"]]
        count = min(len(negatives), negative_ratio * len(positives))
        if count:
            negatives = negatives.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
        rows.append(pd.concat([positives, negatives], ignore_index=True))
    return pd.concat(rows, ignore_index=True) if rows else candidates.iloc[0:0].copy()


def _cap_trip_rows(candidates: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(candidates) <= max_rows:
        return candidates
    positives = candidates[candidates["reference_in_candidates"]]
    negatives = candidates[~candidates["reference_in_candidates"]]
    positive_budget = min(len(positives), max_rows // 4)
    negative_budget = max_rows - positive_budget
    positive_sample = positives.sample(
        n=positive_budget,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    negative_sample = negatives.sample(
        n=min(len(negatives), negative_budget),
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    return pd.concat([positive_sample, negative_sample], ignore_index=True)


def build_training_data(
    partition: str = "train",
    data_root: str | Path | None = None,
    config: TrainingConfig = TrainingConfig(),
    trip_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    root = resolve_data_root(data_root)
    ids = trip_ids or list_trip_ids(partition, root)
    rng = np.random.default_rng(config.random_state)
    per_trip_limit = None
    if config.max_training_rows is not None:
        per_trip_limit = max(1000, config.max_training_rows // max(len(ids), 1))
    feature_frames: list[pd.DataFrame] = []
    labels: list[pd.Series] = []
    groups: list[pd.Series] = []
    summaries = []
    for trip_id in ids:
        trip = load_trip(trip_id, partition, root)
        candidates = generate_candidates(trip, k=config.k)
        sampled = _sample_training_candidates(candidates, config.negative_ratio, rng)
        if per_trip_limit is not None:
            sampled = _cap_trip_rows(sampled, per_trip_limit, rng)
        if sampled.empty:
            continue
        enriched = add_features(trip, sampled)
        features, target = split_features_and_label(enriched)
        feature_frames.append(features)
        labels.append(target.astype(int))
        groups.append(pd.Series([trip_id] * len(target)))
        summaries.append(
            {
                "partition": partition,
                "trip_id": trip_id,
                "observations": candidates["observation_index"].nunique(),
                "candidate_rows": len(candidates),
                "training_rows": len(sampled),
                "candidate_miss_rate": float(1.0 - candidates.groupby("observation_index")["reference_in_candidates"].any().mean()),
            }
        )
    features = pd.concat(feature_frames, ignore_index=True)
    targets = pd.concat(labels, ignore_index=True)
    group_series = pd.concat(groups, ignore_index=True)
    if config.max_training_rows is not None and len(features) > config.max_training_rows:
        rng = np.random.default_rng(config.random_state)
        positive_indices = np.flatnonzero(targets.to_numpy() == 1)
        negative_indices = np.flatnonzero(targets.to_numpy() == 0)
        positive_budget = min(len(positive_indices), config.max_training_rows // 4)
        negative_budget = config.max_training_rows - positive_budget
        selected_positive = rng.choice(positive_indices, size=positive_budget, replace=False)
        selected_negative = rng.choice(negative_indices, size=min(len(negative_indices), negative_budget), replace=False)
        selected = np.sort(np.r_[selected_positive, selected_negative])
        features = features.iloc[selected].reset_index(drop=True)
        targets = targets.iloc[selected].reset_index(drop=True)
        group_series = group_series.iloc[selected].reset_index(drop=True)
    return features, targets, group_series, pd.DataFrame(summaries)


def _score_predictions(model, enriched: pd.DataFrame) -> dict[str, float]:
    features, target = split_features_and_label(enriched)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    top1_correct = 0
    observation_count = 0
    miss_count = 0
    for _, group in enriched.assign(_probability=probabilities).groupby("observation_index"):
        observation_count += 1
        if not group["reference_in_candidates"].any():
            miss_count += 1
            continue
        best = group.iloc[int(np.argmax(group["_probability"].to_numpy()))]
        top1_correct += int(bool(best["reference_in_candidates"]))
    return {
        "candidate_macro_f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "candidate_log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "top1_accuracy": float(top1_correct / observation_count) if observation_count else 0.0,
        "candidate_miss_rate": float(miss_count / observation_count) if observation_count else 1.0,
        "observations": float(observation_count),
    }


def evaluate_on_partition(model, partition: str, data_root: str | Path | None, config: TrainingConfig) -> dict[str, float]:
    root = resolve_data_root(data_root)
    scores = []
    for trip_id in list_trip_ids(partition, root):
        trip = load_trip(trip_id, partition, root)
        enriched = add_features(trip, generate_candidates(trip, k=config.k))
        scores.append(_score_predictions(model, enriched))
    frame = pd.DataFrame(scores)
    weighted = frame["observations"]
    return {
        key: float(np.average(frame[key], weights=weighted))
        for key in ("candidate_macro_f1", "candidate_log_loss", "top1_accuracy", "candidate_miss_rate")
    }


def train_and_evaluate(
    model_name: str,
    data_root: str | Path | None = None,
    config: TrainingConfig = TrainingConfig(),
    artifact_dir: str | Path = "artifacts",
    results_dir: str | Path = "experiments/results",
    max_train_trips: int | None = None,
) -> pd.DataFrame:
    result_path = Path(results_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    trip_ids = list_trip_ids("train", resolve_data_root(data_root))
    if max_train_trips is not None:
        trip_ids = trip_ids[:max_train_trips]
    X, y, groups, summary = build_training_data("train", data_root, config, trip_ids=trip_ids)
    summary.to_csv(result_path / f"{model_name}_trip_summary.csv", index=False)
    splitter = GroupKFold(n_splits=config.cv_folds)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model(model_name, config.random_state)
        started = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_seconds = time.perf_counter() - started
        valid_probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        valid_predictions = (valid_probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "macro_f1": f1_score(y.iloc[valid_idx], valid_predictions, average="macro", zero_division=0),
                "log_loss": log_loss(y.iloc[valid_idx], valid_probabilities, labels=[0, 1]),
                "train_seconds": train_seconds,
                "valid_rows": len(valid_idx),
            }
        )
    final_model = make_model(model_name, config.random_state)
    started = time.perf_counter()
    final_model.fit(X, y)
    final_train_seconds = time.perf_counter() - started
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": FEATURE_COLUMNS, "config": config}, artifact_path / f"{model_name}.joblib")
    student_scores = evaluate_on_partition(final_model, "student_test", data_root, config)
    result_frame = pd.DataFrame(rows)
    result_frame["student_test_top1_accuracy"] = student_scores["top1_accuracy"]
    result_frame["student_test_candidate_macro_f1"] = student_scores["candidate_macro_f1"]
    result_frame["student_test_log_loss"] = student_scores["candidate_log_loss"]
    result_frame["student_test_candidate_miss_rate"] = student_scores["candidate_miss_rate"]
    result_frame["final_train_seconds"] = final_train_seconds
    result_frame["k"] = config.k
    result_frame["negative_ratio"] = config.negative_ratio
    result_frame["cv_folds"] = config.cv_folds
    result_frame["max_training_rows"] = config.max_training_rows
    result_frame.to_csv(result_path / f"{model_name}_results.csv", index=False)
    return result_frame
