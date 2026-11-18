"""Train the Team 32 Logistic Regression and Random Forest baselines."""

from __future__ import annotations

import argparse

from models.baselines import TrainingConfig, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", choices=["logistic_regression", "random_forest", "both"], default="both")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-train-trips", type=int, default=None)
    args = parser.parse_args()
    config = TrainingConfig(k=args.k, negative_ratio=args.negative_ratio, cv_folds=args.cv_folds)
    model_names = ["logistic_regression", "random_forest"] if args.model == "both" else [args.model]
    for model_name in model_names:
        result = train_and_evaluate(model_name, args.data_root, config, max_train_trips=args.max_train_trips)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
