
from pathlib import Path
import numpy as np
import pandas as pd

CANDIDATE_DIR = Path("data/processed/candidates")
TRAIN_DIR = Path("data/raw/train")

TRIPS = [
    "00000001",
    "00000033",
    "00000054",
    "00000072",
]


def inspect_trip(trip_id):
    print("\n" + "=" * 65)
    print(f"DISTANCE PATTERN: {trip_id}")
    print("=" * 65)

    path = CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    candidates = pd.read_csv(path)

    # Minimum candidate distance for each GPS observation.
    nearest = (
        candidates.groupby("observation_id")
        .agg(
            nearest_candidate_distance=("distance_m", "min"),
            candidate_count=("arc_id", "count"),
            timestamp=("timestamp", "first"),
        )
        .reset_index()
        .sort_values("observation_id")
    )

    print("\nNearest candidate distance distribution:")
    print(
        nearest["nearest_candidate_distance"]
        .describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        .round(2)
    )

    print("\nObservations grouped by nearest candidate distance:")

    bins = [-np.inf, 10, 20, 40, 60, 80, 100, np.inf]
    labels = [
        "<=10m", "10-20m", "20-40m", "40-60m",
        "60-80m", "80-100m", ">100m"
    ]

    nearest["distance_band"] = pd.cut(
        nearest["nearest_candidate_distance"],
        bins=bins,
        labels=labels
    )

    print(
        nearest["distance_band"]
        .value_counts(sort=False)
        .to_string()
    )

    print("\nCandidate counts per observation:")
    print(nearest["candidate_count"].value_counts().sort_index().to_string())

    print("\nDistance pattern across trip sections:")

    nearest["section"] = pd.qcut(
        nearest["observation_id"],
        q=10,
        duplicates="drop"
    )

    section_summary = nearest.groupby(
        "section", observed=True
    ).agg(
        observations=("observation_id", "count"),
        mean_distance=("nearest_candidate_distance", "mean"),
        median_distance=("nearest_candidate_distance", "median"),
        max_distance=("nearest_candidate_distance", "max"),
    )

    print(section_summary.round(2).to_string())

    # Show observations with the largest nearest-road distances.
    print("\n20 observations with the largest nearest candidate distances:")
    print(
        nearest.nlargest(20, "nearest_candidate_distance")[
            [
                "observation_id",
                "timestamp",
                "nearest_candidate_distance",
                "candidate_count",
            ]
        ].round(2).to_string(index=False)
    )


for trip in TRIPS:
    inspect_trip(trip)

print("\n" + "=" * 65)
print("DISTANCE PATTERN INSPECTION COMPLETED")
print("=" * 65)
