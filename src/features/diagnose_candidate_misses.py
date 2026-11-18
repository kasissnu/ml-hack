
import pandas as pd
from pathlib import Path

TRIP_ID = "00000001"

file = (
    Path("data/processed/training")
    / f"{TRIP_ID}_training_data.csv"
)

df = pd.read_csv(file)

# Find observations where the reference-route proxy
# was not present among the generated candidates.
summary = df.groupby("observation_id").agg(
    label_arc_id=("label_arc_id", "first"),
    reference_distance_m=("reference_distance_m", "first"),
    min_candidate_distance_m=("distance_m", "min"),
    max_candidate_distance_m=("distance_m", "max"),
    positive_candidates=("label", "sum"),
    candidate_count=("arc_id", "count")
).reset_index()

misses = summary[summary["positive_candidates"] == 0].copy()

print("=" * 65)
print("PATHSNAP - CANDIDATE MISS DIAGNOSTICS")
print("=" * 65)

print("Total observations:", len(summary))
print("Missed observations:", len(misses))
print(
    "Miss percentage:",
    round(len(misses) / len(summary) * 100, 2)
)

print("\nFIRST 20 MISSED OBSERVATIONS")
print(misses.head(20).to_string(index=False))

print("\nDISTANCE SUMMARY FOR MISSES")
if not misses.empty:
    print(
        misses[
            ["reference_distance_m", "min_candidate_distance_m"]
        ].describe().round(3)
    )

output_dir = Path("reports")
output_dir.mkdir(exist_ok=True)

misses.to_csv(
    output_dir / f"{TRIP_ID}_candidate_misses.csv",
    index=False
)

print("\nMiss report saved successfully.")
