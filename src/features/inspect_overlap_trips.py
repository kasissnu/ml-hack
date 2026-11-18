
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REPORT = ROOT / "reports" / "candidate_route_overlap.csv"

df = pd.read_csv(REPORT, dtype={"trip_id": str})

print("\n========== OVERLAP DIAGNOSTICS ==========")

for trip_id in ["00000054", "00000033", "00000072", "00000002"]:

    row = df[df["trip_id"] == trip_id].iloc[0]

    print(f"\nTrip: {trip_id}")
    print("GPS observations:", row["gps_observations"])
    print("With route candidate:", row["observations_with_route_candidate"])
    print("Without route candidate:", row["observations_without_route_candidate"])
    print("Coverage:", row["route_candidate_coverage_pct"], "%")

print("\n========== OVERALL STATISTICS ==========")

print(df["route_candidate_coverage_pct"].describe())

print("\nTrips below 95%:")
print(
    df[df["route_candidate_coverage_pct"] < 95][
        ["trip_id", "route_candidate_coverage_pct"]
    ].to_string(index=False)
)
