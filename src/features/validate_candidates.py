
from pathlib import Path
import pandas as pd
import numpy as np

TRIP_ID = "00000001"

trip_dir = Path("data/raw/train") / TRIP_ID
candidate_file = (
    Path("data/processed/candidates")
    / f"{TRIP_ID}_candidates.csv"
)

# Load original road network and reference route
arcs = pd.read_csv(
    trip_dir / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None
).to_numpy(dtype=int)

route = pd.read_csv(
    trip_dir / f"{TRIP_ID}.route",
    sep=r"\s+",
    header=None
).iloc[:, 0].to_numpy(dtype=int)

candidates = pd.read_csv(candidate_file)

# Check route arc IDs
valid_route_mask = (
    (route >= 0) & (route < len(arcs))
)

valid_route = route[valid_route_mask]
invalid_route_count = int((~valid_route_mask).sum())

# Candidate arc IDs generated across the entire trip
candidate_arc_ids = set(
    candidates["arc_id"].astype(int).unique()
)

# Route-level candidate coverage
covered_mask = np.array([
    arc_id in candidate_arc_ids
    for arc_id in valid_route
])

covered_entries = int(covered_mask.sum())
total_valid_entries = len(valid_route)

coverage = (
    covered_entries / total_valid_entries * 100
    if total_valid_entries else 0
)

unique_route_arcs = set(valid_route)
unique_covered_arcs = unique_route_arcs & candidate_arc_ids

unique_coverage = (
    len(unique_covered_arcs) / len(unique_route_arcs) * 100
    if unique_route_arcs else 0
)

# Check continuity of the reference route
if len(valid_route) > 1:
    consecutive_arcs = arcs[valid_route]

    continuity_mask = (
        consecutive_arcs[:-1, 1]
        == consecutive_arcs[1:, 0]
    )

    continuous_pairs = int(continuity_mask.sum())
    total_pairs = len(continuity_mask)

    continuity_percentage = (
        continuous_pairs / total_pairs * 100
        if total_pairs else 0
    )
else:
    continuous_pairs = 0
    total_pairs = 0
    continuity_percentage = 0

print("=" * 60)
print("PATHSNAP - CANDIDATE VALIDATION")
print("=" * 60)

print("\nROUTE VALIDATION")
print("Total reference route entries:", len(route))
print("Invalid route arc IDs:", invalid_route_count)

print("\nCANDIDATE COVERAGE")
print("Unique generated candidate arcs:", len(candidate_arc_ids))
print("Unique reference route arcs:", len(unique_route_arcs))
print("Covered unique reference arcs:", len(unique_covered_arcs))
print(f"Unique route coverage: {unique_coverage:.2f}%")

print("\nROUTE ENTRY COVERAGE")
print("Covered route entries:", covered_entries)
print("Valid route entries:", total_valid_entries)
print(f"Route-entry coverage: {coverage:.2f}%")

print("\nREFERENCE ROUTE CONTINUITY")
print("Connected consecutive arc pairs:", continuous_pairs)
print("Total consecutive pairs:", total_pairs)
print(f"Continuity: {continuity_percentage:.2f}%")

# Save a validation report
report = pd.DataFrame([{
    "trip_id": TRIP_ID,
    "route_entries": len(route),
    "invalid_route_ids": invalid_route_count,
    "unique_route_arcs": len(unique_route_arcs),
    "unique_candidate_arcs": len(candidate_arc_ids),
    "covered_unique_route_arcs": len(unique_covered_arcs),
    "unique_route_coverage_pct": unique_coverage,
    "route_entry_coverage_pct": coverage,
    "connected_pairs": continuous_pairs,
    "total_pairs": total_pairs,
    "route_continuity_pct": continuity_percentage
}])

output = Path("reports")
output.mkdir(exist_ok=True)

report.to_csv(
    output / f"{TRIP_ID}_candidate_validation.csv",
    index=False
)

print("\nValidation report saved successfully!")
