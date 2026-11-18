
from pathlib import Path
import pandas as pd

REPORT = Path("reports/all_trip_candidate_validation.csv")
TRAIN_DIR = Path("data/raw/train")

report = pd.read_csv(REPORT)

print("=" * 65)
print("PATHSNAP - PROBLEM TRIP DIAGNOSTICS")
print("=" * 65)

print("\nREPORT COLUMNS:")
print(report.columns.tolist())

print("\nPROBLEM TRIPS:")
problem_ids = ["00000054", "00000033", "00000072", "00000002", "00000001"]

report["trip_id"] = report["trip_id"].astype(str).str.zfill(8)

print(
    report[report["trip_id"].isin(problem_ids)]
    .sort_values("trip_id")
    .to_string(index=False)
)

print("\n" + "=" * 65)
print("DATASET FILE COUNTS")
print("=" * 65)

for trip_id in problem_ids:
    trip_dir = TRAIN_DIR / trip_id

    print(f"\nTRIP: {trip_id}")

    if not trip_dir.exists():
        print("  ERROR: Trip folder not found")
        continue

    for extension in ["track", "nodes", "arcs", "route"]:
        file_path = trip_dir / f"{trip_id}.{extension}"

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as file:
                count = sum(1 for _ in file)

            print(f"  {extension}: {count} rows")
        else:
            print(f"  {extension}: MISSING")

print("\n" + "=" * 65)
print("DIAGNOSTICS COMPLETED")
print("=" * 65)
