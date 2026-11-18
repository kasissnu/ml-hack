
from pathlib import Path
import pandas as pd

from candidate_generation import load_trip, generate_candidates


TRAIN_DIR = Path("data/raw/train")
OUTPUT_DIR = Path("data/processed/candidates")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    trip_dirs = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    )

    print("=" * 65)
    print("PATHSNAP - ALL TRAINING TRIP CANDIDATE GENERATION")
    print("=" * 65)

    print("Total training trips:", len(trip_dirs))

    results = []

    for index, trip_dir in enumerate(trip_dirs, start=1):

        trip_id = trip_dir.name

        try:
            track, nodes, arcs = load_trip(trip_dir)

            candidates = generate_candidates(
                track,
                nodes,
                arcs,
                radius_m=100,
                top_n=8
            )

            output_file = (
                OUTPUT_DIR / f"{trip_id}_candidates.csv"
            )

            candidates.to_csv(output_file, index=False)

            counts = candidates.groupby("observation_id").size()

            result = {
                "trip_id": trip_id,
                "gps_observations": len(track),
                "road_nodes": len(nodes),
                "road_arcs": len(arcs),
                "candidate_records": len(candidates),
                "mean_candidates_per_observation": counts.mean(),
                "min_candidates_per_observation": counts.min(),
                "max_candidates_per_observation": counts.max(),
                "observations_with_fewer_than_8": int(
                    (counts < 8).sum()
                ),
                "generation_status": "SUCCESS"
            }

            results.append(result)

            print(
                f"[{index}/{len(trip_dirs)}] "
                f"{trip_id}: {len(candidates)} candidates"
            )

        except Exception as error:

            print(f"[{index}/{len(trip_dirs)}] "
                  f"{trip_id}: FAILED - {error}")

            results.append({
                "trip_id": trip_id,
                "generation_status": f"FAILED: {error}"
            })

    summary = pd.DataFrame(results)

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    report_file = (
        report_dir / "all_training_candidate_summary.csv"
    )

    summary.to_csv(report_file, index=False)

    successful = summary[
        summary["generation_status"] == "SUCCESS"
    ]

    print("\n" + "=" * 65)
    print("GENERATION SUMMARY")
    print("=" * 65)

    print("Successful trips:", len(successful))
    print("Failed trips:", len(summary) - len(successful))

    if not successful.empty:
        print(
            "Total GPS observations:",
            successful["gps_observations"].sum()
        )

        print(
            "Total candidate records:",
            successful["candidate_records"].sum()
        )

        print(
            "Total observations with fewer than 8 candidates:",
            successful["observations_with_fewer_than_8"].sum()
        )

    print("\nSummary saved to:", report_file)


if __name__ == "__main__":
    main()
