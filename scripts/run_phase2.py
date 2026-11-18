"""Measure candidate generation coverage across the permitted partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.loader import list_trip_ids, load_trip, resolve_data_root
from features.candidates import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output-dir", default="reports/model_comparison")
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    rows = []
    for partition in ("train", "student_test"):
        for trip_id in list_trip_ids(partition, root):
            candidates = generate_candidates(load_trip(trip_id, partition, root), k=args.k)
            by_observation = candidates.groupby("observation_index")
            rows.append(
                {
                    "partition": partition,
                    "trip_id": trip_id,
                    "k": args.k,
                    "observations": candidates["observation_index"].nunique(),
                    "candidate_rows": len(candidates),
                    "candidate_miss_count": int((~by_observation["reference_in_candidates"].any()).sum()),
                    "candidate_miss_rate": float((~by_observation["reference_in_candidates"].any()).mean()),
                    "median_point_distance_m": float(candidates.groupby("observation_index")["point_segment_distance_m"].min().median()),
                    "p95_point_distance_m": float(candidates.groupby("observation_index")["point_segment_distance_m"].min().quantile(0.95)),
                }
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"candidate_summary_k{args.k}.csv", index=False)
    frame.groupby("partition", as_index=False).agg(
        trips=("trip_id", "count"),
        observations=("observations", "sum"),
        candidate_miss_rate_mean=("candidate_miss_rate", "mean"),
        candidate_miss_rate_max=("candidate_miss_rate", "max"),
        median_point_distance_m_mean=("median_point_distance_m", "mean"),
        p95_point_distance_m_mean=("p95_point_distance_m", "mean"),
    ).to_csv(output_dir / f"candidate_partition_summary_k{args.k}.csv", index=False)
    print(f"Wrote candidate coverage results for {len(frame)} trips")


if __name__ == "__main__":
    main()
