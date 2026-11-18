"""Create a reproducible inventory of the supplied PathSnap partitions."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "Student folder"
OUT = Path(__file__).resolve().parents[1] / "reports" / "dataset_inventory.csv"


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def inventory_partition(partition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    partition_root = DATA_ROOT / partition
    for trip_dir in sorted(p for p in partition_root.iterdir() if p.is_dir()):
        trip_id = trip_dir.name
        record: dict[str, object] = {
            "team": 32,
            "partition": partition,
            "trip_id": trip_id,
        }
        for suffix in ("track", "nodes", "arcs", "route"):
            path = trip_dir / f"{trip_id}.{suffix}"
            record[f"{suffix}_present"] = path.exists()
            record[f"{suffix}_rows"] = count_rows(path) if path.exists() else 0
            record[f"{suffix}_bytes"] = path.stat().st_size if path.exists() else 0
        rows.append(record)
    return rows


def main() -> None:
    rows = inventory_partition("train") + inventory_partition("student_test")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["team", "partition", "trip_id"]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Team 32: wrote {len(rows)} trip records to {OUT}")


if __name__ == "__main__":
    main()
