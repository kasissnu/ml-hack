
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "train"
CANDIDATE_DIR = ROOT / "data" / "processed" / "temporal_features"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def read_file(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )


def analyse_trip(trip_id):

    folder = DATA_DIR / trip_id

    route = read_file(
        folder / f"{trip_id}.route"
    ).iloc[:, 0].dropna().astype(int)

    candidates = pd.read_csv(
        CANDIDATE_DIR / f"{trip_id}_features.csv"
    )

    route_ids = set(route.tolist())

    # Candidate arc IDs that occur in the reference route
    candidates["route_match"] = (
        candidates["arc_id"].isin(route_ids)
    )

    matched_observations = (
        candidates.groupby("observation_id")["route_match"]
        .any()
    )

    return {
        "trip_id": trip_id,
        "gps_observations": len(matched_observations),
        "observations_with_route_candidate": int(
            matched_observations.sum()
        ),
        "observations_without_route_candidate": int(
            (~matched_observations).sum()
        ),
        "route_candidate_coverage_pct": round(
            matched_observations.mean() * 100, 2
        ),
        "candidate_records": len(candidates),
        "route_candidate_records": int(
            candidates["route_match"].sum()
        )
    }


def main():

    results = []

    for file in sorted(
        CANDIDATE_DIR.glob("*_features.csv")
    ):

        trip_id = file.stem.replace("_features", "")

        try:
            results.append(analyse_trip(trip_id))
            print("Processed:", trip_id)

        except Exception as error:
            print("Failed:", trip_id, error)

    df = pd.DataFrame(results)

    output = REPORT_DIR / "candidate_route_overlap.csv"

    df.to_csv(output, index=False)

    print("\n========== OVERLAP SUMMARY ==========")
    print("Trips analysed:", len(df))
    print(
        "Average route candidate coverage:",
        round(df["route_candidate_coverage_pct"].mean(), 2),
        "%"
    )

    print("\nLowest coverage trips:")
    print(
        df.nsmallest(
            10, "route_candidate_coverage_pct"
        ).to_string(index=False)
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
