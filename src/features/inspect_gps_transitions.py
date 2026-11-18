
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_DIR = Path("data/raw/train")
CANDIDATE_DIR = Path("data/processed/candidates")

# These ranges cover the suspicious sections.
RANGES = {
    "00000054": [(1850, 2050), (3800, 4000)],
    "00000033": [(2000, 2150), (5200, 5400)],
}


def inspect_range(trip_id, start, end):
    folder = TRAIN_DIR / trip_id

    track = pd.read_csv(
        folder / f"{trip_id}.track",
        sep=r"\s+",
        header=None
    )

    candidates = pd.read_csv(
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    gps = track.iloc[:, :3].copy()
    gps.columns = ["longitude", "latitude", "timestamp"]
    gps["observation_id"] = np.arange(len(gps))

    # Calculate consecutive GPS movement in approximate metres.
    lon = np.radians(gps["longitude"])
    lat = np.radians(gps["latitude"])

    dlat = np.diff(lat) * 6371000
    dlon = (
        np.diff(lon)
        * 6371000
        * np.cos((lat.iloc[:-1].to_numpy() + lat.iloc[1:].to_numpy()) / 2)
    )

    movement = np.sqrt(dlat**2 + dlon**2)

    gps["step_distance_m"] = np.r_[np.nan, movement]

    time_diff = gps["timestamp"].diff()
    gps["time_difference_s"] = time_diff

    gps["approx_speed_mps"] = (
        gps["step_distance_m"] / gps["time_difference_s"]
    )

    nearest = (
        candidates.groupby("observation_id")["distance_m"]
        .min()
        .rename("nearest_candidate_distance_m")
    )

    gps = gps.join(nearest, on="observation_id")

    section = gps[
        (gps["observation_id"] >= start)
        & (gps["observation_id"] <= end)
    ].copy()

    print("\n" + "=" * 75)
    print(f"TRIP {trip_id}: OBSERVATIONS {start} TO {end}")
    print("=" * 75)

    print("\nCoordinate and movement samples:")
    print(
        section.iloc[::10][
            [
                "observation_id",
                "longitude",
                "latitude",
                "timestamp",
                "step_distance_m",
                "time_difference_s",
                "approx_speed_mps",
                "nearest_candidate_distance_m",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nLargest consecutive GPS movements in this range:")
    print(
        section.nlargest(10, "step_distance_m")[
            [
                "observation_id",
                "longitude",
                "latitude",
                "timestamp",
                "step_distance_m",
                "time_difference_s",
                "approx_speed_mps",
                "nearest_candidate_distance_m",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nNearest candidate distance summary:")
    print(
        section["nearest_candidate_distance_m"]
        .describe()
        .round(2)
    )


for trip_id, ranges in RANGES.items():
    for start, end in ranges:
        inspect_range(trip_id, start, end)

print("\nGPS TRANSITION INSPECTION COMPLETED")
