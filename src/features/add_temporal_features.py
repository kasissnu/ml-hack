
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

GPS_DIR = ROOT / "data" / "raw" / "train"
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates"

OUTPUT_DIR = ROOT / "data" / "processed" / "temporal_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000


# =====================================================
# GEOGRAPHIC CALCULATIONS
# =====================================================

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def calculate_bearing(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def circular_difference(angle1, angle2):

    difference = (angle1 - angle2 + 180) % 360 - 180

    return np.abs(difference)


# =====================================================
# LOAD GPS DATA
# =====================================================

def load_gps(trip_id):

    path = GPS_DIR / trip_id / f"{trip_id}.track"

    gps = pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )

    gps = gps.iloc[:, :3].copy()

    gps.columns = [
        "longitude",
        "latitude",
        "timestamp"
    ]

    gps = gps.apply(
        pd.to_numeric,
        errors="coerce"
    )

    gps = gps.dropna().reset_index(drop=True)

    return gps


# =====================================================
# TEMPORAL FEATURE ENGINEERING
# =====================================================

def generate_temporal_features(gps):

    gps = gps.copy()

    # Ensure original observation order is preserved
    gps["observation_id"] = np.arange(len(gps))

    # Time difference from previous observation
    gps["time_diff_s"] = gps["timestamp"].diff()

    # Distance travelled between consecutive observations
    gps["gps_step_distance_m"] = haversine(
        gps["longitude"].shift(),
        gps["latitude"].shift(),
        gps["longitude"],
        gps["latitude"]
    )

    # First observation has no previous movement
    gps.loc[0, "gps_step_distance_m"] = 0

    # Speed calculation
    valid_time = gps["time_diff_s"] > 0

    gps["gps_speed_mps"] = np.nan

    gps.loc[valid_time, "gps_speed_mps"] = (
        gps.loc[valid_time, "gps_step_distance_m"]
        / gps.loc[valid_time, "time_diff_s"]
    )

    # Calculate movement bearing
    gps["movement_bearing_deg"] = calculate_bearing(
        gps["longitude"].shift(),
        gps["latitude"].shift(),
        gps["longitude"],
        gps["latitude"]
    )

    # First observation has no previous heading
    gps.loc[0, "movement_bearing_deg"] = np.nan

    # Change in heading
    gps["gps_heading_change_deg"] = circular_difference(
        gps["movement_bearing_deg"],
        gps["movement_bearing_deg"].shift()
    )

    # First two observations have no heading change
    gps.loc[0, "gps_heading_change_deg"] = np.nan

    # Acceleration = change in speed / elapsed time
    gps["gps_acceleration_mps2"] = (
        gps["gps_speed_mps"].diff()
        / gps["time_diff_s"]
    )

    # First observations cannot have acceleration
    gps.loc[0, "gps_acceleration_mps2"] = np.nan

    return gps


# =====================================================
# PROCESS ONE TRIP
# =====================================================

def process_trip(trip_id):

    print(f"\nProcessing trip: {trip_id}")

    gps_path = GPS_DIR / trip_id / f"{trip_id}.track"

    candidate_path = (
        CANDIDATE_DIR / f"{trip_id}_candidates.csv"
    )

    if not gps_path.exists():
        print("GPS file missing. Skipping.")
        return

    if not candidate_path.exists():
        print("Candidate file missing. Skipping.")
        return

    gps = load_gps(trip_id)

    candidates = pd.read_csv(candidate_path)

    required_columns = {
        "observation_id",
        "timestamp",
        "arc_id",
        "distance_m"
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {missing}"
        )

    temporal = generate_temporal_features(gps)

    feature_columns = [
        "observation_id",
        "gps_step_distance_m",
        "time_diff_s",
        "gps_speed_mps",
        "gps_acceleration_mps2",
        "gps_heading_change_deg",
        "movement_bearing_deg"
    ]

    # Add GPS-level features to every candidate
    enriched = candidates.merge(
        temporal[feature_columns],
        on="observation_id",
        how="left",
        validate="many_to_one"
    )

    # Verify observation timestamps match
    if "timestamp" in enriched.columns:

        mismatch = ~np.isclose(
            enriched["timestamp"],
            enriched["observation_id"].map(
                temporal.set_index("observation_id")["timestamp"]
            ),
            equal_nan=True
        )

        if mismatch.any():
            raise ValueError(
                "Candidate timestamps do not match GPS observations."
            )

    output_path = (
        OUTPUT_DIR / f"{trip_id}_features.csv"
    )

    enriched.to_csv(output_path, index=False)

    print("GPS observations:", len(gps))
    print("Candidate records:", len(candidates))
    print("Enriched records:", len(enriched))
    print("New features:", len(feature_columns) - 1)
    print("Saved:", output_path)


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n====================================")
    print("PATHSNAP - TEMPORAL FEATURE ENGINEERING")
    print("====================================")

    trip_folders = sorted(
        folder
        for folder in GPS_DIR.iterdir()
        if folder.is_dir()
        and (CANDIDATE_DIR / f"{folder.name}_candidates.csv").exists()
    )

    successful = 0
    failed = 0

    for index, folder in enumerate(trip_folders, 1):

        try:
            process_trip(folder.name)
            successful += 1

        except Exception as error:
            failed += 1
            print(f"FAILED: {folder.name}: {error}")

        print(
            f"Progress: {index}/{len(trip_folders)}"
        )

    print("\n========== SUMMARY ==========")
    print("Trips processed:", successful)
    print("Trips failed:", failed)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
