
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# Accept trip ID from command line.
# Default trip is 00000001.
TRIP_ID = sys.argv[1] if len(sys.argv) > 1 else "00000001"

DATA_DIR = ROOT / "data" / "raw" / "train"
FEATURE_DIR = ROOT / "data" / "processed" / "temporal_features"

OUTPUT_DIR = ROOT / "data" / "processed" / "alignment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = FEATURE_DIR / f"{TRIP_ID}_features.csv"
ROUTE_FILE = DATA_DIR / TRIP_ID / f"{TRIP_ID}.route"
OUTPUT_FILE = OUTPUT_DIR / f"{TRIP_ID}_alignment.csv"


# ============================================================
# LOAD REFERENCE ROUTE
# ============================================================

def load_route():

    if not ROUTE_FILE.exists():
        raise FileNotFoundError(
            f"Reference route not found: {ROUTE_FILE}"
        )

    route_df = pd.read_csv(
        ROUTE_FILE,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )

    return route_df.iloc[:, 0].astype(int).tolist()


# ============================================================
# EMISSION COST
# ============================================================

def emission_cost(row):

    distance = float(row["distance_m"])
    heading = float(row["heading_difference_deg"])

    if not np.isfinite(heading):
        heading = 90.0

    return distance / 25.0 + heading / 90.0


# ============================================================
# DYNAMIC PROGRAMMING ALIGNMENT
# ============================================================

def align_trip():

    print("\nStarting route alignment...")
    print("Trip ID:", TRIP_ID)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    candidates = pd.read_csv(FEATURE_FILE)
    route = load_route()

    required_columns = {
        "observation_id",
        "arc_id",
        "distance_m",
        "heading_difference_deg",
        "candidate_rank"
    }

    missing_columns = required_columns - set(candidates.columns)

    if missing_columns:
        raise ValueError(
            f"Missing candidate columns: {missing_columns}"
        )

    # Map each road arc to every occurrence in the reference route.
    arc_positions = {}

    for position, arc_id in enumerate(route):
        arc_positions.setdefault(arc_id, []).append(position)

    observations = []

    for obs_id, group in candidates.groupby(
        "observation_id", sort=True
    ):

        states = []

        for _, row in group.iterrows():

            arc_id = int(row["arc_id"])

            if arc_id not in arc_positions:
                continue

            for position in arc_positions[arc_id]:

                states.append({
                    "position": position,
                    "arc_id": arc_id,
                    "cost": emission_cost(row),
                    "distance_m": float(row["distance_m"]),
                    "candidate_rank": int(row["candidate_rank"])
                })

        observations.append({
            "observation_id": int(obs_id),
            "states": states
        })

    total_observations = len(observations)

    if not observations:
        raise RuntimeError(
            "No GPS observations were found."
        )

    # --------------------------------------------------------
    # Dynamic programming
    # --------------------------------------------------------

    dp = []
    backtrack = []

    first_states = observations[0]["states"]

    if not first_states:
        raise RuntimeError(
            "First observation has no reference-route candidate."
        )

    dp.append(
        np.array([state["cost"] for state in first_states])
    )

    backtrack.append(
        np.full(len(first_states), -1, dtype=int)
    )

    for i in range(1, total_observations):

        previous_states = observations[i - 1]["states"]
        current_states = observations[i]["states"]

        current_costs = []
        current_back = []

        if not previous_states:
            raise RuntimeError(
                f"Observation {observations[i-1]['observation_id']} "
                "has no route candidate. Alignment needs gap handling."
            )

        if not current_states:
            raise RuntimeError(
                f"Observation {observations[i]['observation_id']} "
                "has no route candidate. Alignment needs gap handling."
            )

        for current in current_states:

            best_cost = float("inf")
            best_index = -1

            for j, previous in enumerate(previous_states):

                progress = (
                    current["position"] - previous["position"]
                )

                if progress < 0:
                    transition_cost = 5.0 * abs(progress)
                else:
                    transition_cost = 0.02 * progress

                total_cost = (
                    dp[i - 1][j]
                    + transition_cost
                    + current["cost"]
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_index = j

            current_costs.append(best_cost)
            current_back.append(best_index)

        dp.append(np.array(current_costs))
        backtrack.append(np.array(current_back))

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    state_index = int(np.argmin(dp[-1]))

    selected = [None] * total_observations

    for i in range(total_observations - 1, -1, -1):

        selected[i] = observations[i]["states"][state_index]

        if i > 0:
            state_index = backtrack[i][state_index]

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    results = []

    for observation, state in zip(observations, selected):

        results.append({
            "observation_id": observation["observation_id"],
            "matched_arc_id": state["arc_id"],
            "route_position": state["position"],
            "distance_m": state["distance_m"],
            "alignment_status": "matched"
        })

    output = pd.DataFrame(results)

    output.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    matched = len(output)

    print("\n========== ALIGNMENT RESULTS ==========")
    print("Trip:", TRIP_ID)
    print("Total observations with route candidates:", total_observations)
    print("Matched:", matched)
    print("Mean matched distance:",
          round(output["distance_m"].mean(), 2), "metres")
    print("Route positions:",
          output["route_position"].min(),
          "to",
          output["route_position"].max())

    print("Backward transitions:",
          (output["route_position"].diff() < 0).sum())

    print("Zero-progress transitions:",
          (output["route_position"].diff() == 0).sum())

    print("Saved:", OUTPUT_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        align_trip()

    except Exception as error:
        print("\nALIGNMENT FAILED:")
        print(error)
        sys.exit(1)
