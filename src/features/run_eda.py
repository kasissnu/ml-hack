
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "train"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "reports" / "eda_figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# CONFIGURATION
# =====================================================

EARTH_RADIUS_M = 6371000

# =====================================================
# DATA LOADING
# =====================================================

def read_data(path):
    return pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python"
    )


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians,
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

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# =====================================================
# TRIP ANALYSIS
# =====================================================

def analyse_trip(trip_folder):

    trip_id = trip_folder.name

    gps = read_data(trip_folder / f"{trip_id}.track")
    nodes = read_data(trip_folder / f"{trip_id}.nodes")
    arcs = read_data(trip_folder / f"{trip_id}.arcs")
    route = read_data(trip_folder / f"{trip_id}.route")

    gps = gps.iloc[:, :3].copy()
    gps.columns = ["lon", "lat", "timestamp"]

    nodes = nodes.iloc[:, :2].copy()
    nodes.columns = ["lon", "lat"]

    arcs = arcs.iloc[:, :2].copy()
    arcs.columns = ["from_node", "to_node"]

    route = pd.to_numeric(
        route.iloc[:, 0],
        errors="coerce"
    ).dropna().astype(int).tolist()

    gps = gps.apply(pd.to_numeric, errors="coerce").dropna()
    nodes = nodes.apply(pd.to_numeric, errors="coerce").dropna()
    arcs = arcs.apply(pd.to_numeric, errors="coerce").dropna()

    # GPS movement distances
    gps["step_distance_m"] = 0.0

    if len(gps) > 1:
        gps["step_distance_m"] = haversine(
            gps["lon"].shift(),
            gps["lat"].shift(),
            gps["lon"],
            gps["lat"]
        )

        gps.loc[gps.index[0], "step_distance_m"] = 0

    # Timestamp differences
    timestamps = gps["timestamp"].to_numpy(dtype=float)
    time_diffs = np.diff(timestamps)

    positive_time_diffs = time_diffs[time_diffs > 0]

    # Speed calculation
    speeds = np.array([])

    if len(time_diffs) > 0:
        distances = gps["step_distance_m"].to_numpy()[1:]

        valid = time_diffs > 0

        speeds = distances[valid] / time_diffs[valid]

    # Route integrity
    valid_arc_ids = [
        arc_id for arc_id in route
        if 0 <= arc_id < len(arcs)
    ]

    route_edges = []

    invalid_node_refs = 0

    for arc_id in valid_arc_ids:

        start = int(arcs.iloc[arc_id, 0])
        end = int(arcs.iloc[arc_id, 1])

        if (
            0 <= start < len(nodes)
            and 0 <= end < len(nodes)
        ):
            route_edges.append((start, end))
        else:
            invalid_node_refs += 1

    disconnected = sum(
        route_edges[i][1] != route_edges[i + 1][0]
        for i in range(len(route_edges) - 1)
    )

    # GPS trajectory duration
    duration = (
        timestamps[-1] - timestamps[0]
        if len(timestamps) > 1 else np.nan
    )

    # Geographic displacement
    displacement = np.nan

    if len(gps) > 1:
        displacement = haversine(
            gps["lon"].iloc[0],
            gps["lat"].iloc[0],
            gps["lon"].iloc[-1],
            gps["lat"].iloc[-1]
        )

    # Summary
    result = {
        "trip_id": trip_id,
        "gps_observations": len(gps),
        "network_nodes": len(nodes),
        "network_arcs": len(arcs),
        "route_arcs": len(route),
        "valid_route_arcs": len(valid_arc_ids),
        "invalid_route_arc_ids": len(route) - len(valid_arc_ids),
        "invalid_node_references": invalid_node_refs,
        "disconnected_transitions": disconnected,
        "duration_timestamp_units": duration,
        "mean_sampling_interval": (
            np.mean(positive_time_diffs)
            if len(positive_time_diffs) else np.nan
        ),
        "median_sampling_interval": (
            np.median(positive_time_diffs)
            if len(positive_time_diffs) else np.nan
        ),
        "mean_gps_step_distance_m": (
            gps["step_distance_m"].mean()
        ),
        "median_gps_step_distance_m": (
            gps["step_distance_m"].median()
        ),
        "max_gps_step_distance_m": (
            gps["step_distance_m"].max()
        ),
        "mean_speed_m_per_timestamp_unit": (
            np.mean(speeds) if len(speeds) else np.nan
        ),
        "max_speed_m_per_timestamp_unit": (
            np.max(speeds) if len(speeds) else np.nan
        ),
        "gps_displacement_m": displacement,
        "gps_lon_min": gps["lon"].min(),
        "gps_lon_max": gps["lon"].max(),
        "gps_lat_min": gps["lat"].min(),
        "gps_lat_max": gps["lat"].max()
    }

    return result, gps


# =====================================================
# VISUALIZATION
# =====================================================

def create_histogram(data, title, xlabel, filename):

    values = pd.to_numeric(
        pd.Series(data),
        errors="coerce"
    ).dropna()

    if values.empty:
        print(f"Skipping empty plot: {title}")
        return

    plt.figure(figsize=(9, 5))

    plt.hist(values, bins=30, edgecolor="black")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of trips")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()


def create_boxplot(data, title, ylabel, filename):

    values = pd.to_numeric(
        pd.Series(data),
        errors="coerce"
    ).dropna()

    if values.empty:
        return

    plt.figure(figsize=(8, 5))

    plt.boxplot(values)

    plt.title(title)
    plt.ylabel(ylabel)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()


def create_correlation_plot(df):

    columns = [
        "gps_observations",
        "network_nodes",
        "network_arcs",
        "route_arcs",
        "mean_gps_step_distance_m",
        "gps_displacement_m"
    ]

    correlation = df[columns].corr()

    plt.figure(figsize=(9, 7))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(range(len(columns)), columns)

    plt.title("Trip-Level Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_correlation.png",
        dpi=200
    )

    plt.close()


# =====================================================
# MAIN EDA PIPELINE
# =====================================================

def main():

    print("\n====================================")
    print("PATHSNAP - EXPLORATORY DATA ANALYSIS")
    print("====================================")

    trip_folders = sorted(
        [
            folder for folder in DATA_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    results = []
    gps_data = []

    for i, folder in enumerate(trip_folders, 1):

        try:
            result, gps = analyse_trip(folder)

            results.append(result)
            gps_data.append(gps)

            print(
                f"Processed {i}/{len(trip_folders)}: "
                f"{folder.name}"
            )

        except Exception as error:

            print(
                f"Failed: {folder.name} | {error}"
            )

    if not results:
        print("No trips were successfully analysed.")
        return

    df = pd.DataFrame(results)

    # Save consolidated statistics
    output_csv = REPORT_DIR / "eda_trip_summary.csv"

    df.to_csv(output_csv, index=False)

    # Overall summary
    print("\n========== DATASET SUMMARY ==========")

    print("Trips analysed:", len(df))
    print("Total GPS observations:", df["gps_observations"].sum())
    print("Total network nodes:", df["network_nodes"].sum())
    print("Total network arcs:", df["network_arcs"].sum())
    print("Total route arcs:", df["route_arcs"].sum())

    print("\n========== STATISTICAL SUMMARY ==========")

    print(
        df[
            [
                "gps_observations",
                "network_nodes",
                "network_arcs",
                "route_arcs",
                "mean_gps_step_distance_m",
                "gps_displacement_m"
            ]
        ].describe().round(3)
    )

    print("\n========== ROUTE INTEGRITY ==========")

    print(
        "Trips with disconnected transitions:",
        (df["disconnected_transitions"] > 0).sum()
    )

    print(
        "Total disconnected transitions:",
        df["disconnected_transitions"].sum()
    )

    # Generate figures
    create_histogram(
        df["gps_observations"],
        "GPS Observations per Trip",
        "Number of GPS observations",
        "gps_observations_distribution.png"
    )

    create_histogram(
        df["network_nodes"],
        "Network Nodes per Trip",
        "Number of nodes",
        "network_nodes_distribution.png"
    )

    create_histogram(
        df["network_arcs"],
        "Network Arcs per Trip",
        "Number of arcs",
        "network_arcs_distribution.png"
    )

    create_histogram(
        df["route_arcs"],
        "Reference Route Length Distribution",
        "Number of route arcs",
        "route_length_distribution.png"
    )

    create_histogram(
        df["mean_gps_step_distance_m"],
        "Mean GPS Step Distance per Trip",
        "Mean distance (metres)",
        "gps_step_distance_distribution.png"
    )

    create_histogram(
        df["mean_sampling_interval"],
        "Mean GPS Sampling Interval",
        "Timestamp units",
        "sampling_interval_distribution.png"
    )

    create_boxplot(
        df["mean_gps_step_distance_m"],
        "GPS Step Distance Boxplot",
        "Distance (metres)",
        "gps_step_distance_boxplot.png"
    )

    create_correlation_plot(df)

    print("\nEDA completed successfully!")

    print("Summary CSV:", output_csv)
    print("Figures saved in:", FIGURE_DIR)


if __name__ == "__main__":
    main()
