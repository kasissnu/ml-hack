"""Phase 7 - Folium maps of the final predictions on student_test trips.

For the best, median and worst student_test trip (by classifier+HMM Top-1) the map shows, as
separate toggleable layers with a legend:
  road network (grey), reference route (green), classifier-only arcs (orange),
  classifier+HMM stitched route (red), GPS observations (blue),
  GPS points whose HMM arc differs from the reference arc (purple).
Requires scripts/05_hmm_eval.py (cache/hmm_predictions_student_test.pkl).

Outputs: results/maps/result_<trip>.html, results/maps/index.csv
         results/maps/png/result_<trip>.png (headless-browser screenshot if Edge/Chrome is found)
"""
import _bootstrap  # noqa: F401

import os
import pickle
import shutil
import subprocess
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from pathsnap import viz
from pathsnap.config import partition_dir, path
from pathsnap.io import load_route, load_trip

OUT = path("results") / "maps"; (OUT / "png").mkdir(parents=True, exist_ok=True)
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shutil.which("chromium") or "", shutil.which("google-chrome") or ""]


def make_map(tid, preds, labels, per_trip):
    d = partition_dir("student_test") / tid
    trip = load_trip(d); route = load_route(d)
    hmm = preds["hmm_tuned"]; clf = preds["classifier_only"]
    m = viz.base_map(trip.track)
    viz.add_network(m, trip.nodes, trip.arcs, trip.track, show=True)
    viz.add_route(m, trip.nodes, trip.arcs, route, "Reference route", viz.COLORS["reference"], weight=9, opacity=0.5)
    clf_arcs = np.unique(clf["point_arcs"][clf["point_arcs"] >= 0])
    viz.add_route(m, trip.nodes, trip.arcs, clf_arcs, "Classifier-only arcs", viz.COLORS["classifier"],
                  weight=4, opacity=0.9, show=False)
    viz.add_route(m, trip.nodes, trip.arcs, hmm["route"], "Predicted route (classifier + HMM)", viz.COLORS["hmm"],
                  weight=4, opacity=0.9)
    viz.add_gps(m, trip.track, every=max(1, len(trip.track) // 3000))
    wrong = np.flatnonzero((labels >= 0) & (hmm["point_arcs"] != labels))
    fg = folium.FeatureGroup(name=f"GPS points with wrong HMM arc ({len(wrong)})")
    for i in wrong:
        folium.CircleMarker([trip.track[i, 1], trip.track[i, 0]], radius=4, color=viz.COLORS["wrong"], fill=True,
                            tooltip=f"#{i}: predicted arc {hmm['point_arcs'][i]}, reference {labels[i]}").add_to(fg)
    fg.add_to(m)
    r = per_trip.loc[tid]
    viz.add_legend(m, [("Road network", viz.COLORS["network"]), ("Reference route", viz.COLORS["reference"]),
                       ("Classifier-only arcs", viz.COLORS["classifier"]),
                       ("Predicted route (HMM)", viz.COLORS["hmm"]), ("GPS observations", viz.COLORS["gps"]),
                       ("Wrong HMM point", viz.COLORS["wrong"])],
                   title=f"student_test {tid}: Top-1 {100 * r.top1:.1f}% | route F1 {100 * r.route_f1:.1f}%")
    out = OUT / f"result_{tid}.html"
    viz.finish(m).save(str(out))
    return out, len(wrong)


def screenshot(html: Path, png: Path):
    exe = next((b for b in BROWSERS if b and os.path.exists(b)), None)
    if exe is None:
        return False
    cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1400,950",
           "--virtual-time-budget=15000", f"--screenshot={png}", html.resolve().as_uri()]
    try:
        subprocess.run(cmd, timeout=90, capture_output=True)
    except Exception:  # noqa: BLE001
        return False
    return png.exists()


def main():
    preds = pickle.load(open(path("cache") / "hmm_predictions_student_test.pkl", "rb"))
    labels = pickle.load(open(path("cache") / "labels_student_test.pkl", "rb"))
    per = pd.read_csv(path("results") / "hmm" / "per_trip_student_test.csv", dtype={"trip_id": str})
    per = per[per.method == "hmm_tuned"].set_index("trip_id").sort_values("top1")
    picks = {"worst": per.index[0], "median": per.index[len(per) // 2], "best": per.index[-1]}
    rows = []
    for tag, tid in picks.items():
        html, n_wrong = make_map(tid, preds[tid], labels[tid], per)
        png = OUT / "png" / f"result_{tid}.png"
        ok = screenshot(html, png)
        rows.append(dict(which=tag, trip_id=tid, top1=per.loc[tid, "top1"], route_f1=per.loc[tid, "route_f1"],
                         wrong_points=n_wrong, html=html.name, png=png.name if ok else ""))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)


if __name__ == "__main__":
    main()
