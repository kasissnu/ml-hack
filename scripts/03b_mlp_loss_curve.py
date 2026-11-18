"""Phase 3b - Epoch-wise training AND validation loss of the MLP on a trip-grouped split.

sklearn's MLPClassifier(early_stopping=True) only records validation *accuracy* on a random 10 % of
rows (rows of the same trip leak between the two). For a proper loss plot we train the same
architecture epoch by epoch with partial_fit:
  * training trips  : CV folds 1-4 (64 trips), 80 000 GPS observations sub-sampled as in Phase 3
  * validation trips: CV fold 0 (16 trips, never seen in training)
and record, after every epoch, log loss on the training rows and on the validation trips, plus the
validation observation-level Top-1. Training stops after 6 epochs without validation-loss
improvement (max 60 epochs).

Outputs: results/models/mlp/loss_curve_grouped.csv, results/models/mlp/loss_curve_grouped.png
"""
import _bootstrap  # noqa: F401

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.config import load_config, path
from pathsnap.features import FEATURE_COLS
from pathsnap.metrics import argmax_per_group, obs_key

CFG = load_config()
SEED = CFG["seed"]
OUT = path("results") / "models" / "mlp"


def main():
    df = pd.read_pickle(path("cache") / "features_train.pkl")
    df["trip_id"] = df.trip_id.astype(str)
    folds = pd.read_csv(path("results") / "models" / "folds.csv", dtype={"trip_id": str})
    val_trips = set(folds[folds.fold == 0].trip_id)
    m = (df.label_arc >= 0).to_numpy()
    is_val = df.trip_id.isin(val_trips).to_numpy()
    tcode = pd.Categorical(df.trip_id).codes
    key = obs_key(tcode, df.obs.to_numpy())

    rng = np.random.default_rng(SEED)
    fit_keys = np.unique(key[~is_val & m])
    keep = np.isin(key, rng.choice(fit_keys, min(80_000, len(fit_keys)), replace=False))
    fit = ~is_val & m & keep
    val = is_val & m
    scaler = StandardScaler().fit(df.loc[fit, FEATURE_COLS].to_numpy())
    Xf, yf = scaler.transform(df.loc[fit, FEATURE_COLS].to_numpy()), df.y.to_numpy()[fit]
    Xv, yv = scaler.transform(df.loc[val, FEATURE_COLS].to_numpy()), df.y.to_numpy()[val]
    kv = key[val]
    n_val_obs = len(np.unique(kv))

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                        random_state=SEED)
    rows, best, wait = [], np.inf, 0
    t0 = time.time()
    order = np.arange(len(yf))
    for epoch in range(1, 61):
        rng.shuffle(order)
        mlp.partial_fit(Xf[order], yf[order], classes=[0, 1])
        pf = mlp.predict_proba(Xf)[:, 1]; pv = mlp.predict_proba(Xv)[:, 1]
        sel = argmax_per_group(kv, pv)
        rows.append(dict(epoch=epoch, train_log_loss=log_loss(yf, np.clip(pf, 1e-7, 1 - 1e-7)),
                         val_log_loss=log_loss(yv, np.clip(pv, 1e-7, 1 - 1e-7)),
                         val_top1=yv[sel].sum() / n_val_obs, seconds=time.time() - t0))
        print(rows[-1], flush=True)
        if rows[-1]["val_log_loss"] < best - 1e-4:
            best, wait = rows[-1]["val_log_loss"], 0
        else:
            wait += 1
            if wait >= 6:
                break
    h = pd.DataFrame(rows)
    h.to_csv(OUT / "loss_curve_grouped.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(h.epoch, h.train_log_loss, "o-", ms=3, label="training loss (64 train trips)")
    ax.plot(h.epoch, h.val_log_loss, "o-", ms=3, label="validation loss (16 held-out trips)")
    b = h.val_log_loss.idxmin()
    ax.axvline(h.epoch[b], color="grey", ls="--", lw=1, label=f"best validation epoch ({h.epoch[b]})")
    ax.set(xlabel="epoch", ylabel="log loss", title="MLP (128-64) training vs validation loss, trip-grouped split")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "loss_curve_grouped.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
