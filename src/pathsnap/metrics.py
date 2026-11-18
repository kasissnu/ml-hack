"""Evaluation metrics.

Candidate level (binary: is this candidate the reference arc?)
  accuracy, precision, recall, macro-F1 (threshold 0.5), log loss, confusion matrix.

Observation level
  Top-1 accuracy : for each matched GPS point, the highest-scoring candidate is compared with the
                   derived reference arc. Points whose reference arc is not among the candidates,
                   or that have no candidates at all, count as errors (denominator = matched points).
  strict accuracy: same, but unmatched points (label -1) are also counted, always as errors.

Route level
  continuity     : share of transitions between consecutive *different* predicted arcs that are
                   physically possible (to_node of the first == from_node of the second).
  route P/R/F1   : set overlap between predicted arcs and reference-route arcs, plus a
                   length-weighted version.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def candidate_metrics(y: np.ndarray, p: np.ndarray | None, yhat: np.ndarray | None = None) -> dict:
    if yhat is None:
        yhat = (p >= 0.5).astype(int)
    out = dict(accuracy=accuracy_score(y, yhat),
               precision=precision_score(y, yhat, zero_division=0),
               recall=recall_score(y, yhat, zero_division=0),
               macro_f1=f1_score(y, yhat, average="macro"),
               precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
               recall_macro=recall_score(y, yhat, average="macro", zero_division=0))
    out["log_loss"] = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]) if p is not None else np.nan
    out["confusion"] = confusion_matrix(y, yhat, labels=[0, 1]).tolist()
    return out


def obs_key(trip_codes: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return trip_codes.astype(np.int64) * 10_000_000 + obs.astype(np.int64)


def argmax_per_group(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Row index of the highest score within each group of `key`."""
    order = np.lexsort((-score, key))
    k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return order[first]


def top1_accuracy(chosen_y: np.ndarray, n_matched_points: int) -> float:
    """chosen_y: y of the selected candidate for each matched point that has candidates."""
    return float(chosen_y.sum()) / max(n_matched_points, 1)


def route_continuity(arc_seq: np.ndarray, arcs: np.ndarray) -> tuple[int, int]:
    """(valid transitions, total transitions) between consecutive different arcs."""
    a = np.asarray(arc_seq)
    a = a[a >= 0]
    if len(a) < 2:
        return 0, 0
    change = a[1:] != a[:-1]
    x, y = a[:-1][change], a[1:][change]
    return int((arcs[x, 1] == arcs[y, 0]).sum()), int(change.sum())


def route_prf(pred_arcs, ref_arcs, length: np.ndarray) -> dict:
    p = set(int(a) for a in pred_arcs if a >= 0)
    r = set(int(a) for a in ref_arcs)
    tp = p & r
    prec = len(tp) / max(len(p), 1); rec = len(tp) / max(len(r), 1)
    lw = lambda s: float(length[list(s)].sum()) if s else 0.0  # noqa: E731
    lp, lr, lt = lw(p), lw(r), lw(tp)
    lprec = lt / lp if lp else 0.0; lrec = lt / lr if lr else 0.0
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0  # noqa: E731
    return dict(route_precision=prec, route_recall=rec, route_f1=f(prec, rec),
                route_len_precision=lprec, route_len_recall=lrec, route_len_f1=f(lprec, lrec))
