"""Shared classification metrics.

ROC-AUC is always computed from predicted probabilities. (The released
notebooks computed it from hard labels, which understates AUC; no paper
table depends on the label-based values.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def metrics(y, prob, pred) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "N": len(y),
        "WetFrac": float(y.mean()),
        "Accuracy": accuracy_score(y, pred),
        "ROC-AUC": (roc_auc_score(y, prob)
                    if len(np.unique(y)) > 1 else float("nan")),
        "WetF1": f1_score(y, pred, pos_label=1, zero_division=0),
        "DryF1": f1_score(y, pred, pos_label=0, zero_division=0),
        "DryRecall": (float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1))
                      if (y == 0).any() else float("nan")),
    }


def per_class(y, pred) -> dict:
    """Precision / recall / F1 for both classes (dry=0, wet=1)."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {}
    for cls, name in [(0, "dry"), (1, "wet")]:
        tp = int(((pred == cls) & (y == cls)).sum())
        prec = tp / max(int((pred == cls).sum()), 1)
        rec = tp / max(int((y == cls).sum()), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        out[name] = {"precision": prec, "recall": rec, "f1": f1,
                     "support": int((y == cls).sum())}
    return out


def summary_table(results: dict) -> pd.DataFrame:
    """{name: metrics-dict} -> one-row-per-name DataFrame."""
    return pd.DataFrame([{"Split": k, **v} for k, v in results.items()])
