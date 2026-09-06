"""Shared train/eval engine for the tabular baselines (LR, XGBoost).

Replicates the released notebook protocol exactly (lr.ipynb cell 27 /
xgb.ipynb cell 26): split -> drop id/order columns -> (LR only) scale with a
train-fit StandardScaler -> ADASYN on training rows -> fit -> predict.
Accuracy / F1 / per-class metrics reproduce the released numbers; ROC-AUC is
computed from probabilities (the notebooks used hard labels there).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

from hja.evaluation import metrics, per_class, summary_table
from hja.paths import RESULTS
from hja.splits import RELEASED_SPLITS

# Released notebook drop list ("stream order columns excluded from training —
# reserved for post-hoc evaluation only").
RELEASED_DROP = ["NHDPlusID", "SiteIDCode", "Date",
                 "StreamOrde", "FCode", "n_discharge", "n_water_presence",
                 "has_data"]

BASELINES_DIR = RESULTS / "baselines"


def train_eval(frame: pd.DataFrame, make_model, split: str = "temporal",
               seed: int = 42, scale: bool = False) -> dict:
    """One split, one model. `make_model(seed)` returns an unfit estimator."""
    X_tr, X_te, y_tr, y_te = RELEASED_SPLITS[split](frame)

    # Post-hoc breakdown columns, saved before dropping.
    orders_te = X_te["StreamOrde"].to_numpy() if "StreamOrde" in X_te else None
    sites_te = X_te["SiteIDCode"].to_numpy()
    dates_te = X_te["Date"].to_numpy()

    X_tr = X_tr.drop(RELEASED_DROP, axis=1, errors="ignore")
    X_te = X_te.drop(RELEASED_DROP, axis=1, errors="ignore")
    feats = X_tr.columns.tolist()

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    X_res, y_res = ADASYN(random_state=seed).fit_resample(X_tr, y_tr)
    model = make_model(seed)
    model.fit(X_res, y_res)

    prob = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    y_true = np.asarray(y_te).astype(int)
    return {
        "split": split, "model": model, "scaler": scaler, "features": feats,
        "y_true": y_true, "prob": prob, "pred": np.asarray(pred).astype(int),
        "orders": orders_te, "sites": sites_te, "dates": dates_te,
        "metrics": metrics(y_true, prob, pred),
        "per_class": per_class(y_true, pred),
    }


def run_all_splits(frame: pd.DataFrame, make_model, seed: int = 42,
                   scale: bool = False) -> dict:
    return {s: train_eval(frame, make_model, s, seed, scale)
            for s in RELEASED_SPLITS}


def write_report(name: str, results: dict, notes: str) -> Path:
    """Metrics JSON + summary markdown under results/baselines/."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {s: {"metrics": r["metrics"], "per_class": r["per_class"]}
               for s, r in results.items()}
    (BASELINES_DIR / f"{name}_released_splits.json").write_text(
        json.dumps(payload, indent=2))

    table = summary_table({s: r["metrics"] for s, r in results.items()})
    cols = table.columns.tolist()
    md_rows = ["| " + " | ".join(cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in table.iterrows():
        md_rows.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    lines = [f"# {name} — released split protocol", "", notes, "",
             *md_rows, "",
             "Per-class (dry / wet):", ""]
    for s, r in results.items():
        pc = r["per_class"]
        lines.append(
            f"- **{s}**: dry P/R/F1 "
            f"{pc['dry']['precision']:.3f}/{pc['dry']['recall']:.3f}/{pc['dry']['f1']:.3f}"
            f" (n={pc['dry']['support']}), wet P/R/F1 "
            f"{pc['wet']['precision']:.3f}/{pc['wet']['recall']:.3f}/{pc['wet']['f1']:.3f}"
            f" (n={pc['wet']['support']})")
    out = BASELINES_DIR / f"{name}_released_splits.md"
    out.write_text("\n".join(lines) + "\n")
    return out
