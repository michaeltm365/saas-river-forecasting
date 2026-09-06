"""XGBoost baseline (released protocol).

Run all three released splits and write results/baselines/xgb_*.md/json:

    uv run python -m hja.models.xgb
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from hja.data import build_hobo_frame
from hja.evaluation import summary_table
from hja.importance import importance_frame
from hja.models.tabular import run_all_splits, train_eval, write_report


def make_model(seed: int = 42):
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         random_state=seed)


def run(frame: pd.DataFrame | None = None, split: str = "temporal",
        seed: int = 42) -> dict:
    """Train/evaluate one split; result includes a gain importance table."""
    frame = build_hobo_frame() if frame is None else frame
    res = train_eval(frame, make_model, split, seed, scale=False)
    res["importance"] = importance_frame(res["features"],
                                         res["model"].feature_importances_)
    return res


def run_all(frame: pd.DataFrame | None = None, seed: int = 42) -> dict:
    frame = build_hobo_frame() if frame is None else frame
    results = run_all_splits(frame, make_model, seed, scale=False)
    for r in results.values():
        r["importance"] = importance_frame(r["features"],
                                           r["model"].feature_importances_)
    return results


def predict_site_date(model, central_df, feature_cols, site_id, date,
                      days_ahead: int = 3) -> str:
    """Predict wet/dry status days_ahead from `date` at `site_id` (demo)."""
    date = pd.to_datetime(date)
    row = central_df[(central_df["SiteIDCode"] == str(site_id))
                     & (central_df["Date"] == date)]
    if row.empty:
        return f"No data found for Site {site_id} on {date.date()}"
    Xq = row[feature_cols]
    pred = model.predict(Xq)
    prob = model.predict_proba(Xq)[:, 1]
    return (f"Site {site_id} on {(date + pd.Timedelta(days=days_ahead)).date()} "
            f"(predicted from {date.date()}): {'DRY' if pred == 0 else 'WET'}, "
            f"(P(wet)={prob[0]:.4f})")


def main() -> int:
    results = run_all()
    print(summary_table({s: r["metrics"] for s, r in results.items()})
          .to_string(index=False))
    out = write_report(
        "xgb", results,
        "XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1), "
        "ADASYN on training rows, no scaling (released protocol; seed 42). "
        "ROC-AUC from probabilities.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
