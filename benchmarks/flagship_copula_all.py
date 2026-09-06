"""Gaussian-copula annual dry-day estimation on the flagship splits.

CANONICAL (q65): Platt-calibrated probabilities + rho ceiling, scored on ALL
HOBO reaches with >= MIN_VAL q65 validation labels (22 reaches). The Platt
calibrator is fit on the q65 model's TRAINING-period Day-3 predictions
(HOBO-labeled stride-3 rows, day-3 date <= 2020-09-10). Raw-method coverage
on the same reaches is reported alongside as the diagnostic comparison.

DIAGNOSTIC (ph / q80 / site): the raw uncalibrated method on the original
site selections, retained to show the probability-calibration failure mode
is shared by every independently trained flagship model.

Predictions: each split's own seed-42 stride-1 Day-3 export. Simulation
seeds are per-site (hja.copula), so numbers are order-independent and match
rgcn/rgcn_eval.ipynb exactly.

Run:  uv run python benchmarks/flagship_copula_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from flagship_analysis import (  # noqa: E402  (imports build STATUS too)
    CUTOFFS, GUARD_DAYS, HOLDOUT, SEASON, in_blocks, stride1_dir)
from hja.copula import (RHO_CEIL, fit_platt, load_hobo_daily,  # noqa: E402
                        site_row)

OUT = REPO / "results/flagship/copula_dryday_allsplits.md"
MIN_VAL = 20  # q65: minimum val labels for a reach to be scored

HOBO = load_hobo_daily()
HOBO_IDS = set(HOBO["NHDPlusID"])
HOBO_KEYS = set(map(tuple, HOBO[["NHDPlusID", "Date"]]
                    .itertuples(index=False, name=None)))


def train_dates_fn(split: str):
    if split in ("ph", "site"):
        return lambda d: ~in_blocks(d, GUARD_DAYS)
    return lambda d: d <= pd.Timestamp(CUTOFFS[split])


def val_predictions(split: str) -> pd.DataFrame:
    df = pd.read_csv(stride1_dir(split, 42) / "train_val_predictions_day3.csv",
                     usecols=["date", "site_id", "true_wetdry",
                              "pred_wetdry_prob", "has_true_label"],
                     parse_dates=["date"])
    df = df[df["has_true_label"]].copy()
    df["site_id"] = df["site_id"].astype("int64")
    if split == "ph":
        df = df[in_blocks(df["date"]) & df["site_id"].isin(HOBO_IDS)]
    elif split in CUTOFFS:
        df = df[(df["date"] > pd.Timestamp(CUTOFFS[split]))
                & df["site_id"].isin(HOBO_IDS)]
    else:  # site
        df = df[df["site_id"].isin(HOLDOUT)
                & (df["date"] >= pd.Timestamp(SEASON[0]))
                & (df["date"] <= pd.Timestamp(SEASON[1]))]
    return df


def q65_calibrator():
    """Platt fit on the q65 model's training-period Day-3 predictions
    (HOBO-labeled rows only, stride-3 export)."""
    tr = pd.read_csv(REPO / "data/retrain/flagship/predictions_flag_q65/"
                     "train_val_predictions_day3.csv",
                     usecols=["date", "site_id", "true_wetdry",
                              "pred_wetdry_prob", "has_true_label"],
                     parse_dates=["date"])
    tr = tr[tr["has_true_label"]
            & (tr["date"] <= pd.Timestamp(CUTOFFS["q65"]))].copy()
    tr["site_id"] = tr["site_id"].astype("int64")
    tr = tr[[(r.site_id, r.date) in HOBO_KEYS for r in tr.itertuples()]]
    print(f"q65 Platt calibration pool: {len(tr)} HOBO-labeled train rows")
    return fit_platt(tr["pred_wetdry_prob"], tr["true_wetdry"].round())


def rows_for(split, sites, df, calibrate, rho_ceil):
    tdf = train_dates_fn(split)
    rows = []
    for site in sites:
        sv = df[df["site_id"] == site]
        rows.append(site_row(site, sv["pred_wetdry_prob"], sv["true_wetdry"],
                             HOBO, tdf, calibrate, rho_ceil))
    rows.sort(key=lambda r: -r["true"])
    return rows


def table(rows, show_used):
    head = ("| Site | N val | ρ (lag-1) | ρ used | ρ pairs | True dry d/yr | "
            "Mean pred | 95% CI | In CI |" if show_used else
            "| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | "
            "Mean pred | 95% CI | In CI |")
    sep = "|---|--:|--:|--:|--:|--:|" + ("--:|" if show_used else "") + "---|:--:|"
    lines = [head, sep]
    for r in rows:
        used = f" {r['rho_used']:.3f} |" if show_used else ""
        lines.append(f"| {r['site']} | {r['n']} | {r['rho']:.3f} |{used} "
                     f"{r['np']} | {r['true']:.1f} | {r['mean']:.1f} | "
                     f"[{r['lo']:.1f}, {r['hi']:.1f}] | "
                     f"{'yes' if r['ok'] else 'NO'} |")
    return lines


def coverage(rows):
    k = sum(r["ok"] for r in rows)
    return k, len(rows), f"{k}/{len(rows)} ({k / len(rows):.0%})"


def main() -> int:
    lines = ["# Annual dry-day estimation — flagship RGCN, Gaussian copula", "",
             "**Canonical (q65)**: Platt-calibrated Day-3 dry probabilities "
             f"(calibrator fit on training-period predictions) with a ρ ceiling "
             f"of {RHO_CEIL}, scored on all HOBO reaches with ≥{MIN_VAL} q65 "
             "validation labels. Raw-method coverage on the same reaches is "
             "reported for comparison. ph/q80/site sections use the raw "
             "uncalibrated method (diagnostic; original site selections). "
             "ρ = lag-1 autocorrelation from consecutive-day pairs of the raw "
             "daily HOBO series over each split's training dates; each split "
             "scored with its own seed-42 stride-1 Day-3 export; 10,000 "
             "sims/site with per-site seeds.", ""]

    # ---- canonical q65
    df = val_predictions("q65")
    counts = df.groupby("site_id").size().sort_values(ascending=False)
    sites = counts[counts >= MIN_VAL].index.tolist()
    cal = q65_calibrator()
    cal_rows = rows_for("q65", sites, df, cal, RHO_CEIL)
    raw_rows = rows_for("q65", sites, df, None, None)
    kc, n, cov_c = coverage(cal_rows)
    kr, _, cov_r = coverage(raw_rows)
    import numpy as np
    w_c = np.mean([r["hi"] - r["lo"] for r in cal_rows])
    w_r = np.mean([r["hi"] - r["lo"] for r in raw_rows])
    lines += [f"## q65 — CANONICAL (Platt + ρ-clip; all {n} HOBO val reaches)", ""]
    lines += table(cal_rows, show_used=True)
    lines += ["", f"Coverage: **{cov_c}**; mean 95% CI width {w_c:.1f} days.",
              f"Raw method on the same {n} reaches: {cov_r} coverage, mean CI "
              f"width {w_r:.1f} days — the improvement is almost entirely the "
              "probability calibration (mean perennial-reach p_dry drops from "
              "a few percent to <1%); the ρ ceiling keeps intervals at "
              "ρ≈1.0 reaches informative.", ""]

    # ---- diagnostic raw splits
    for split in ("ph", "q80", "site"):
        df = val_predictions(split)
        if split == "site":
            sites = [s for s in HOLDOUT if (df["site_id"] == s).any()]
            pick = f"all {len(sites)} holdout reaches"
        else:
            counts = df.groupby("site_id").size().sort_values(ascending=False)
            sites = counts.head(8).index.tolist()
            tied = int((counts == counts.iloc[0]).sum())
            pick = ("top-8 by val label count"
                    + (f" (NOTE: {tied} sites tied at {counts.iloc[0]} labels "
                       "— selection arbitrary among ties)" if tied > 8 else ""))
        rows = rows_for(split, sites, df, None, None)
        _, _, cov = coverage(rows)
        lines += [f"## {split} — raw method, diagnostic  ({pick})", ""]
        lines += table(rows, show_used=False)
        lines += ["", f"Coverage: {cov}.", ""]

    lines += ["Caveats: q65 val is late-season only (dry-biased 'typical "
              "year'); the q65 calibration pool is 579 HOBO train rows from a "
              "single season (a leave-site-out calibration check is the "
              "natural robustness follow-up); 'site' scores the flag_sh model "
              "at reaches whose labels were masked from its loss.", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
