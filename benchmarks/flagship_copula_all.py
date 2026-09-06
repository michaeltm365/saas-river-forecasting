"""Gaussian-copula annual dry-day estimation on ALL flagship splits.

Extends the ph-only run in flagship_analysis.part_copula:
  ph   — val = the three 12-day blocks; top-8 HOBO reaches (all tied at 36
         val labels — selection is arbitrary among ties, noted in output)
  q65  — val = after 2020-09-10 (49 HOBO val days/site)
  q80  — val = after 2020-09-28 (31 HOBO val days/site)
  site — the 5 with-sensor holdout reaches (flag_sh predictions), full
         labeled season; the annual-dry-day product at spatially unseen sites

Fix vs the earlier version: rho is estimated from the RAW daily HOBO label
series on the split's TRAINING dates (consecutive calendar-day pairs only),
so it no longer depends on export coverage — the q65/q80 stride-1 exports
only span val dates. Predictions: seed-42 stride-1 day-3 exports.

Run:  uv run python benchmarks/flagship_copula_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from flagship_analysis import (  # noqa: E402  (imports build STATUS too)
    CUTOFFS, GUARD_DAYS, HOLDOUT, SEASON, in_blocks, stride1_dir)

OUT = REPO / "results/flagship/copula_dryday_allsplits.md"
N_SIMS, HORIZON_DAYS = 10_000, 365


def hobo_daily() -> pd.DataFrame:
    obs = pd.read_csv(REPO / "data/sciencebase/obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["NHDPlusID"] = obs["NHDPlusID"].astype("int64")
    g = (obs.groupby(["NHDPlusID", "Date"])["HoboWetDry0.05"].mean()
            .round().dropna().rename("wet").reset_index())
    return g


HOBO = hobo_daily()
HOBO_IDS = set(HOBO["NHDPlusID"])


def train_dates_mask(split: str, dates: pd.Series) -> pd.Series:
    if split in ("ph", "site"):
        return ~in_blocks(dates, GUARD_DAYS)
    return dates <= pd.Timestamp(CUTOFFS[split])


def rho_lag1(split: str, site: int) -> tuple[float, int]:
    s = HOBO[HOBO["NHDPlusID"] == site].sort_values("Date")
    s = s[train_dates_mask(split, s["Date"])]
    v = s["wet"].to_numpy()
    consec = s["Date"].diff().dt.days.to_numpy()[1:] == 1
    a, b = v[:-1][consec], v[1:][consec]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0, int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))


def val_predictions(split: str) -> pd.DataFrame:
    df = pd.read_csv(stride1_dir(split, 42) / "train_val_predictions_day3.csv",
                     usecols=["date", "site_id", "true_wetdry",
                              "pred_wetdry_prob", "has_true_label"],
                     parse_dates=["date"])
    df = df[df["has_true_label"]].copy()
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


def run_split(split: str, rng) -> list[str]:
    df = val_predictions(split)
    if split == "site":
        sites = [s for s in HOLDOUT if (df["site_id"] == s).any()]
        pick = f"all {len(sites)} holdout reaches"
    else:
        counts = df.groupby("site_id").size().sort_values(ascending=False)
        sites = counts.head(8).index.tolist()
        tied = int((counts == counts.iloc[0]).sum())
        pick = ("top-8 by val label count"
                + (f" (NOTE: {tied} sites tied at {counts.iloc[0]} labels — "
                   f"selection arbitrary among ties)" if tied > 8 else ""))
    rows = []
    for site in sites:
        sv = df[df["site_id"] == site]
        rho, n_pairs = rho_lag1(split, site)
        p_dry = 1.0 - sv["pred_wetdry_prob"].to_numpy()
        innov = rng.standard_normal((N_SIMS, HORIZON_DAYS)) * np.sqrt(
            max(1 - rho ** 2, 0.0))
        z = np.empty((N_SIMS, HORIZON_DAYS))
        z[:, 0] = rng.standard_normal(N_SIMS)
        for t in range(1, HORIZON_DAYS):
            z[:, t] = rho * z[:, t - 1] + innov[:, t]
        u = norm.cdf(z)
        sampled = rng.choice(p_dry, size=(N_SIMS, HORIZON_DAYS), replace=True)
        counts_sim = (u < sampled).sum(axis=1)
        lo, hi = np.percentile(counts_sim, [2.5, 97.5])
        true_dry = (1.0 - sv["true_wetdry"].round().mean()) * HORIZON_DAYS
        rows.append(dict(site=site, n=len(sv), rho=rho, np=n_pairs,
                         true=true_dry, mean=counts_sim.mean(), lo=lo, hi=hi,
                         ok=bool(lo <= true_dry <= hi)))
    rows.sort(key=lambda r: -r["true"])
    cov = sum(r["ok"] for r in rows)
    lines = [f"## {split}  ({pick})", "",
             "| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | "
             "Mean pred | 95% CI | In CI |",
             "|---|--:|--:|--:|--:|--:|---|:--:|"]
    for r in rows:
        lines.append(f"| {r['site']} | {r['n']} | {r['rho']:.3f} | {r['np']} | "
                     f"{r['true']:.1f} | {r['mean']:.1f} | "
                     f"[{r['lo']:.1f}, {r['hi']:.1f}] | "
                     f"{'yes' if r['ok'] else 'NO'} |")
    lines += ["", f"Coverage: {cov}/{len(rows)} ({cov/len(rows):.0%}).", ""]
    return lines


def main() -> int:
    rng = np.random.default_rng(42)
    lines = ["# Annual dry-day estimation — flagship RGCN, Gaussian copula, all splits", "",
             "Seed-42 stride-1 day-3 predictions per split; ρ = genuine lag-1 "
             "autocorrelation of the raw daily HOBO label series on the "
             "split's TRAINING dates (consecutive-day pairs only). p_dry "
             "bootstrapped from the split's val-day predictions; observed dry "
             "d/yr = (1 - val wet fraction) x 365; 10,000 sims/site.", "",
             "Caveats: ph val = 36 days across three phases; q65/q80 val = "
             "late season only (dry-biased 'typical year'); 'site' scores the "
             "flag_sh model at reaches whose labels were masked from its "
             "loss (with-sensor spatial regime).", ""]
    for split in ("ph", "q65", "q80", "site"):
        lines += run_split(split, rng)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
