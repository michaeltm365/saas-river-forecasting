"""Gaussian-copula annual dry-day estimation with probability calibration.

Canonical method (q65 split): per-day dry probabilities from the RGCN's
stride-1 Day-3 export, passed through a Platt calibrator fit on the model's
TRAINING-period Day-3 predictions (HOBO-labeled rows), then integrated over a
365-day year via a Gaussian copula with AR(1) temporal correlation. rho is the
lag-1 autocorrelation of the raw daily HOBO series over training dates
(consecutive-calendar-day pairs only), clipped at RHO_CEIL so near-unit
estimates keep the AR(1) intervals informative.

Simulation randomness is seeded per (seed, site), so results are independent
of site ordering and identical between benchmarks/flagship_copula_all.py and
rgcn/rgcn_eval.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from hja.paths import SCIENCEBASE

RHO_CEIL = 0.98
N_SIMS = 10_000
HORIZON_DAYS = 365


def load_hobo_daily() -> pd.DataFrame:
    """Raw daily HOBO wet/dry series: one row per (NHDPlusID, Date), the
    mean-rounded HoboWetDry0.05 value."""
    obs = pd.read_csv(SCIENCEBASE / "obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["NHDPlusID"] = obs["NHDPlusID"].astype("int64")
    return (obs.groupby(["NHDPlusID", "Date"])["HoboWetDry0.05"].mean()
               .round().dropna().rename("wet").reset_index())


def fit_platt(p_wet, y_wet):
    """Platt scaling: near-unregularized logistic fit of wet labels on the
    model's wet probabilities. Returns a vectorized p_wet -> calibrated
    p_wet function."""
    lr = LogisticRegression(C=1e6).fit(
        np.asarray(p_wet).reshape(-1, 1), np.asarray(y_wet).astype(int))
    return lambda p: lr.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def rho_lag1(daily: pd.DataFrame, site: int, train_dates) -> tuple[float, int]:
    """Lag-1 autocorrelation from consecutive-calendar-day pairs of the raw
    daily HOBO series restricted to training dates. `train_dates` is a
    callable mapping a Date Series to a boolean mask."""
    s = daily[daily["NHDPlusID"] == site].sort_values("Date")
    s = s[train_dates(s["Date"])]
    v = s["wet"].to_numpy()
    consec = s["Date"].diff().dt.days.to_numpy()[1:] == 1
    a, b = v[:-1][consec], v[1:][consec]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0, int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))


def simulate_annual(p_dry, rho: float, site: int, seed: int = 42,
                    n_sims: int = N_SIMS, days: int = HORIZON_DAYS):
    """Copula Monte Carlo: (mean, lo, hi) annual dry-day count. Correlated
    uniforms from a stationary AR(1) Gaussian; p_dry bootstrapped per day."""
    rng = np.random.default_rng([seed, int(site) % (2**63)])
    innov = rng.standard_normal((n_sims, days)) * np.sqrt(max(1 - rho**2, 0.0))
    z = np.empty((n_sims, days))
    z[:, 0] = rng.standard_normal(n_sims)
    for t in range(1, days):
        z[:, t] = rho * z[:, t - 1] + innov[:, t]
    u = norm.cdf(z)
    sampled = rng.choice(np.asarray(p_dry), size=(n_sims, days), replace=True)
    counts = (u < sampled).sum(axis=1)
    lo, hi = np.percentile(counts, [2.5, 97.5])
    return float(counts.mean()), float(lo), float(hi)


def site_row(site: int, val_probs, val_true_wet, daily: pd.DataFrame,
             train_dates, calibrate=None, rho_ceil: float | None = RHO_CEIL,
             seed: int = 42) -> dict:
    """One site's copula row. `calibrate` maps raw p_wet to calibrated p_wet
    (None = raw); `rho_ceil=None` disables clipping."""
    p_wet = np.asarray(val_probs, dtype=float)
    if calibrate is not None:
        p_wet = np.clip(calibrate(p_wet), 0.0, 1.0)
    rho, n_pairs = rho_lag1(daily, site, train_dates)
    rho_used = min(rho, rho_ceil) if rho_ceil is not None else rho
    mean, lo, hi = simulate_annual(1.0 - p_wet, rho_used, site, seed)
    true = float((1.0 - np.asarray(val_true_wet).round().mean()) * HORIZON_DAYS)
    return dict(site=int(site), n=len(p_wet), rho=rho, rho_used=rho_used,
                np=n_pairs, true=true, mean=mean, lo=lo, hi=hi,
                ok=bool(lo <= true <= hi))
