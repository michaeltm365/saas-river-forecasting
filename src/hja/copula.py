"""Raw Gaussian-copula counts on sensor-observed dates."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from hja.paths import SCIENCEBASE, REPO
N_SIMS = 10_000

def load_hobo_daily() -> pd.DataFrame:
    """Raw daily HOBO wet/dry series: one row per (NHDPlusID, Date), the
    mean-rounded HoboWetDry0.05 value."""
    saved = REPO / "results/paper/predictions/sensor_daily.csv"
    if saved.exists():
        return pd.read_csv(saved, parse_dates=["Date"])
    obs = pd.read_csv(SCIENCEBASE / "obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["NHDPlusID"] = obs["NHDPlusID"].astype("int64")
    return (obs.groupby(["NHDPlusID", "Date"])["HoboWetDry0.05"].mean()
               .round().dropna().rename("wet").reset_index())

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

def observed_period_counts(p_dry, dates, rho: float, site: int, seed: int = 42,
                           n_sims: int = N_SIMS) -> np.ndarray:
    """Simulated dry counts on the supplied dates, preserving calendar gaps.

    Probabilities stay attached to dates; no probability bootstrap or annual
    scaling. Latent AR(1) evolves on every calendar day, but only supplied
    dates contribute to the count. Inputs describe rolling forecasts, not
    a joint forecast issued before the whole period.
    """
    p = np.asarray(p_dry, dtype=float)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    if p.ndim != 1 or len(p) == 0 or len(p) != len(d):
        raise ValueError("Nonempty, aligned one-dimensional probabilities and dates required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if d.hasnans or d.has_duplicates or not (d == d.normalize()).all():
        raise ValueError("Dates must be unique, valid calendar days")
    if not np.isfinite(rho) or not -1 <= rho <= 1 or n_sims < 1:
        raise ValueError("rho must be in [-1,1] and n_sims positive")
    order = np.argsort(d)
    d, p = d[order], p[order]
    offsets = (d - d[0]).days.to_numpy()
    rng = np.random.default_rng([seed, int(site) % (2**63)])
    z = rng.standard_normal(n_sims)
    counts = np.zeros(n_sims, dtype=np.int32)
    scale = np.sqrt(max(1 - rho**2, 0.0))
    j = 0
    for day in range(int(offsets[-1]) + 1):
        if day:
            z = rho * z + scale * rng.standard_normal(n_sims)
        if day == offsets[j]:
            counts += norm.cdf(z) < p[j]
            j += 1
            if j == len(p):
                break
    return counts
