"""Shared data construction for the supervised baselines.

HOBO-only input construction with causal depth filling:

- build_hobo_frame():      HOBO-sensor rows only (lr.ipynb / xgb.ipynb cells
                           5-8; lstm_hobo_sites.ipynb cells 5-6 with
                           include_order=False).
The target is `wet_dry_next` = observed wet/dry status exactly 3 calendar days ahead within reach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hja.paths import HUGGINGFACE, RETRAIN, SCIENCEBASE

OBS_EXTRA_COLS = ["MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor"]
BINARY_COLS = {"MaxDepth_Threshold", "MaxDepth_Censor", "wetdry_status"}
DROP_COLS = ["NHDPlusID", "SiteIDCode", "Date", "wet_dry_next",
             "target_date", "StreamOrde", "FCode", "n_discharge", "n_water_presence",
             "has_data", "is_hobo"]
DRY_THRESHOLD = 0.00014  # CMS; discharge below this is a discretized "dry"
DRIVER_COLS = ["etalfalfa", "etgrass", "prcp", "rhmax", "rhmin", "sph",
               "srad", "tmax", "tmin", "vp", "ws"]


def load_drivers() -> pd.DataFrame:
    """GridMET drivers; prefers the parquet cache written by
    rgcn.pipeline.prepare_data, falling back to the raw 2.13 GB CSV."""
    pq = RETRAIN / "met_drivers.parquet"
    if pq.exists():
        drivers = pd.read_parquet(pq)
    else:
        drivers = pd.read_csv(SCIENCEBASE / "met_drivers.csv")
    drivers["NHDPlusID"] = drivers["NHDPlusID"].astype("int64")
    drivers["Date"] = pd.to_datetime(drivers["Date"])
    return drivers


def _aux_tables():
    drivers = load_drivers()
    statics = pd.read_csv(SCIENCEBASE / "static_vars.csv")
    degrees = pd.read_parquet(HUGGINGFACE / "degrees.parquet")
    order = pd.read_csv(HUGGINGFACE / "nhd_id_stream_order_permanence.csv")
    for df in (statics, degrees, order):
        df["NHDPlusID"] = df["NHDPlusID"].astype("int64")
    return drivers, statics, degrees, order


def build_hobo_frame(include_order: bool = True, include_target_dates: bool = False,
                     keep_unlabeled: bool = False) -> pd.DataFrame:
    """HOBO-only frame.

    Depth features are filled forward within reach; leading missing values are zero.
    include_target_dates retains target-date metadata for chronological sequences.
    keep_unlabeled preserves input history rows without an observed t+3 target;
    sequence callers must exclude their unlabeled endpoints after windowing.

    include_order=True mirrors lr.ipynb / xgb.ipynb cells 5-8 (stream-order
    columns merged in, excluded from features via DROP_COLS and used for
    post-hoc breakdowns). include_order=False mirrors lstm_hobo_sites.ipynb
    cells 5-6, which never merge the stream-order table.
    """
    obs = pd.read_csv(SCIENCEBASE / "obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    hobo = obs[obs["HoboWetDry0.05"].notna()][
        ["NHDPlusID", "SiteIDCode", "Date", "HoboWetDry0.05"]
    ].rename(columns={"HoboWetDry0.05": "wetdry_status"}).copy()
    maxd = obs.loc[obs[OBS_EXTRA_COLS].notna().any(axis=1),
                   ["NHDPlusID", "Date"] + OBS_EXTRA_COLS]
    for df in (hobo, maxd):
        df["NHDPlusID"] = df["NHDPlusID"].astype("int64")
    drivers, statics, degrees, order = _aux_tables()

    df = hobo.merge(drivers, on=["NHDPlusID", "Date"], how="inner")
    df = df.merge(statics, on="NHDPlusID", how="left")
    df = df.merge(degrees, on="NHDPlusID", how="left")
    if include_order:
        df = df.merge(order, on="NHDPlusID", how="left")
    df = df.merge(maxd, on=["NHDPlusID", "Date"], how="left")

    df = df.sort_values(["NHDPlusID", "Date"])
    df[OBS_EXTRA_COLS] = (df.groupby("NHDPlusID")[OBS_EXTRA_COLS]
                          .transform(lambda g: g.ffill()))
    df[OBS_EXTRA_COLS] = df[OBS_EXTRA_COLS].fillna(0)

    df = attach_hobo_calendar_targets(df, obs)
    if not keep_unlabeled:
        df = df.dropna(subset=["wet_dry_next"])
    if not include_target_dates:
        df = df.drop(columns="target_date")
    return df.reset_index(drop=True)


def attach_hobo_calendar_targets(frame: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Look up labels at exactly issue date + 3 days from raw observations,
    independent of feature joins. Missing target dates stay missing.
    """
    truth = obs.loc[obs["HoboWetDry0.05"].notna(),
                    ["NHDPlusID", "Date", "HoboWetDry0.05"]].copy()
    truth["Date"] = pd.to_datetime(truth["Date"])
    if truth.duplicated(["NHDPlusID", "Date"]).any():
        raise ValueError("HOBO target lookup requires unique reach/date labels")
    lookup = truth.set_index(["NHDPlusID", "Date"])["HoboWetDry0.05"]
    frame = frame.copy()
    frame["target_date"] = pd.to_datetime(frame["Date"]) + pd.Timedelta(days=3)
    keys = pd.MultiIndex.from_arrays([frame.NHDPlusID, frame.target_date],
                                     names=["NHDPlusID", "Date"])
    frame["wet_dry_next"] = lookup.reindex(keys).to_numpy()
    return frame


def feature_frame(df: pd.DataFrame, causal: bool = False):
    """Numeric feature matrix + feature-name list (everything numeric outside
    DROP_COLS). With causal=True, fill only forward within each reach;
    otherwise retain the legacy global filling used by older callers."""
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in DROP_COLS]
    if causal:
        ordered = df.sort_values(["NHDPlusID", "Date"])
        X = ordered.groupby("NHDPlusID")[feats].ffill().fillna(0).reindex(df.index)
    else:
        X = df[feats].copy().ffill().bfill().fillna(0)
    return X, feats


def scale_train_only(X_tr, X_te, feats):
    """Z-score continuous features with statistics fit on the training rows
    only; binary columns are left unscaled."""
    cont = [c for c in feats if c not in BINARY_COLS]
    scaler = StandardScaler().fit(X_tr[cont])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[cont] = scaler.transform(X_tr[cont])
    X_te[cont] = scaler.transform(X_te[cont])
    return X_tr, X_te
