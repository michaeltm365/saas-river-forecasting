"""Shared data construction for the supervised baselines.

Two central frames, both replicating the released notebook preprocessing
row-for-row (the notebooks are visualization layers over these functions):

- build_hobo_frame():      HOBO-sensor rows only (lr.ipynb / xgb.ipynb cells
                           5-8; lstm_hobo_sites.ipynb cells 5-6 with
                           include_order=False).
- build_allsites_frame():  HOBO + discretized-discharge rows
                           (lstm_all_sites.ipynb cells 6-7).

Both target `wet_dry_next` = wet/dry status 3 rows ahead within site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hja.paths import HUGGINGFACE, RETRAIN, SCIENCEBASE

OBS_EXTRA_COLS = ["MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor"]
BINARY_COLS = {"MaxDepth_Threshold", "MaxDepth_Censor", "wetdry_status"}
DROP_COLS = ["NHDPlusID", "SiteIDCode", "Date", "wet_dry_next",
             "StreamOrde", "FCode", "n_discharge", "n_water_presence",
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


def build_hobo_frame(include_order: bool = True) -> pd.DataFrame:
    """HOBO-only frame.

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
                          .transform(lambda g: g.ffill().bfill()))
    df[OBS_EXTRA_COLS] = df[OBS_EXTRA_COLS].fillna(0)

    df["wet_dry_next"] = df.groupby("NHDPlusID")["wetdry_status"].shift(-3)
    df = df.dropna(subset=["wet_dry_next"])
    return df.reset_index(drop=True)


def build_allsites_frame() -> pd.DataFrame:
    """HOBO + discretized-discharge frame (mirrors lstm_all_sites cells 6-7)."""
    obs = pd.read_csv(SCIENCEBASE / "obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs_wide = (obs.drop(columns="SiteIDCode", errors="ignore")
                   .groupby(["NHDPlusID", "Date"], as_index=False).first())
    obs_wide["NHDPlusID"] = obs_wide["NHDPlusID"].astype("int64")
    drivers, statics, degrees, _ = _aux_tables()

    df = obs_wide.merge(drivers, on=["NHDPlusID", "Date"], how="left")
    df = df.merge(statics, on="NHDPlusID", how="left")
    df = df.merge(degrees, on="NHDPlusID", how="left")

    df["is_hobo"] = df["HoboWetDry0.05"].notna().astype(int)
    df["wetdry_discharge"] = (df["Discharge_CMS"] >= DRY_THRESHOLD).astype(int)
    df["wetdry_status"] = df["HoboWetDry0.05"].fillna(df["wetdry_discharge"])
    df = df[df["HoboWetDry0.05"].notna() | df["Discharge_CMS"].notna()]

    df = df.sort_values(["NHDPlusID", "Date"])
    df[DRIVER_COLS] = (df.groupby("NHDPlusID")[DRIVER_COLS]
                       .transform(lambda g: g.ffill().bfill()))
    df[OBS_EXTRA_COLS] = (df.groupby("NHDPlusID")[OBS_EXTRA_COLS]
                          .transform(lambda g: g.ffill().bfill()))
    df[OBS_EXTRA_COLS] = df[OBS_EXTRA_COLS].fillna(0)

    df["wet_dry_next"] = df.groupby("NHDPlusID")["wetdry_status"].shift(-3)
    df = df.dropna(subset=["wet_dry_next"])
    df = df.drop(columns=["wetdry_discharge", "FromNode", "ToNode",
                          "Flow_Status", "HoboWetDry0.05", "Discharge_CMS"],
                 errors="ignore")
    return df.reset_index(drop=True)


def feature_frame(df: pd.DataFrame):
    """Numeric feature matrix + feature-name list (everything numeric outside
    DROP_COLS, with the released global ffill/bfill/zero-fill)."""
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in DROP_COLS]
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
