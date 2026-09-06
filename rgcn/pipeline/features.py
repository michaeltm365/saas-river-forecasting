"""Canonical feature ordering — the single source of truth for the input tensor.

Feature order MUST keep the 11 GridMET drivers first (indices 0-10) so
rgcn/inspect_driver_weights.py continues to read driver rows correctly. Static
features are appended after the original 20-feature block (plan §11).

    idx  0-10 : 11 GridMET drivers                         (z-scored, train-only)
    idx 11-12 : Discharge_CMS log-lags (1, 7)              (log1p then z-scored)
    idx 13-14 : HoboWetDry0.05 lags (1, 7)                 (binary 0/1, not scaled)
    idx 15-16 : MaxDepth_Censor, MaxDepth_Threshold        (binary 0/1, not scaled)
    idx    17 : MaxDepth_cm                                 (z-scored, train-only)
    idx 18-19 : month, day                                 (z-scored, train-only)
    idx 20-36 : 17 static NHDPlus watershed vars            (z-scored, train-only)
"""

from __future__ import annotations

# 11 GridMET meteorological drivers (columns of met_drivers.csv, in file order).
DRIVER_VARS = [
    "etalfalfa", "etgrass", "prcp", "rhmax", "rhmin", "sph",
    "srad", "tmax", "tmin", "vp", "ws",
]

# Observation-derived lag features (built per node from obs.csv).
DISCHARGE_LAG_VARS = ["Discharge_CMS_lag_1", "Discharge_CMS_lag_7"]
WETDRY_LAG_VARS = ["HoboWetDry0.05_lag_1", "HoboWetDry0.05_lag_7"]

# Raw MaxDepth obs features (censor/threshold are binary flags; cm is continuous).
MAXDEPTH_BINARY_VARS = ["MaxDepth_Censor", "MaxDepth_Threshold"]
MAXDEPTH_CONT_VARS = ["MaxDepth_cm"]

# Temporal features.
TEMPORAL_VARS = ["month", "day"]

# 17 static NHDPlus watershed vars (static_vars.csv minus NHDPlusID/FromNode/ToNode).
STATIC_VARS = [
    "aspect_ne_pct", "aspect_sw_pct", "aspect_nw_pct", "aspect_se_pct",
    "elev_min_cm", "elev_max_cm", "elev_median_cm", "elev_mean_cm",
    "slp_median_pct", "slp_mean_pct", "curv_median", "curv_mean",
    "ArbolateSu", "AreaSqKm", "TotDASqKm", "Slope", "LengthKM",
]

# Full ordered feature list fed to the model (input_dim = len(FEATURE_VARS) = 37).
FEATURE_VARS = (
    DRIVER_VARS
    + DISCHARGE_LAG_VARS
    + WETDRY_LAG_VARS
    + MAXDEPTH_BINARY_VARS
    + MAXDEPTH_CONT_VARS
    + TEMPORAL_VARS
    + STATIC_VARS
)

# Targets: classification (wet/dry) + regression (log-discharge).
TARGET_VARS = ["HoboWetDry0.05", "Discharge_CMS"]
WETDRY_IDX = TARGET_VARS.index("HoboWetDry0.05")
DISCHARGE_IDX = TARGET_VARS.index("Discharge_CMS")

# Which feature columns get z-score standardization (train-only stats). Binary
# flags and the wet/dry lags stay on their native 0/1 scale.
ZSCORE_FEATURES = (
    DRIVER_VARS
    + DISCHARGE_LAG_VARS
    + MAXDEPTH_CONT_VARS
    + TEMPORAL_VARS
    + STATIC_VARS
)
# Discharge lags are log1p-transformed before z-scoring (spans many orders of
# magnitude), mirroring the log1p applied to the discharge target.
LOG1P_FEATURES = set(DISCHARGE_LAG_VARS)

# Non-time-varying features (broadcast across all timesteps of a node).
STATIC_FEATURE_SET = set(STATIC_VARS)

INPUT_DIM = len(FEATURE_VARS)

# Number of time-varying feature columns (the leading block of FEATURE_VARS;
# X_time's last dim). Statics are appended after these at window assembly.
N_TIME_FEATURES = INPUT_DIM - len(STATIC_VARS)


def feature_index(name: str) -> int:
    return FEATURE_VARS.index(name)


def time_feature_selection(exclude: list[str] | None):
    """Column subset of the time-varying block for feature ablations.

    ``exclude`` names time-varying features to drop (e.g. the 4 obs-lag vars
    for the no-lag spatial-transfer variant). Returns (keep_indices,
    effective_feature_vars) where effective_feature_vars is the full ordered
    input list (kept time features + statics) whose length is the model's
    input_dim. Static features cannot be excluded here (use a model ablation
    for that). Raises on unknown or non-time-varying names.
    """
    exclude = list(exclude or [])
    time_vars = FEATURE_VARS[:N_TIME_FEATURES]
    unknown = [v for v in exclude if v not in time_vars]
    if unknown:
        raise ValueError(
            f"features.exclude_time contains non-time-varying or unknown vars: {unknown}"
        )
    keep = [i for i, v in enumerate(time_vars) if v not in exclude]
    effective = [time_vars[i] for i in keep] + STATIC_VARS
    return keep, effective
