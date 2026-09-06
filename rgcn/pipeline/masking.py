"""Forecast-tail input masking — make day-2/3 forecasts honest.

Without masking, the window's forecast tail (positions seq_len .. wl-1, i.e.
days t+1 .. t+horizon) carries observation-derived inputs from *inside* the
forecast period: lag-1 features at day t+k reference the true observation at
t+k-1, and the MaxDepth columns are same-day sensor readings. The LSTM
baseline sees nothing after day t, so the unmasked RGCN day-3 metric is
effectively a day-1 metric wherever observations exist.

Masking replaces each leaking column with its last legitimately-known value
("freeze"), matching what a deployed forecaster would have at issue time t:

    mode "obs"          (variant A)
      - lag-1 columns (Discharge_CMS_lag_1, HoboWetDry0.05_lag_1): positions
        seq_len+1 .. wl-1 are frozen to position seq_len (whose lag-1 value is
        the day-t observation — known at issue time). Position seq_len (day
        t+1) is already legitimate.
      - MaxDepth columns (Censor/Threshold/cm): positions seq_len .. wl-1
        frozen to position seq_len-1 (day t, the last observed day).
      - lag-7 columns are untouched: with forecast_horizon <= 7 they only ever
        reference days <= t.
      - month/day and (in this mode) GridMET drivers are untouched — calendar
        is deterministic and drivers stand in for a weather forecast.

    mode "obs+drivers"  (variant A-strict)
      - everything in "obs", plus the 11 driver columns frozen to their day-t
        values (a persistence "weather forecast"): the model gets NO
        information dated after t, exactly matching the LSTM's information set.

    mode "none" — leave the window as built (released/retrain behavior).
"""

from __future__ import annotations

import torch

from . import features as F

MASK_MODES = ("none", "obs", "obs+drivers")

_LAG1_COLS = [F.feature_index("Discharge_CMS_lag_1"),
              F.feature_index("HoboWetDry0.05_lag_1")]
_LAG7_MIN_LAG = 7
_MAXDEPTH_COLS = [F.feature_index(v)
                  for v in F.MAXDEPTH_BINARY_VARS + F.MAXDEPTH_CONT_VARS]
_DRIVER_COLS = [F.feature_index(v) for v in F.DRIVER_VARS]


def mask_forecast_tail(xt: torch.Tensor, seq_len: int, mode: str) -> torch.Tensor:
    """Apply forecast-tail masking to a time-feature window tensor.

    xt: (..., wl, N, n_time_features) — window batch with time as dim -3.
    Returns xt (modified in place for efficiency; also returned).
    """
    if mode == "none":
        return xt
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown forecast mask mode: {mode!r} (valid: {MASK_MODES})")
    wl = xt.shape[-3]
    horizon = wl - seq_len
    if horizon > _LAG7_MIN_LAG:
        raise ValueError(
            f"forecast_horizon={horizon} exceeds the smallest unmasked lag "
            f"({_LAG7_MIN_LAG}); lag-7 columns would leak and need masking too."
        )

    # Lag-1 columns: freeze positions seq_len+1.. at position seq_len's value.
    frozen = xt[..., seq_len : seq_len + 1, :, _LAG1_COLS]
    xt[..., seq_len + 1 :, :, _LAG1_COLS] = frozen

    # MaxDepth (same-day obs): freeze the whole tail at day t's value.
    frozen = xt[..., seq_len - 1 : seq_len, :, _MAXDEPTH_COLS]
    xt[..., seq_len:, :, _MAXDEPTH_COLS] = frozen

    if mode == "obs+drivers":
        frozen = xt[..., seq_len - 1 : seq_len, :, _DRIVER_COLS]
        xt[..., seq_len:, :, _DRIVER_COLS] = frozen
    return xt


def mask_mode(config) -> str:
    """Read masking.forecast_mask from config (default 'none')."""
    return (config.get("masking") or {}).get("forecast_mask", "none")
