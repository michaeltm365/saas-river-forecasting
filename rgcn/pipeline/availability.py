"""Optional lag-1 status availability, derived before missing targets are filled."""

import torch

from . import features as F

NAME = "status_available_lag_1"


def enabled(config):
    return bool((config.get("features") or {}).get("status_availability", False))


def append_channel(x_time, wetdry):
    assert x_time.shape[-1] == F.N_TIME_FEATURES
    available = torch.zeros_like(wetdry, dtype=x_time.dtype)
    available[1:] = torch.isfinite(wetdry[:-1]).to(x_time.dtype)
    return torch.cat([x_time, available.unsqueeze(-1)], dim=-1)


def selection(exclude, use_availability):
    keep, names = F.time_feature_selection(exclude)
    if use_availability:
        keep.append(F.N_TIME_FEATURES)
        names.insert(len(keep) - 1, NAME)
    return keep, names
