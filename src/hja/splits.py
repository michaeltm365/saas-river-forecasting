"""Train/test split strategies.

Released splits (verbatim from lr.ipynb / xgb.ipynb cells 10/12/14 — these
seeds and orderings produce the paper's released LR/XGBoost numbers):

- random_split:   80/20 shuffled row split (random_state=42).
- temporal_split: rows before 2020-09-15 train, on/after test.
- site_split:     first 80% of SiteIDCodes (in date-sorted first-appearance
                  order) train, last 20% test.

"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Released splits (LR / XGBoost / LSTM-HOBO protocol)
# --------------------------------------------------------------------------- #


def random_split(central_df: pd.DataFrame):
    X = central_df.drop("wet_dry_next", axis=1)
    y = central_df["wet_dry_next"]
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def temporal_split(central_df: pd.DataFrame, split_date: str = "2020-9-15"):
    df = central_df.sort_values(["Date"])
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]
    return (train.drop("wet_dry_next", axis=1), test.drop("wet_dry_next", axis=1),
            train["wet_dry_next"], test["wet_dry_next"])


def site_split(central_df: pd.DataFrame):
    central_df = central_df.sort_values(["Date"]).reset_index(drop=True)
    sites = central_df["SiteIDCode"].unique()
    train_sites = sites[:int(0.8 * len(sites))]
    test_sites = sites[int(0.8 * len(sites)):]
    train = central_df[central_df["SiteIDCode"].isin(train_sites)].copy()
    test = central_df[central_df["SiteIDCode"].isin(test_sites)].copy()
    return (train.drop("wet_dry_next", axis=1), test.drop("wet_dry_next", axis=1),
            train["wet_dry_next"], test["wet_dry_next"])


RELEASED_SPLITS = {
    "random": random_split,
    "temporal": temporal_split,
    "site": site_split,
}
