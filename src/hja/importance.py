"""Feature-importance helpers shared across model notebooks (the category
color scheme and renames behind the paper's Figure 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

RENAME = {
    "Out-degree": "out_degree",
    "In-degree": "in_degree",
    "wetdry_status": "lagged_target",
}

FEATURE_CATEGORIES = {
    "lagged_target": "Lagged Target",
    "in_degree": "Degrees",
    "out_degree": "Degrees",
}
for _v in ["tmax", "tmin", "srad", "prcp", "vp", "ws",
           "etgrass", "etalfalfa", "rhmax", "rhmin", "sph"]:
    FEATURE_CATEGORIES[_v] = "Drivers"
for _v in ["ArbolateSu", "AreaSqKm", "TotDASqKm", "Slope", "LengthKM",
           "FromNode", "ToNode",
           "aspect_ne_pct", "aspect_nw_pct", "aspect_se_pct", "aspect_sw_pct",
           "curv_mean", "curv_median",
           "elev_max_cm", "elev_mean_cm", "elev_median_cm", "elev_min_cm",
           "slp_mean_pct", "slp_median_pct"]:
    FEATURE_CATEGORIES[_v] = "Static"
for _v in ["MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor"]:
    FEATURE_CATEGORIES[_v] = "Other Obs"

CATEGORY_COLORS = {
    "Lagged Target": "#d62728",  # red
    "Drivers":       "#1f77b4",  # blue
    "Static":        "#2ca02c",  # green
    "Degrees":       "#ff7f0e",  # orange
    "Other Obs":     "#9467bd",  # purple
    "Other":         "#7f7f7f",  # gray
}


def feature_color(name: str) -> str:
    return CATEGORY_COLORS[FEATURE_CATEGORIES.get(name, "Other")]


def add_category_legend(ax, features):
    seen = []
    for f in features:
        cat = FEATURE_CATEGORIES.get(f, "Other")
        if cat not in seen:
            seen.append(cat)
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in seen]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)


def importance_frame(features, values) -> pd.DataFrame:
    """Raw importance values -> sorted table with display names and a 0-1
    max-scaled column (the notebooks' shared presentation)."""
    df = pd.DataFrame({"Feature": list(features), "Value": np.asarray(values)})
    df["Abs"] = df["Value"].abs()
    df = df.sort_values("Abs", ascending=False).reset_index(drop=True)
    df["Display"] = df["Feature"].replace(RENAME)
    df["Importance"] = df["Abs"] / max(df["Abs"].max(), 1e-12)
    return df


def plot_top10(df: pd.DataFrame, title: str, ax=None):
    """Horizontal bar chart of the top-10 rows of an importance_frame()."""
    import matplotlib.pyplot as plt

    top = df.head(10)
    names = top["Display"][::-1].tolist()
    vals = top["Importance"][::-1].tolist()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color=[feature_color(f) for f in names])
    ax.set_xlabel("Feature Importance (0-1 Maximum Scaling)")
    ax.set_title(title)
    add_category_legend(ax, names)
    return ax
