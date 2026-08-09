"""eval_hjflp.py — held-out-site evaluation on HJFlp single-visit observations.

The HJFlp campaign (obs.csv SiteIDCode prefix "HJFlp") is 472 one-off field
visits in Jul-Oct 2020 recording Flow_Status: 0 = dry, 1 = flowing, 0.5 =
partial. None of these rows carry HoboWetDry0.05, so they were never training
targets; and reaches with no obs.csv history at all have all-zero lag/MaxDepth
inputs by construction — for those, the RGCN prediction rests purely on
drivers + statics + topology. That makes HJFlp a genuine ungauged-site
(spatial transferability) test requiring no retraining.

Scores the exported predictions against HJFlp visits, per horizon and by
site-history tier:
    unobserved  — reach has NO obs.csv rows at all (strictest: truly ungauged)
    no-labels   — reach has some obs (e.g. discharge) but no wet/dry labels
    labeled     — reach carries HOBO wet/dry labels (not a spatial holdout)

Flow_Status=0.5 ("partial") rows are scored separately against both label
conventions and excluded from the headline metrics.

Run:  RGCN_CONFIG=rgcn/config_consistph.yml uv run python -m rgcn.pipeline.eval_hjflp
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import load_config


def load_hjflp(config) -> pd.DataFrame:
    obs = pd.read_csv(config.path("obs_csv"))
    obs["Date"] = pd.to_datetime(obs["Date"])
    is_flp = obs["SiteIDCode"].astype(str).str.startswith("HJFlp")

    flp = obs[is_flp & obs["Flow_Status"].notna()].copy()
    flp["NHDPlusID"] = flp["NHDPlusID"].astype("int64")

    # Site-history tiers from the NON-HJFlp portion of obs.csv.
    rest = obs[~is_flp]
    has_any = set(rest.loc[rest[["Discharge_CMS", "HoboWetDry0.05", "MaxDepth_cm"]]
                           .notna().any(axis=1), "NHDPlusID"].astype("int64"))
    has_label = set(rest.loc[rest["HoboWetDry0.05"].notna(), "NHDPlusID"].astype("int64"))

    def tier(nid):
        if nid in has_label:
            return "labeled"
        if nid in has_any:
            return "no-labels"
        return "unobserved"

    flp["tier"] = flp["NHDPlusID"].map(tier)
    return flp[["NHDPlusID", "Date", "Flow_Status", "tier"]]


def _metrics(y_true, y_prob, y_pred):
    out = {"N": len(y_true)}
    if len(y_true) == 0:
        return out
    out["WetFrac"] = float(np.mean(y_true))
    out["Accuracy"] = accuracy_score(y_true, y_pred)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["ROC-AUC"] = (roc_auc_score(y_true, y_prob)
                      if len(np.unique(y_true)) > 1 else float("nan"))
    out["DryRecall"] = (float(((y_true == 0) & (y_pred == 0)).sum() / max((y_true == 0).sum(), 1))
                        if (y_true == 0).any() else float("nan"))
    return out


def main() -> int:
    config = load_config()
    flp = load_hjflp(config)
    pred_dir = config.path("predictions_dir")

    lines = ["# RGCN held-out-site evaluation — HJFlp single-visit observations", ""]
    lines.append(f"Checkpoint: `{config['paths']['checkpoint']}` | "
                 f"predictions: `{config['paths']['predictions_dir']}`")
    lines.append("")
    tiers = flp["tier"].value_counts().to_dict()
    lines.append(f"HJFlp visits: {len(flp)} across {flp['NHDPlusID'].nunique()} reaches "
                 f"({flp['Date'].min().date()} .. {flp['Date'].max().date()}). "
                 f"Visits by site tier: {tiers}.")
    lines.append("")
    lines.append("| Horizon | Tier | N visits | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|")

    partial_rows = []
    joined = []
    for h in (1, 2, 3):
        df = pd.read_csv(pred_dir / f"train_val_predictions_day{h}.csv",
                         usecols=["date", "site_id", "pred_wetdry_prob", "pred_wetdry_label"],
                         parse_dates=["date"])
        # A date may fall in several windows at this horizon; average the prob.
        df = (df.groupby(["site_id", "date"], as_index=False)
                .agg(pred_wetdry_prob=("pred_wetdry_prob", "mean")))
        df["pred_wetdry_label"] = (df["pred_wetdry_prob"] >= 0.5).astype(int)
        j = flp.merge(df, left_on=["NHDPlusID", "Date"],
                      right_on=["site_id", "date"], how="inner")
        j["horizon"] = f"Day {h}"
        joined.append(j)

        part = j[j["Flow_Status"] == 0.5]
        if len(part):
            partial_rows.append(
                f"| Day {h} | {len(part)} | {part['pred_wetdry_prob'].mean():.3f} | "
                f"{(part['pred_wetdry_label'] == 1).mean():.3f} |")

    # With stride == horizon, each visit date is scored at exactly one horizon,
    # so per-horizon cells are small; the pooled block is the headline number.
    pooled = pd.concat(joined, ignore_index=True)
    pooled["horizon"] = "All"
    for j in [pooled] + joined:
        h = j["horizon"].iloc[0] if len(j) else "?"
        binary = j[j["Flow_Status"].isin([0.0, 1.0])]
        for tier_name in ("unobserved", "no-labels", "labeled", "ALL"):
            s = binary if tier_name == "ALL" else binary[binary["tier"] == tier_name]
            if len(s) == 0:
                continue
            m = _metrics(s["Flow_Status"].astype(int).to_numpy(),
                         s["pred_wetdry_prob"].to_numpy(),
                         s["pred_wetdry_label"].to_numpy())
            lines.append(
                f"| {h} | {tier_name} | {m['N']:,} | {m.get('WetFrac', float('nan')):.2f} | "
                f"{m.get('Accuracy', float('nan')):.3f} | "
                f"{m.get('ROC-AUC', float('nan')):.3f} | {m.get('F1', float('nan')):.3f} | "
                f"{m.get('DryRecall', float('nan')):.3f} |")

    if partial_rows:
        lines.append("")
        lines.append("## Flow_Status = 0.5 (partial flow) — excluded from metrics above")
        lines.append("")
        lines.append("| Horizon | N | Mean P(wet) | Frac predicted wet |")
        lines.append("|---|--:|--:|--:|")
        lines.extend(partial_rows)

    lines.append("")
    lines.append("Notes: 'unobserved' reaches have no obs.csv history, so their lag and")
    lines.append("MaxDepth inputs are all-zero — predictions rely purely on meteorology,")
    lines.append("statics, and network topology (true ungauged-site transfer). 'labeled'")
    lines.append("reaches host HOBO sensors used in training and are not a spatial holdout.")

    out = config.repo_root / config["paths"].get(
        "hjflp_report", "results/rgcn_eval_hjflp.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
