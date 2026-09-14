"""Read-only scoring of campaign predictions; all reports stay in campaign outputs."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from hja.evaluation import metrics
from hja.copula import load_hobo_daily, fit_platt, site_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/correction_sep11"
DATA = ROOT / "data/retrain"
KEYS = ["site_id", "target_date", "issue_date"]


def rgcn_path(seed, variant, daily=True):
    if variant == "control":
        name = "predictions_flag_q65" + ("" if seed == 42 else f"_s{seed}")
        parent = DATA / "flagship"
    else:
        name = f"predictions_rgcn_availability_s{seed}"
        parent = DATA / "correction_sep11"
    return parent / (name + ("_stride1" if daily else ""))


def sensors():
    return load_hobo_daily().rename(columns={"NHDPlusID": "site_id", "Date": "target_date", "wet": "sensor"})


def read_rgcn(seed, variant):
    df = pd.read_csv(rgcn_path(seed, variant) / "train_val_predictions_day3.csv", parse_dates=["date"])
    df = df.rename(columns={"date": "target_date"})
    df["issue_date"] = df.target_date - pd.Timedelta(days=3)
    return df[df.target_date > "2020-09-10"]


def copula():
    daily = load_hobo_daily()
    truth = sensors()
    val = read_rgcn(42, "control").merge(truth, on=["site_id", "target_date"], validate="one_to_one")
    tr = pd.read_csv(rgcn_path(42, "control", False) / "train_val_predictions_day3.csv", parse_dates=["date"])
    tr = tr.rename(columns={"date": "target_date"})
    tr = tr[tr.target_date <= "2020-09-10"].merge(truth, on=["site_id", "target_date"], validate="one_to_one")
    cal = fit_platt(tr.pred_wetdry_prob, tr.sensor)
    rows = []
    for sid, g in val.groupby("site_id"):
        if len(g) >= 20:
            rows.append(site_row(sid, g.pred_wetdry_prob, g.sensor, daily,
                                  lambda d: d <= pd.Timestamp("2020-09-10"), cal))
    pd.DataFrame(rows).to_csv(OUT / "copula_sensor_only.csv", index=False)
    result = dict(sensor_rows=len(val), calibration_rows=len(tr), reaches=len(rows),
                  covered=sum(r["ok"] for r in rows),
                  interpretation="Annualized late-season equivalents, not observed annual counts; in-sample training calibration")
    (OUT / "copula_sensor_only.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


def evaluate():
    truth = sensors()
    frames = {}
    for seed in (42, 43, 44):
        for variant in ("control", "availability"):
            for model in ("lstm", "rgcn"):
                df = (read_rgcn(seed, variant) if model == "rgcn" else
                      pd.read_csv(OUT / f"lstm_{variant}/preds_q65_s{seed}.csv",
                                  parse_dates=["target_date", "issue_date"]))
                assert not df.duplicated(KEYS).any()
                assert ((df.target_date - df.issue_date).dt.days == 3).all()
                df = df.merge(truth, on=["site_id", "target_date"], validate="one_to_one")
                frames[(model, variant, seed)] = df
    common = None
    for df in frames.values():
        keys = pd.MultiIndex.from_frame(df[KEYS])
        common = keys if common is None else common.intersection(keys)
    rows = []
    for (model, variant, seed), df in frames.items():
        selected = df.set_index(KEYS).loc[common]
        p = selected.pred_wetdry_prob.to_numpy()
        y = selected.sensor.to_numpy()
        m = metrics(y, p, p >= .5)
        m["DryPrecision"] = precision_score(y, p >= .5, pos_label=0, zero_division=0)
        rows.append(dict(model=model, variant=variant, seed=seed,
                         sensor_available=len(df), excluded_from_common=len(df)-len(selected), **m))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "matched_metrics.csv", index=False)
    cols = ["Accuracy", "ROC-AUC", "WetF1", "DryPrecision", "DryRecall", "DryF1"]
    summary = table.groupby(["model", "variant"])[cols].agg(["mean", "std"])
    summary.to_csv(OUT / "matched_summary.csv")
    (OUT / "SUMMARY.md").write_text(
        f"# Correction campaign\n\nCommon sensor rows: {len(common)}. Exact calendar t+3, threshold 0.5.\n\n"
        "Standard deviation across seeds measures training variability, not sampling uncertainty.\n\n"
        "```\n" + table.to_string(index=False) + "\n```\n\n```\n" + summary.to_string() + "\n```\n")
    # Discharge is kept on the existing stride-3 per-horizon convention for controls and variants.
    discharge = []
    for seed in (42, 43, 44):
        for variant in ("control", "availability"):
            for horizon in (1, 2, 3):
                df = pd.read_csv(rgcn_path(seed, variant, False) / f"train_val_predictions_day{horizon}.csv")
                df = df[df.date > "2020-09-10"].dropna(subset=["true_discharge", "pred_discharge"])
                y, p = df.true_discharge.to_numpy(), df.pred_discharge.to_numpy()
                nse = 1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2)
                kge = 1 - np.sqrt((np.corrcoef(y,p)[0,1]-1)**2 + (p.std()/y.std()-1)**2 + (p.mean()/y.mean()-1)**2)
                discharge.append(dict(seed=seed, variant=variant, horizon=horizon, N=len(y), NSE=nse, KGE=kge))
    pd.DataFrame(discharge).to_csv(OUT / "discharge_metrics.csv", index=False)
    print(summary, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copula-only", action="store_true")
    args = ap.parse_args()
    if args.copula_only:
        copula()
    else:
        evaluate()
