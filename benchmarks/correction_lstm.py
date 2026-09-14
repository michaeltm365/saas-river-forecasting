"""q65 correction campaign: calendar windows, causal filling, optional availability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lstm_flagship_splits as runner
from hja.data import (BINARY_COLS, DRIVER_COLS, OBS_EXTRA_COLS, DRY_THRESHOLD,
                      _aux_tables)
from hja.paths import SCIENCEBASE

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "data/retrain/correction_sep11"


def build_frame():
    obs = pd.read_csv(SCIENCEBASE / "obs.csv", parse_dates=["Date"])
    obs = obs.drop(columns="SiteIDCode").groupby(["NHDPlusID", "Date"], as_index=False).first()
    drivers, statics, degrees, _ = _aux_tables()
    lo, hi = drivers.Date.min(), drivers.Date.max()
    grids = []
    for sid, g in obs.groupby("NHDPlusID"):
        labeled = g[g["HoboWetDry0.05"].notna() | g.Discharge_CMS.notna()]
        if labeled.empty:
            continue
        dates = pd.date_range(max(lo, labeled.Date.min() - pd.Timedelta(days=32)),
                              min(hi, labeled.Date.max()), freq="D")
        if len(dates) < 33:
            continue
        daily = g.set_index("Date").reindex(dates).rename_axis("Date").reset_index()
        daily["NHDPlusID"] = int(sid)
        # Observation-derived values may propagate only forward within this site.
        for col in OBS_EXTRA_COLS:
            source = daily.Date.where(daily[col].notna()).ffill()
            assert (source.dropna() <= daily.loc[source.notna(), "Date"]).all()
            daily[col] = daily[col].ffill()
        grids.append(daily)
    df = pd.concat(grids, ignore_index=True)
    df = df.merge(drivers, on=["NHDPlusID", "Date"], how="left", validate="one_to_one")
    del drivers
    df = df.merge(statics, on="NHDPlusID", how="left", validate="many_to_one")
    df = df.merge(degrees, on="NHDPlusID", how="left", validate="many_to_one")
    df = df.sort_values(["NHDPlusID", "Date"]).reset_index(drop=True)
    # Missing weather can only use earlier weather, never a future reading.
    df[DRIVER_COLS] = df.groupby("NHDPlusID")[DRIVER_COLS].ffill()
    df["wetdry_status"] = df["HoboWetDry0.05"]
    dry = df.wetdry_status.isna() & df.Discharge_CMS.notna() & (df.Discharge_CMS <= DRY_THRESHOLD)
    df.loc[dry, "wetdry_status"] = 0.0
    df["status_available"] = df.wetdry_status.notna().astype(float)
    df["is_hobo"] = df["HoboWetDry0.05"].notna().astype(int)
    df["target_date"] = df.Date + pd.Timedelta(days=3)
    df["wet_dry_next"] = df.groupby("NHDPlusID").wetdry_status.shift(-3)
    df["label_is_hobo"] = df.groupby("NHDPlusID").is_hobo.shift(-3).fillna(0).astype(int)
    exclude = {"NHDPlusID", "Discharge_CMS", "HoboWetDry0.05", "Flow_Status",
               "FromNode", "ToNode", "wet_dry_next", "is_hobo", "label_is_hobo",
               "status_available"}
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    assert "wetdry_status" in feats
    for _, g in df.groupby("NHDPlusID"):
        assert (g.Date.diff().dropna() == pd.Timedelta(days=1)).all()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CAMPAIGN / "lstm_calendar_frame.parquet", index=False)
    (CAMPAIGN / "lstm_features.json").write_text(json.dumps(feats, indent=2))
    return df, feats


def sequences(df, feats):
    X, y, sites, dates, hobo = [], [], [], [], []
    for sid, g in df.groupby("NHDPlusID"):
        f = g[feats].to_numpy(dtype=np.float32)
        lab = g.wet_dry_next.to_numpy()
        for j in np.flatnonzero(np.isfinite(lab)):
            if j < 29:
                continue
            assert g.Date.iloc[j] - g.Date.iloc[j - 29] == pd.Timedelta(days=29)
            assert g.target_date.iloc[j] - g.Date.iloc[j] == pd.Timedelta(days=3)
            X.append(f[j - 29:j + 1])
            y.append(lab[j])
            sites.append(sid)
            dates.append(g.target_date.iloc[j])
            hobo.append(g.label_is_hobo.iloc[j])
    return (np.array(X, dtype=np.float32), np.array(y), np.array(sites),
            pd.DatetimeIndex(dates), np.array(hobo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    ap.add_argument("--variant", choices=["control", "availability"], default="control")
    args = ap.parse_args()
    if args.prepare:
        df, feats = build_frame()
        # Exercise geometry and provenance before any GPU job starts.
        x, y, sites, dates, hobo = sequences(df.fillna({c: 0 for c in feats}), feats)
        tr = dates <= pd.Timestamp("2020-09-10")
        audit = dict(rows=len(df), features=feats, sequences=len(y),
                     train=int(tr.sum()), val=int((~tr).sum()),
                     train_dry=int((y[tr] == 0).sum()), train_wet=int((y[tr] == 1).sum()),
                     sensor_val=int(hobo[~tr].sum()), exact_calendar=True,
                     causal_filling=True)
        (ROOT / "results/correction_sep11/preflight_data.json").write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the campaign")
    torch.set_num_threads(4)
    df = pd.read_parquet(CAMPAIGN / "lstm_calendar_frame.parquet")
    feats = json.loads((CAMPAIGN / "lstm_features.json").read_text())
    if args.variant == "availability":
        feats.append("status_available")
        BINARY_COLS.add("status_available")
    runner.make_sequences_dated = sequences
    runner.OUT = ROOT / "results/correction_sep11" / ("lstm_" + args.variant)
    runner.OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = CAMPAIGN / f"lstm_{args.variant}_s{args.seed}.pt"
    if checkpoint.exists() or (runner.OUT / f"preds_q65_s{args.seed}.csv").exists():
        raise FileExistsError("Run artifacts already exist; refusing to overwrite")
    extras = {}
    runner.train_eval("q65", args.seed, df, feats, torch.device("cuda:0"),
                      use_adasyn=False, extras=extras)
    torch.save(dict(model_state_dict=extras["model"].state_dict(), features=feats,
                    seed=args.seed, variant=args.variant, hyperparameters=runner.HP,
                    cutoff="2020-09-10", calendar_days=30, horizon=3), checkpoint)
    pred = extras["preds"]
    pred["issue_date"] = pred.target_date - pd.Timedelta(days=3)
    pred["label_source"] = np.where(pred.label_is_hobo == 1, "sensor", "discharge_dry")
    pred.to_csv(runner.OUT / f"preds_q65_s{args.seed}.csv", index=False)
    print(f"COMPLETE {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
