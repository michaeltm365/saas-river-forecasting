"""Flagship analysis pass: persistence baseline, matched RGCN-vs-LSTM
head-to-head, and the Gaussian-copula annual dry-day rerun.

Inputs (must exist first):
  - stride-3 exports  data/retrain/flagship/predictions_flag_{ph,q65,q80}/
  - stride-1 exports  data/retrain/flagship/predictions_flag_{ph,q65,q80,sh}*_stride1/
  - LSTM predictions  results/flagship/lstm_all/preds_{split}_s{seed}{,_noad}.csv

Outputs:
  results/flagship/persistence_baseline.md
  results/flagship/matched_headtohead.md
  results/flagship/copula_dryday.md

Persistence is horizon-matched: the day-h forecast for date d uses the last
observed status as of d-h (forward-fill only; never information after d-h).

Run:  uv run python benchmarks/flagship_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/flagship"
PRED = REPO / "data/retrain/flagship"
LSTM = OUT / "lstm_all"

HOLDOUT = [55000900097170, 55000900100137, 55000900099610,
           55000900235848, 55000900271029]
BLOCKS = [("2020-07-18", "2020-07-29"), ("2020-09-10", "2020-09-21"),
          ("2020-10-10", "2020-10-21")]
GUARD_DAYS = 3
CUTOFFS = {"q65": "2020-09-10", "q80": "2020-09-28"}
SEASON = ("2020-06-16", "2020-10-29")
DRY_THRESHOLD = 0.00014
SEEDS = (42, 43, 44)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def in_blocks(dates: pd.Series, pad_days: int = 0) -> pd.Series:
    pad = pd.Timedelta(days=pad_days)
    mask = pd.Series(False, index=dates.index)
    for s, e in BLOCKS:
        mask |= (dates >= pd.Timestamp(s) - pad) & (dates <= pd.Timestamp(e) + pad)
    return mask


def val_mask(split: str, dates: pd.Series, sites: pd.Series) -> pd.Series:
    if split == "ph":
        return in_blocks(dates)
    if split in CUTOFFS:
        return dates > pd.Timestamp(CUTOFFS[split])
    if split == "site":
        return sites.isin(HOLDOUT) & (dates >= pd.Timestamp(SEASON[0])) \
            & (dates <= pd.Timestamp(SEASON[1]))
    raise ValueError(split)


def build_status_frame() -> pd.DataFrame:
    """Daily observed wet/dry status per reach (HOBO preferred, else
    discharge-discretized), forward-filled — status 'as known' on each day."""
    obs = pd.read_csv(REPO / "data/sciencebase/obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["NHDPlusID"] = obs["NHDPlusID"].astype("int64")
    obs["dis"] = ((obs["Discharge_CMS"] >= DRY_THRESHOLD).astype(float)
                  .where(obs["Discharge_CMS"].notna()))
    g = (obs.groupby(["NHDPlusID", "Date"])
            .agg(hobo=("HoboWetDry0.05", "mean"), dis=("dis", "mean")))
    status = g["hobo"].round().fillna(g["dis"].round()).dropna()
    wide = status.unstack("NHDPlusID")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full).ffill()


STATUS = build_status_frame()


def persistence_pred(sites: np.ndarray, dates: pd.DatetimeIndex, h: int):
    """Status as of (date - h) per row; NaN where no observation exists yet."""
    out = np.full(len(sites), np.nan)
    lag_dates = dates - pd.Timedelta(days=h)
    for sid in np.unique(sites):
        if sid not in STATUS.columns:
            continue
        m = sites == sid
        out[m] = STATUS[sid].reindex(lag_dates[m]).to_numpy()
    return out


def metr(y, pred, prob=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {
        "N": len(y),
        "Acc": accuracy_score(y, pred) if len(y) else float("nan"),
        "AUC": (roc_auc_score(y, prob) if prob is not None
                and len(np.unique(y)) > 1 else float("nan")),
        "WetF1": f1_score(y, pred, pos_label=1, zero_division=0),
        "DryF1": f1_score(y, pred, pos_label=0, zero_division=0),
        "DryRec": (float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1))
                   if (y == 0).any() else float("nan")),
    }
    return out


def agg(rows: list[dict]) -> dict:
    keys = ("Acc", "AUC", "WetF1", "DryF1", "DryRec")
    out = {"N": rows[0]["N"]}
    for k in keys:
        a = np.array([r[k] for r in rows], dtype=float)
        out[k] = (f"{np.nanmean(a):.3f}" if len(rows) == 1
                  else f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}")
    return out


def fmt_row(label, m):
    n = m["N"]
    return (f"| {label} | {n} | {m['Acc']} | {m['AUC']} | {m['WetF1']} | "
            f"{m['DryF1']} | {m['DryRec']} |")


def stride1_dir(split: str, seed: int) -> Path:
    tag = {"site": "flag_sh"}.get(split, f"flag_{split}")
    if seed != 42:
        tag += f"_s{seed}"
    return PRED / f"predictions_{tag}_stride1"


def load_rgcn_stride1(split: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(stride1_dir(split, seed) / "train_val_predictions_day3.csv",
                     usecols=["date", "site_id", "true_wetdry",
                              "pred_wetdry_prob", "has_true_label"],
                     parse_dates=["date"])
    df = df[df["has_true_label"]].copy()
    df = df[val_mask(split, df["date"], df["site_id"])]
    # one row per (site, date) at stride 1; average defensively if dup
    return (df.groupby(["site_id", "date"], as_index=False)
              .agg(true_wetdry=("true_wetdry", "first"),
                   pred_wetdry_prob=("pred_wetdry_prob", "mean")))


# --------------------------------------------------------------------------- #
# Part 1 — persistence baseline on each split's standard RGCN val set
# --------------------------------------------------------------------------- #
def part_persistence() -> None:
    lines = ["# Persistence baseline on the flagship splits", "",
             "Horizon-matched persistence: the day-h forecast for date d is the "
             "last observed status (HOBO preferred, else discretized discharge) "
             "as of d-h, forward-filled. Scored on the exact rows of the "
             "flagship RGCN's stride-3 val exports (has_true_label & val "
             "dates). RGCN rows are seed 42; flagship multi-seed means are in "
             "the eval reports.", ""]
    for split in ("ph", "q65", "q80"):
        lines += [f"## {split}", "",
                  "| Model / horizon | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        pooled = {"y": [], "pp": [], "rp": [], "rl": []}
        for h in (1, 2, 3):
            df = pd.read_csv(
                PRED / f"predictions_flag_{split}/train_val_predictions_day{h}.csv",
                usecols=["date", "site_id", "true_wetdry", "pred_wetdry_prob",
                         "has_true_label"], parse_dates=["date"])
            df = df[df["has_true_label"]].copy()
            df = df[val_mask(split, df["date"], df["site_id"])]
            y = df["true_wetdry"].round().astype(int).to_numpy()
            pp = persistence_pred(df["site_id"].to_numpy(),
                                  pd.DatetimeIndex(df["date"]), h)
            ok = ~np.isnan(pp)
            if (~ok).any():
                print(f"[persist {split} d{h}] {(~ok).sum()} rows lack "
                      f"persistence input; dropped from both models")
            y, pp = y[ok], pp[ok].astype(int)
            rp = df["pred_wetdry_prob"].to_numpy()[ok]
            rl = (rp >= 0.5).astype(int)
            lines.append(fmt_row(f"Persistence d{h}", agg([metr(y, pp)])))
            lines.append(fmt_row(f"RGCN d{h} (s42)", agg([metr(y, rl, rp)])))
            pooled["y"].append(y); pooled["pp"].append(pp)
            pooled["rp"].append(rp); pooled["rl"].append(rl)
        y = np.concatenate(pooled["y"])
        lines.append(fmt_row("Persistence pooled",
                             agg([metr(y, np.concatenate(pooled["pp"]))])))
        lines.append(fmt_row("RGCN pooled (s42)",
                             agg([metr(y, np.concatenate(pooled["rl"]),
                                       np.concatenate(pooled["rp"]))])))
        lines.append("")
    (OUT / "persistence_baseline.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT/'persistence_baseline.md'}")


# --------------------------------------------------------------------------- #
# Part 2 — matched-set head-to-head at t+3 (RGCN vs LSTM variants vs persistence)
# --------------------------------------------------------------------------- #
def part_matched() -> None:
    lines = ["# Matched-set head-to-head at t+3 — flagship RGCN vs LSTM (all sites) vs persistence", "",
             "All models scored on the identical (reach, date) label rows: the "
             "inner join of the RGCN stride-1 day-3 export (val region) and the "
             "LSTM's per-sequence predictions, seeds paired 42/43/44 "
             "(mean ± std). Persistence uses only the status as of d-3. "
             "'site' = the 5 with-sensor holdout reaches over the full labeled "
             "season (RGCN = flag_sh, labels masked from its loss).", ""]
    for split in ("ph", "q65", "q80", "site"):
        rows = {m: [] for m in ("RGCN (flagship)", "LSTM (ADASYN)",
                                "LSTM (no ADASYN)", "Persistence")}
        hobo_rows = {m: [] for m in rows}
        n_matched = n_rgcn = n_lstm = 0
        for seed in SEEDS:
            r = load_rgcn_stride1(split, seed)
            lstm_a = pd.read_csv(LSTM / f"preds_{split}_s{seed}.csv",
                                 parse_dates=["target_date"])
            lstm_n = pd.read_csv(LSTM / f"preds_{split}_s{seed}_noad.csv",
                                 parse_dates=["target_date"])
            for lab, ldf in (("a", lstm_a), ("n", lstm_n)):
                ldf.rename(columns={"target_date": "date",
                                    "pred_wetdry_prob": f"lstm_{lab}"},
                           inplace=True)
            m = r.merge(lstm_a[["site_id", "date", "true_wetdry", "lstm_a",
                                "label_is_hobo"]],
                        on=["site_id", "date"], suffixes=("", "_l"))
            m = m.merge(lstm_n[["site_id", "date", "lstm_n"]],
                        on=["site_id", "date"])
            m = m[m["true_wetdry"].round() == m["true_wetdry_l"].round()]
            pp = persistence_pred(m["site_id"].to_numpy(),
                                  pd.DatetimeIndex(m["date"]), 3)
            m = m[~np.isnan(pp)]
            pp = pp[~np.isnan(pp)]
            if seed == 42:
                n_matched, n_rgcn, n_lstm = len(m), len(r), len(lstm_a)
            y = m["true_wetdry"].round().astype(int).to_numpy()
            hb = m["label_is_hobo"].to_numpy() > 0.5

            def add(dst, yy, ppv, sub):
                dst["RGCN (flagship)"].append(
                    metr(yy, (sub["pred_wetdry_prob"] >= 0.5).astype(int),
                         sub["pred_wetdry_prob"]))
                dst["LSTM (ADASYN)"].append(
                    metr(yy, (sub["lstm_a"] >= 0.5).astype(int), sub["lstm_a"]))
                dst["LSTM (no ADASYN)"].append(
                    metr(yy, (sub["lstm_n"] >= 0.5).astype(int), sub["lstm_n"]))
                dst["Persistence"].append(metr(yy, ppv.astype(int)))

            add(rows, y, pp, m)
            if hb.any() and (~hb).any():
                add(hobo_rows, y[hb], pp[hb], m[hb])
        lines += [f"## {split}  (matched N={n_matched}; RGCN val rows "
                  f"{n_rgcn}, LSTM val rows {n_lstm})", "",
                  "| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for name, rs in rows.items():
            lines.append(fmt_row(name, agg(rs)))
        if hobo_rows["Persistence"]:
            lines += ["", "HOBO-labeled rows only:", "",
                      "| Model | N | Acc | AUC | Wet F1 | Dry F1 | Dry recall |",
                      "|---|--:|--:|--:|--:|--:|--:|"]
            for name, rs in hobo_rows.items():
                lines.append(fmt_row(name, agg(rs)))
        lines.append("")
    (OUT / "matched_headtohead.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT/'matched_headtohead.md'}")


# --------------------------------------------------------------------------- #
# Part 3 — Gaussian copula annual dry-day counts (flagship ph, stride-1, s42)
# --------------------------------------------------------------------------- #
def part_copula() -> None:
    n_sims, horizon_days = 10_000, 365
    rng = np.random.default_rng(42)
    df = pd.read_csv(stride1_dir("ph", 42) / "train_val_predictions_day3.csv",
                     usecols=["date", "site_id", "true_wetdry",
                              "pred_wetdry_prob", "has_true_label"],
                     parse_dates=["date"])
    df = df[df["has_true_label"]].copy()
    # HOBO sensor reaches only (the paper's Table 9 population)
    obs = pd.read_csv(REPO / "data/sciencebase/obs.csv")
    hobo_ids = set(obs.loc[obs["HoboWetDry0.05"].notna(), "NHDPlusID"]
                   .astype("int64"))
    df = df[df["site_id"].isin(hobo_ids)]
    val = df[in_blocks(df["date"])]
    train = df[~in_blocks(df["date"], GUARD_DAYS)]

    top8 = (val.groupby("site_id").size().sort_values(ascending=False)
            .head(8).index.tolist())
    rows = []
    for site in top8:
        sv = val[val["site_id"] == site]
        st = (train[train["site_id"] == site].sort_values("date"))
        # TRUE lag-1 autocorrelation: stride-1 rows are daily; only pair
        # consecutive calendar days (guards/blocks make gaps).
        s = st["true_wetdry"].round().to_numpy()
        consec = (st["date"].diff().dt.days.to_numpy()[1:] == 1)
        a, b = s[:-1][consec], s[1:][consec]
        rho = (float(np.corrcoef(a, b)[0, 1])
               if len(a) >= 3 and a.std() > 0 and b.std() > 0 else 0.0)

        p_dry = 1.0 - sv["pred_wetdry_prob"].to_numpy()
        innov = rng.standard_normal((n_sims, horizon_days)) * np.sqrt(
            max(1 - rho ** 2, 0.0))
        z = np.empty((n_sims, horizon_days))
        z[:, 0] = rng.standard_normal(n_sims)
        for t in range(1, horizon_days):
            z[:, t] = rho * z[:, t - 1] + innov[:, t]
        u = norm.cdf(z)
        sampled = rng.choice(p_dry, size=(n_sims, horizon_days), replace=True)
        counts = (u < sampled).sum(axis=1)
        lo, hi = np.percentile(counts, [2.5, 97.5])
        true_dry = (1.0 - sv["true_wetdry"].round().mean()) * horizon_days
        rows.append({
            "site": site, "n_val": len(sv), "rho": rho,
            "true": true_dry, "mean": counts.mean(),
            "lo": lo, "hi": hi, "in_ci": bool(lo <= true_dry <= hi),
        })
    rows.sort(key=lambda r: -r["true"])
    cov = sum(r["in_ci"] for r in rows)
    lines = ["# Annual dry-day estimation — flagship RGCN, Gaussian copula (AR1)", "",
             "Flagship (A-strict no-lag-7) phases predictions, STRIDE-1 day-3 "
             "export (true daily grid), seed 42. Fixes the released "
             "ρ-estimation bug: ρ is now the genuine lag-1 autocorrelation "
             "(consecutive calendar days only; the stride-3 export computed "
             "lag-3 mislabeled as lag-1). p_dry bootstrapped from the 36 "
             "val-block days (3 hydrologic phases); observed dry days/yr = "
             "(1 - val wet fraction) x 365. 10,000 sims/site, top-8 HOBO "
             "reaches by val label count.", "",
             "| Site | N val | ρ (lag-1) | True dry d/yr | Mean pred | 95% CI | In CI |",
             "|---|--:|--:|--:|--:|---|:--:|"]
    for r in rows:
        lines.append(f"| {r['site']} | {r['n_val']} | {r['rho']:.3f} | "
                     f"{r['true']:.1f} | {r['mean']:.1f} | "
                     f"[{r['lo']:.1f}, {r['hi']:.1f}] | "
                     f"{'yes' if r['in_ci'] else 'NO'} |")
    lines += ["", f"Coverage: {cov}/{len(rows)} sites "
              f"({cov/len(rows):.0%}) within the 95% interval.",
              "", "Caveat: the val set is three 12-day blocks, so the "
              "bootstrap treats 36 days spanning drying/peak-dry/rewetting as "
              "a typical year; the wet-fraction x 365 'observed' value shares "
              "the same approximation."]
    (OUT / "copula_dryday.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT/'copula_dryday.md'}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    part_persistence()
    part_matched()
    part_copula()
    return 0


if __name__ == "__main__":
    sys.exit(main())
