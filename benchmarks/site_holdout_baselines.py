"""Site-holdout baseline harness — LR / XGBoost / LSTM on the RGCN's 5-reach split.

Executes context/SITE_HOLDOUT_BASELINES_PLAN.md: trains the paper's baseline
models on the 17 non-holdout HOBO reaches and evaluates them at t+3 on the
same 5 held-out reaches used by the RGCN's with-sensor site holdout, then
assembles one comparison table including the RGCN stride-1 rows.

Feature construction replicates lr/lr.ipynb + lstm/lstm_hobo_sites.ipynb with
two deliberate hygiene fixes (documented in the plan): the scaler is fit on
training sites only, and binary features are not z-scored.

Run:  CUDA_VISIBLE_DEVICES=<n> uv run python benchmarks/site_holdout_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
from hja.data import (BINARY_COLS, DROP_COLS, DRY_THRESHOLD,  # noqa: F401
                      OBS_EXTRA_COLS, build_allsites_frame, build_hobo_frame,
                      feature_frame, scale_train_only)
from hja.evaluation import metrics  # noqa: F401
from hja.models.lstm import SEQ_LEN, LSTMModel, make_sequences  # noqa: F401

SEED = 42  # overridden by --seed
HOLDOUT = [55000900097170, 55000900100137, 55000900099610,
           55000900235848, 55000900271029]

# Back-compat aliases (this module's original names; logic now lives in hja.data)
build_central_df = build_hobo_frame
build_central_df_allsites = build_allsites_frame


def rgcn_stride1_dir(seed: int) -> Path:
    tag = "consistph_strict_no7_sh" + ("" if seed == 42 else f"_s{seed}")
    return REPO / f"data/retrain/predictions_{tag}_stride1"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def run_tabular(name, model, tr, te, feats):
    X_tr, X_te = scale_train_only(tr[0], te[0], feats)
    X_res, y_res = ADASYN(random_state=SEED).fit_resample(X_tr, tr[1].astype(int))
    model.fit(X_res, y_res)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def run_lstm(train_df, test_df, feats, device, seed,
             hidden=64, layers=2, dropout=0.3, lr=1e-4, batch=32, epochs=15):
    tr_scaled, te_scaled = scale_train_only(train_df[feats], test_df[feats], feats)
    train_df = train_df.assign(**{c: tr_scaled[c].values for c in feats})
    test_df = test_df.assign(**{c: te_scaled[c].values for c in feats})
    # Sites with no driver coverage keep NaN through the left-join; the released
    # notebook zero-fills AFTER normalization — replicate that here.
    train_df[feats] = train_df[feats].fillna(0.0)
    test_df[feats] = test_df[feats].fillna(0.0)

    X_tr, y_tr, _ = make_sequences(train_df, feats)
    X_te, y_te, s_te = make_sequences(test_df, feats)
    print(f"  LSTM sequences: {len(X_tr)} train / {len(X_te)} held-out")

    n, T, d = X_tr.shape
    X_res, y_res = ADASYN(random_state=seed).fit_resample(
        X_tr.reshape(n, T * d), y_tr.astype(int))
    X_res = X_res.reshape(-1, T, d).astype(np.float32)

    idx = np.random.default_rng(seed).permutation(len(X_res))
    cut = int(0.8 * len(idx))
    tr_i, va_i = idx[:cut], idx[cut:]
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(to_t(X_res[tr_i]),
                                            to_t(y_res[tr_i]).reshape(-1, 1)),
                              batch_size=batch, shuffle=True)
    Xv = to_t(X_res[va_i]).to(device)
    yv = to_t(y_res[va_i]).reshape(-1, 1).to(device)

    model = LSTMModel(d, hidden, layers, dropout).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        print(f"  LSTM epoch {epoch+1:2d}/{epochs} val={vl:.4f}")
        if vl < best:
            best, best_state, patience = vl, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_te), 4096):
            probs.append(torch.sigmoid(
                model(to_t(X_te[i:i + 4096]).to(device))).cpu().numpy().ravel())
    prob = np.concatenate(probs)
    return y_te, prob, (prob >= 0.5).astype(int), s_te


def rgcn_rows(seed):
    """Uniform-metric RGCN rows from the (per-seed) stride-1 t+3 export."""
    df = pd.read_csv(rgcn_stride1_dir(seed) / "train_val_predictions_day3.csv",
                     usecols=["site_id", "date", "has_true_label", "true_wetdry",
                              "pred_wetdry_prob"], parse_dates=["date"])
    df = df[df["has_true_label"] & df["site_id"].isin(HOLDOUT)]
    df = (df.groupby(["site_id", "date"], as_index=False)
            .agg(true_wetdry=("true_wetdry", "first"),
                 pred_wetdry_prob=("pred_wetdry_prob", "mean")))
    y = df["true_wetdry"].round().astype(int).to_numpy()
    prob = df["pred_wetdry_prob"].to_numpy()
    return df, y, prob, (prob >= 0.5).astype(int)


def run_seed(seed: int, with_rgcn: bool):
    """Train + score every baseline for one seed; return (results, per_site)."""
    global SEED
    SEED = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed={seed}")

    df = build_central_df()
    train_df = df[~df["NHDPlusID"].isin(HOLDOUT)].copy()
    test_df = df[df["NHDPlusID"].isin(HOLDOUT)].copy()
    print(f"central_df: {len(df)} rows | train sites "
          f"{train_df['NHDPlusID'].nunique()} ({len(train_df)} rows) | "
          f"holdout sites {test_df['NHDPlusID'].nunique()} ({len(test_df)} rows)")

    X_all, feats = feature_frame(df)
    tr = (X_all.loc[train_df.index], train_df["wet_dry_next"])
    te = (X_all.loc[test_df.index], test_df["wet_dry_next"])

    results, per_site = {}, {}

    def record(name, y, prob, pred, sites):
        results[name] = metrics(y, prob, pred)
        per_site[name] = {sid: metrics(y[sites == sid], prob[sites == sid],
                                       pred[sites == sid]) for sid in HOLDOUT}

    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=3000, random_state=seed)),
        ("XGBoost", XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,
                                  random_state=seed, eval_metric="logloss")),
    ]:
        print(f"\n=== {name} ===")
        prob, pred = run_tabular(name, model, tr, te, feats)
        record(name, te[1].astype(int).to_numpy(), prob, pred,
               test_df["NHDPlusID"].values)

    print("\n=== LSTM (HOBO only) ===")
    aug = lambda base: pd.concat(
        [base, X_all.loc[base.index].drop(
            columns=[c for c in X_all.columns if c in base.columns])], axis=1)
    y_l, prob_l, pred_l, sites_l = run_lstm(aug(train_df), aug(test_df),
                                            feats, device, seed)
    record("LSTM (HOBO only)", y_l, prob_l, pred_l, sites_l)

    print("\n=== LSTM (all sites) ===")
    dfa = build_central_df_allsites()
    tra = dfa[~dfa["NHDPlusID"].isin(HOLDOUT)].copy()
    tea = dfa[dfa["NHDPlusID"].isin(HOLDOUT)].copy()
    Xa, feats_a = feature_frame(dfa)
    print(f"all-sites: {len(dfa)} rows | train {len(tra)} | holdout {len(tea)}")
    aug_a = lambda base: pd.concat(
        [base, Xa.loc[base.index].drop(
            columns=[c for c in Xa.columns if c in base.columns])], axis=1)
    # Optuna-selected released hyperparameters (lstm_all_sites.ipynb)
    y_a, prob_a, pred_a, sites_a = run_lstm(aug_a(tra), aug_a(tea), feats_a, device,
                                            seed, hidden=57, layers=2, dropout=0.15,
                                            lr=0.008, batch=64)
    record("LSTM (all sites)", y_a, prob_a, pred_a, sites_a)

    if with_rgcn:
        print("\n=== RGCN (strict_no7_sh, stride-1 t+3) ===")
        rdf, y_r, prob_r, pred_r = rgcn_rows(seed)
        record("RGCN (strict, with-sensor holdout)", y_r, prob_r, pred_r,
               rdf["site_id"].values)

    return results, per_site


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rgcn", action="store_true",
                    help="Skip RGCN rows (e.g. its per-seed export not ready yet).")
    ap.add_argument("--aggregate", default=None,
                    help="Comma-separated seeds: read their JSONs and write the "
                         "multi-seed mean±std table instead of training.")
    args = ap.parse_args()

    out_json = REPO / "results" / "site_holdout_seeds"
    out_json.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        seeds = [int(s) for s in args.aggregate.split(",")]
        runs = [json.loads((out_json / f"seed{s}.json").read_text()) for s in seeds]
        models = list(runs[0]["results"].keys())
        lines = ["# Site-holdout comparison — multi-seed (mean ± std)", ""]
        lines.append(f"Seeds: {seeds}. Same split/protocol as "
                     "results/site_holdout_comparison.md.")
        lines.append("")
        lines.append("| Model | N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for m in models:
            vals = {k: [r["results"][m][k] for r in runs if m in r["results"]]
                    for k in ("N", "Accuracy", "ROC-AUC", "WetF1", "DryF1", "DryRecall")}
            def ms(k):
                a = np.array(vals[k], dtype=float)
                return f"{a.mean():.3f} ± {a.std():.3f}"
            lines.append(f"| {m} | {int(np.mean(vals['N']))} | {ms('Accuracy')} | "
                         f"{ms('ROC-AUC')} | {ms('WetF1')} | {ms('DryF1')} | "
                         f"{ms('DryRecall')} |")
        out = REPO / "results/site_holdout_comparison_multiseed.md"
        out.write_text("\n".join(lines))
        print("\n".join(lines))
        print(f"\nWrote {out}")
        return 0

    results, per_site = run_seed(args.seed, with_rgcn=not args.no_rgcn)
    (out_json / f"seed{args.seed}.json").write_text(json.dumps(
        {"seed": args.seed, "results": results, "per_site": per_site}, indent=1))
    print(f"\nWrote {out_json / f'seed{args.seed}.json'}")

    # ---- report ----
    lines = ["# Site-holdout comparison — all models, same 5 held-out reaches", ""]
    lines.append("Split: train = 17 HOBO reaches, test = the 5 reaches held out of "
                 "RGCN training (2 intermittent + 3 perennial; see "
                 "context/SITE_HOLDOUT_BASELINES_PLAN.md). Task: wet/dry at t+3. "
                 "All models receive the held-out site's own lagged status as an "
                 "input (with-sensor regime). ROC-AUC from probabilities. "
                 "LR/XGB/LSTM: scaler fit on train sites only; binaries unscaled; "
                 "ADASYN train-only; released hyperparameters.")
    lines.append("")
    lines.append("| Model | N | Wet frac | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, m in results.items():
        lines.append(f"| {name} | {m['N']:,} | {m['WetFrac']:.2f} | "
                     f"{m['Accuracy']:.3f} | {m['ROC-AUC']:.3f} | {m['WetF1']:.3f} | "
                     f"{m['DryF1']:.3f} | {m['DryRecall']:.3f} |")
    lines.append("")
    lines.append("## Per-site accuracy / dry recall")
    lines.append("")
    header = "| Site (dry frac) | " + " | ".join(results.keys()) + " |"
    lines.append(header)
    lines.append("|---|" + "--:|" * len(results))
    site_label = {55000900097170: "097170 (0.40)", 55000900100137: "100137 (0.68)",
                  55000900099610: "099610 (0.00)", 55000900235848: "235848 (0.00)",
                  55000900271029: "271029 (0.00)"}
    for sid in HOLDOUT:
        cells = []
        for name in results:
            m = per_site[name][sid]
            dr = f" / {m['DryRecall']:.2f}" if not np.isnan(m["DryRecall"]) else ""
            cells.append(f"{m['Accuracy']:.3f}{dr}")
        lines.append(f"| {site_label[sid]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Caveats: RGCN full-graph training sees held-out sites' input "
                 "streams as unsupervised context (labels never in loss); baseline "
                 "held-out rows are absent from training entirely. LSTM N is lower "
                 "(30-day history requirement). RGCN N counts (site, date) pairs "
                 "with a t+3 prediction from the stride-1 export.")

    suffix = "" if args.seed == 42 else f"_s{args.seed}"
    out = REPO / f"results/site_holdout_comparison{suffix}.md"
    out.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
