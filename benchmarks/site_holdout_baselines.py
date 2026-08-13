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
SEED = 42
SEQ_LEN = 30
HOLDOUT = [55000900097170, 55000900100137, 55000900099610,
           55000900235848, 55000900271029]
RGCN_STRIDE1_DIR = REPO / "data/retrain/predictions_consistph_strict_no7_sh_stride1"
OBS_EXTRA_COLS = ["MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor"]
BINARY_COLS = {"MaxDepth_Threshold", "MaxDepth_Censor", "wetdry_status"}
DROP_COLS = ["NHDPlusID", "SiteIDCode", "Date", "wet_dry_next",
             "StreamOrde", "FCode", "n_discharge", "n_water_presence", "has_data"]

np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# Data (mirrors lr.ipynb cells 5-8 / lstm_hobo_sites.ipynb cells 5-6)
# --------------------------------------------------------------------------- #
def build_central_df() -> pd.DataFrame:
    obs = pd.read_csv(REPO / "data/sciencebase/obs.csv")
    obs["Date"] = pd.to_datetime(obs["Date"])
    hobo = obs[obs["HoboWetDry0.05"].notna()][
        ["NHDPlusID", "SiteIDCode", "Date", "HoboWetDry0.05"]
    ].rename(columns={"HoboWetDry0.05": "wetdry_status"}).copy()
    maxd = obs.loc[obs[OBS_EXTRA_COLS].notna().any(axis=1),
                   ["NHDPlusID", "Date"] + OBS_EXTRA_COLS]

    drivers = pd.read_parquet(REPO / "data/retrain/met_drivers.parquet")
    statics = pd.read_csv(REPO / "data/sciencebase/static_vars.csv")
    degrees = pd.read_parquet(REPO / "data/huggingface/degrees.parquet")
    order = pd.read_csv(REPO / "data/huggingface/nhd_id_stream_order_permanence.csv")

    for df in (hobo, drivers, statics, degrees, order, maxd):
        df["NHDPlusID"] = df["NHDPlusID"].astype("int64")

    df = hobo.merge(drivers, on=["NHDPlusID", "Date"], how="inner")
    df = df.merge(statics, on="NHDPlusID", how="left")
    df = df.merge(degrees, on="NHDPlusID", how="left")
    df = df.merge(order, on="NHDPlusID", how="left")
    df = df.merge(maxd, on=["NHDPlusID", "Date"], how="left")

    df = df.sort_values(["NHDPlusID", "Date"])
    df[OBS_EXTRA_COLS] = (df.groupby("NHDPlusID")[OBS_EXTRA_COLS]
                          .transform(lambda g: g.ffill().bfill()))
    df[OBS_EXTRA_COLS] = df[OBS_EXTRA_COLS].fillna(0)

    df["wet_dry_next"] = df.groupby("NHDPlusID")["wetdry_status"].shift(-3)
    df = df.dropna(subset=["wet_dry_next"])
    return df.reset_index(drop=True)


def feature_frame(df: pd.DataFrame):
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in DROP_COLS]
    X = df[feats].copy().ffill().bfill().fillna(0)
    return X, feats


def scale_train_only(X_tr, X_te, feats):
    cont = [c for c in feats if c not in BINARY_COLS]
    scaler = StandardScaler().fit(X_tr[cont])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[cont] = scaler.transform(X_tr[cont])
    X_te[cont] = scaler.transform(X_te[cont])
    return X_tr, X_te


def metrics(y, prob, pred):
    y = np.asarray(y).astype(int)
    return {
        "N": len(y),
        "WetFrac": float(y.mean()),
        "Accuracy": accuracy_score(y, pred),
        "ROC-AUC": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "WetF1": f1_score(y, pred, pos_label=1, zero_division=0),
        "DryF1": f1_score(y, pred, pos_label=0, zero_division=0),
        "DryRecall": (float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1))
                      if (y == 0).any() else float("nan")),
    }


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(df, feats, seq_len=SEQ_LEN):
    """Per-site sliding windows (mirrors create_sequences_by_site);
    also returns each sequence's site id."""
    X, y, sites = [], [], []
    for sid, g in df.groupby("NHDPlusID"):
        if len(g) <= seq_len:
            continue
        f, lab = g[feats].values, g["wet_dry_next"].values
        for i in range(len(g) - seq_len):
            X.append(f[i:i + seq_len])
            y.append(lab[i + seq_len - 1])
            sites.append(sid)
    return np.array(X, dtype=np.float32), np.array(y), np.array(sites)


def run_lstm(train_df, test_df, feats, device):
    tr_scaled, te_scaled = scale_train_only(train_df[feats], test_df[feats], feats)
    train_df = train_df.assign(**{c: tr_scaled[c].values for c in feats})
    test_df = test_df.assign(**{c: te_scaled[c].values for c in feats})

    X_tr, y_tr, _ = make_sequences(train_df, feats)
    X_te, y_te, s_te = make_sequences(test_df, feats)
    print(f"  LSTM sequences: {len(X_tr)} train / {len(X_te)} held-out")

    n, T, d = X_tr.shape
    X_res, y_res = ADASYN(random_state=SEED).fit_resample(
        X_tr.reshape(n, T * d), y_tr.astype(int))
    X_res = X_res.reshape(-1, T, d)

    idx = np.random.default_rng(SEED).permutation(len(X_res))
    cut = int(0.8 * len(idx))
    tr_i, va_i = idx[:cut], idx[cut:]
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(to_t(X_res[tr_i]),
                                            to_t(y_res[tr_i]).reshape(-1, 1)),
                              batch_size=32, shuffle=True)
    Xv = to_t(X_res[va_i]).to(device)
    yv = to_t(y_res[va_i]).reshape(-1, 1).to(device)

    model = LSTMModel(input_size=d).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        print(f"  LSTM epoch {epoch+1:2d}/15 val={vl:.4f}")
        if vl < best:
            best, best_state, patience = vl, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(to_t(X_te).to(device))).cpu().numpy().ravel()
    return y_te, prob, (prob >= 0.5).astype(int), s_te


def rgcn_rows():
    """Uniform-metric RGCN rows from the stride-1 t+3 export."""
    df = pd.read_csv(RGCN_STRIDE1_DIR / "train_val_predictions_day3.csv",
                     usecols=["site_id", "date", "has_true_label", "true_wetdry",
                              "pred_wetdry_prob"], parse_dates=["date"])
    df = df[df["has_true_label"] & df["site_id"].isin(HOLDOUT)]
    df = (df.groupby(["site_id", "date"], as_index=False)
            .agg(true_wetdry=("true_wetdry", "first"),
                 pred_wetdry_prob=("pred_wetdry_prob", "mean")))
    y = df["true_wetdry"].round().astype(int).to_numpy()
    prob = df["pred_wetdry_prob"].to_numpy()
    return df, y, prob, (prob >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    df = build_central_df()
    train_df = df[~df["NHDPlusID"].isin(HOLDOUT)].copy()
    test_df = df[df["NHDPlusID"].isin(HOLDOUT)].copy()
    print(f"central_df: {len(df)} rows | train sites "
          f"{train_df['NHDPlusID'].nunique()} ({len(train_df)} rows) | "
          f"holdout sites {test_df['NHDPlusID'].nunique()} ({len(test_df)} rows)")

    X_all, feats = feature_frame(df)
    print(f"features ({len(feats)}): {feats}")
    tr = (X_all.loc[train_df.index], train_df["wet_dry_next"])
    te = (X_all.loc[test_df.index], test_df["wet_dry_next"])

    results = {}
    per_site = {}

    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=3000, random_state=SEED)),
        ("XGBoost", XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,
                                  random_state=SEED, eval_metric="logloss")),
    ]:
        print(f"\n=== {name} ===")
        prob, pred = run_tabular(name, model, tr, te, feats)
        y = te[1].astype(int).to_numpy()
        results[name] = metrics(y, prob, pred)
        per_site[name] = {
            sid: metrics(y[test_df["NHDPlusID"].values == sid],
                         prob[test_df["NHDPlusID"].values == sid],
                         pred[test_df["NHDPlusID"].values == sid])
            for sid in HOLDOUT}

    print("\n=== LSTM (HOBO only) ===")
    y_l, prob_l, pred_l, sites_l = run_lstm(
        pd.concat([train_df, X_all.loc[train_df.index].drop(
            columns=[c for c in X_all.columns if c in train_df.columns])], axis=1),
        pd.concat([test_df, X_all.loc[test_df.index].drop(
            columns=[c for c in X_all.columns if c in test_df.columns])], axis=1),
        feats, device)
    results["LSTM (HOBO only)"] = metrics(y_l, prob_l, pred_l)
    per_site["LSTM (HOBO only)"] = {
        sid: metrics(y_l[sites_l == sid], prob_l[sites_l == sid],
                     pred_l[sites_l == sid])
        for sid in HOLDOUT}

    print("\n=== RGCN (strict_no7_sh, stride-1 t+3) ===")
    rdf, y_r, prob_r, pred_r = rgcn_rows()
    results["RGCN (strict, with-sensor holdout)"] = metrics(y_r, prob_r, pred_r)
    per_site["RGCN (strict, with-sensor holdout)"] = {
        sid: metrics(y_r[rdf["site_id"].values == sid],
                     prob_r[rdf["site_id"].values == sid],
                     pred_r[rdf["site_id"].values == sid])
        for sid in HOLDOUT}

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

    out = REPO / "results/site_holdout_comparison.md"
    out.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
