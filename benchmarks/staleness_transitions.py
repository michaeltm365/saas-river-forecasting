"""Staleness sweep + transition scoring on the shared 5-site holdout (seed 42).

Two analyses that probe what network/meteorology information adds beyond
persistence, using the models from the site-holdout comparison:

1. TRANSITIONS: score each model's fresh t+3 predictions separately on
   "transition-adjacent" target dates (within +-3 days of an observed wet/dry
   status change at that site) vs "stable" dates. Persistence-style models are
   structurally blind at transitions.

2. STALENESS: degrade the HELD-OUT site's own observation-derived inputs so
   the latest available observation is k days old (k in 0,3,7,14,21,28), while
   neighbors' observations stay fresh. Models were trained with fresh inputs;
   this measures deployment robustness to realistic sensor-visit schedules.
   The RGCN can propagate neighboring reaches' fresh observations through the
   graph; tabular/sequence baselines only see the site's own stale stream.

Models: naive persistence, LR, XGBoost, LSTM (HOBO only), LSTM (all sites),
RGCN (consistph_strict_no7_sh checkpoint, obs+drivers tail masking).

Run:  CUDA_VISIBLE_DEVICES=<n> uv run python benchmarks/staleness_transitions.py
"""

from __future__ import annotations

import pickle
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
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO))

from site_holdout_baselines import (  # noqa: E402
    BINARY_COLS, HOLDOUT, SEQ_LEN, LSTMModel, build_central_df,
    build_central_df_allsites, feature_frame)
from rgcn.pipeline import features as F  # noqa: E402
from rgcn.pipeline.config import load_config  # noqa: E402
from rgcn.pipeline.data import load_arrays  # noqa: E402
from rgcn.pipeline.masking import mask_forecast_tail  # noqa: E402
from rgcn.pipeline.model import build_adjacency_matrix, create_model  # noqa: E402
from rgcn.pipeline.windows import (  # noqa: E402
    WindowSpec, build_date_range, generate_windows)

SEED = 42
KS = [0, 3, 7, 14, 21, 28]
OBS_COLS = ["wetdry_status", "MaxDepth_cm", "MaxDepth_Threshold", "MaxDepth_Censor"]
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def metrics(y, prob, pred):
    y = np.asarray(y).astype(int)
    out = {"N": len(y)}
    if len(y) == 0:
        return out
    out["Accuracy"] = accuracy_score(y, pred)
    out["ROC-AUC"] = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    out["DryF1"] = f1_score(y, pred, pos_label=0, zero_division=0)
    out["DryRecall"] = (float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1))
                        if (y == 0).any() else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# Daily observation series per held-out site (for staleness lookups +
# transition detection), built from raw HOBO labels.
# --------------------------------------------------------------------------- #
def daily_series():
    obs = pd.read_csv(REPO / "data/sciencebase/obs.csv",
                      usecols=["NHDPlusID", "Date", "HoboWetDry0.05"])
    obs["Date"] = pd.to_datetime(obs["Date"])
    obs["NHDPlusID"] = obs["NHDPlusID"].astype("int64")
    obs = obs[obs["NHDPlusID"].isin(HOLDOUT) & obs["HoboWetDry0.05"].notna()]
    out = {}
    for sid, g in obs.groupby("NHDPlusID"):
        s = g.set_index("Date")["HoboWetDry0.05"].sort_index()
        s = s[~s.index.duplicated()]
        daily = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        out[sid] = daily.ffill().bfill()
    return out


def transition_dates(series: dict, tol_days=3):
    """Per site: labeled dates within +-tol of an observed status change."""
    adj = {}
    for sid, s in series.items():
        chg = s.index[s.ne(s.shift(1)) & s.shift(1).notna()]
        mask = pd.Series(False, index=s.index)
        for d in chg:
            mask |= (abs((s.index - d).days) <= tol_days)
        adj[sid] = set(s.index[mask])
    n_chg = {sid: int(s.ne(s.shift(1))[s.shift(1).notna()].sum())
             for sid, s in series.items()}
    return adj, n_chg


# --------------------------------------------------------------------------- #
# Baselines: fit once on fresh training data, keep models for stale re-scoring
# --------------------------------------------------------------------------- #
def target_dates(df):
    """Calendar date of each row's positional-shift(-3) target."""
    return df.groupby("NHDPlusID")["Date"].shift(-3)


def fit_tabular(df, X_all, feats):
    tr_idx = df.index[~df["NHDPlusID"].isin(HOLDOUT)]
    cont = [c for c in feats if c not in BINARY_COLS]
    scaler = StandardScaler().fit(X_all.loc[tr_idx, cont])

    def transform(Xf):
        Xf = Xf.copy()
        Xf[cont] = scaler.transform(Xf[cont])
        return Xf

    X_res, y_res = ADASYN(random_state=SEED).fit_resample(
        transform(X_all.loc[tr_idx]), df.loc[tr_idx, "wet_dry_next"].astype(int))
    models = {
        "LR": LogisticRegression(max_iter=3000, random_state=SEED).fit(X_res, y_res),
        "XGBoost": XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100,
                                 random_state=SEED,
                                 eval_metric="logloss").fit(X_res, y_res),
    }
    return models, transform


def fit_lstm(df, X_all, feats, hidden=64, layers=2, dropout=0.3, lr=1e-4,
             batch=32):
    tr_idx = df.index[~df["NHDPlusID"].isin(HOLDOUT)]
    cont = [c for c in feats if c not in BINARY_COLS]
    scaler = StandardScaler().fit(X_all.loc[tr_idx, cont])

    def scaled_frame(idx):
        base = df.loc[idx, ["NHDPlusID", "Date", "wet_dry_next"]].copy()
        Xf = X_all.loc[idx].copy()
        Xf[cont] = scaler.transform(Xf[cont])
        return pd.concat([base, Xf], axis=1).fillna(0.0)

    def sequences(frame):
        X, y, sites, tdates = [], [], [], []
        for sid, g in frame.groupby("NHDPlusID"):
            g = g.sort_values("Date")
            if len(g) <= SEQ_LEN:
                continue
            f = g[feats].values.astype(np.float32)
            lab = g["wet_dry_next"].values
            tdt = g["Date"].shift(-3).values  # positional target date
            for i in range(len(g) - SEQ_LEN):
                j = i + SEQ_LEN - 1
                X.append(f[i:i + SEQ_LEN])
                y.append(lab[j])
                sites.append(sid)
                tdates.append(tdt[j])
        return (np.array(X, dtype=np.float32), np.array(y),
                np.array(sites), pd.to_datetime(pd.Series(tdates)))

    X_tr, y_tr, _, _ = sequences(scaled_frame(tr_idx))
    n, T, d = X_tr.shape
    X_res, y_res = ADASYN(random_state=SEED).fit_resample(
        X_tr.reshape(n, T * d), y_tr.astype(int))
    X_res = X_res.reshape(-1, T, d).astype(np.float32)
    idx = np.random.default_rng(SEED).permutation(len(X_res))
    cut = int(0.8 * len(idx))
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    loader = DataLoader(TensorDataset(to_t(X_res[idx[:cut]]),
                                      to_t(y_res[idx[:cut]]).reshape(-1, 1)),
                        batch_size=batch, shuffle=True)
    Xv = to_t(X_res[idx[cut:]]).to(DEVICE)
    yv = to_t(y_res[idx[cut:]]).reshape(-1, 1).to(DEVICE)

    model = LSTMModel(d, hidden, layers, dropout).to(DEVICE)
    crit, opt = nn.BCEWithLogitsLoss(), None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        print(f"    epoch {epoch+1:2d}/15 val={vl:.4f}", flush=True)
        if vl < best:
            best, best_state, patience = vl, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, scaled_frame, sequences


def lstm_predict(model, X):
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            probs.append(torch.sigmoid(model(
                torch.tensor(X[i:i + 4096], dtype=torch.float32).to(DEVICE)
            )).cpu().numpy().ravel())
    return np.concatenate(probs)


def stale_sequences(X, feats, k):
    """Freeze obs-derived channels so the last k window positions carry the
    value from position SEQ_LEN-1-k (positional staleness approximation)."""
    if k == 0:
        return X
    Xs = X.copy()
    p = max(SEQ_LEN - 1 - k, 0)
    for c in OBS_COLS:
        if c in feats:
            j = feats.index(c)
            Xs[:, p + 1:, j] = Xs[:, p:p + 1, j]
    return Xs


# --------------------------------------------------------------------------- #
# RGCN: batched stride-1 inference with per-site staleness freeze
# --------------------------------------------------------------------------- #
def rgcn_predict_all_k(ks):
    config = load_config(REPO / "rgcn/config_consistph_strict_no7_sh.yml")
    arr = load_arrays(config)
    X_time = torch.from_numpy(arr["X_time"]).to(DEVICE)
    X_static = torch.from_numpy(arr["X_static"]).to(DEVICE)
    y_all = arr["y_all"]
    node_ids = arr["node_ids"].tolist()
    hold_idx = torch.tensor([node_ids.index(s) for s in HOLDOUT], device=DEVICE)
    date_range = build_date_range(config)

    spec = WindowSpec.from_config(config)
    spec1 = WindowSpec(spec.seq_length, spec.forecast_horizon, 1)
    windows = generate_windows(len(date_range), spec1)
    lo, hi = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-11-15")
    win = [(s, e) for s, e in windows if lo <= date_range[e - 1] <= hi]
    seq_len = spec.seq_length

    graph = pickle.load(open(config.path("graph_out"), "rb"))
    adj = build_adjacency_matrix(graph, node_ids)
    ckpt = torch.load(config.path("checkpoint"), map_location=DEVICE,
                      weights_only=False)
    exclude = ckpt.get("exclude_time", [])
    keep_cols, feature_vars = F.time_feature_selection(exclude)
    keep_idx = torch.tensor(keep_cols, device=DEVICE)
    model = create_model(config, adj, len(feature_vars), DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    lag1_cols = [F.feature_index("Discharge_CMS_lag_1"),
                 F.feature_index("HoboWetDry0.05_lag_1")]
    maxd_cols = [F.feature_index(v)
                 for v in F.MAXDEPTH_BINARY_VARS + F.MAXDEPTH_CONT_VARS]

    out = {}
    with torch.no_grad():
        for k in ks:
            rows = []
            for i in range(0, len(win), 64):
                chunk = win[i:i + 64]
                starts = torch.tensor([s for s, _ in chunk], device=DEVICE)
                t_idx = starts[:, None] + torch.arange(
                    seq_len + spec.forecast_horizon, device=DEVICE)[None, :]
                xt = X_time[t_idx].clone()               # (B, wl, N, 20)
                if k > 0:
                    p = seq_len - 1 - k                  # position of day t-k
                    # lag-1 at position p+1 references obs(t-k): last legit
                    # value. MaxDepth is same-day, so its anchor is p itself.
                    for cols, anchor in ((lag1_cols, p + 1), (maxd_cols, p)):
                        for ni in hold_idx.tolist():
                            for c in cols:
                                xt[:, anchor + 1:, ni, c] = xt[:, anchor:anchor + 1, ni, c]
                xt = mask_forecast_tail(xt, seq_len, "obs+drivers")
                xt = xt[..., keep_idx]
                xs = X_static[None, None].expand(xt.shape[0], xt.shape[1],
                                                 X_static.shape[0],
                                                 X_static.shape[1])
                Xb = torch.cat([xt, xs], dim=-1).permute(0, 2, 1, 3)
                pred = model(Xb)                          # (B, N, wl, 2)
                d3 = pred[:, hold_idx, -1, F.WETDRY_IDX].cpu().numpy()
                for bi, (s, e) in enumerate(chunk):
                    tgt = date_range[e - 1]
                    for si, sid in enumerate(HOLDOUT):
                        ni = node_ids.index(sid)
                        yv = y_all[e - 1, ni, F.WETDRY_IDX]
                        if not np.isnan(yv):
                            rows.append((sid, tgt, float(yv), float(d3[bi, si])))
            df = pd.DataFrame(rows, columns=["site", "date", "y", "prob"])
            df = df.groupby(["site", "date"], as_index=False).mean()
            out[k] = df
            print(f"  RGCN k={k}: {len(df)} scored", flush=True)
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    print(f"Device: {DEVICE}", flush=True)
    series = daily_series()
    adj, n_chg = transition_dates(series)
    print("transitions per site:", n_chg, flush=True)

    # ---------------- baselines ----------------
    df = build_central_df()
    X_all, feats = feature_frame(df)
    te_idx = df.index[df["NHDPlusID"].isin(HOLDOUT)]
    tgt = target_dates(df)

    # Per-site daily frames of obs-derived feature columns (for stale lookups).
    daily_obs = {}
    for sid in HOLDOUT:
        g = df[df["NHDPlusID"] == sid].set_index("Date").sort_index()
        d = g[OBS_COLS][~g.index.duplicated()]
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
        daily_obs[sid] = d.ffill().bfill()

    def stale_lookup(sid, dates, k):
        d = daily_obs[sid]
        want = pd.Series(pd.to_datetime(list(dates))) - pd.Timedelta(days=k)
        want = want.clip(lower=d.index.min(), upper=d.index.max())
        return d.reindex(pd.DatetimeIndex(want)).to_numpy()

    print("\nfitting LR/XGBoost...", flush=True)
    tab_models, tab_transform = fit_tabular(df, X_all, feats)

    print("fitting LSTM (HOBO only)...", flush=True)
    lstm_h, frame_h, seq_h = fit_lstm(df, X_all, feats)

    print("fitting LSTM (all sites)...", flush=True)
    dfa = build_central_df_allsites()
    Xa_all, feats_a = feature_frame(dfa)
    lstm_a, frame_a, seq_a = fit_lstm(dfa, Xa_all, feats_a, hidden=57, layers=2,
                                      dropout=0.15, lr=0.008, batch=64)

    # Pre-build held-out sequences once per LSTM (freeze applied per k).
    te_a_idx = dfa.index[dfa["NHDPlusID"].isin(HOLDOUT)]
    Xh, yh, sh, th = seq_h(frame_h(te_idx))
    Xa, ya, sa, ta = seq_a(frame_a(te_a_idx))

    print("\nRGCN staleness inference...", flush=True)
    rgcn = rgcn_predict_all_k(KS)

    # ---------------- assemble per-model / per-k prediction frames ----------
    def preds_frame(sites, dates, y, prob):
        return pd.DataFrame({"site": sites, "date": pd.to_datetime(dates),
                             "y": np.asarray(y).astype(int),
                             "prob": prob,
                             "pred": (np.asarray(prob) >= 0.5).astype(int)})

    all_preds = {}   # (model, k) -> frame
    base_sites = df.loc[te_idx, "NHDPlusID"].to_numpy()
    base_dates = tgt.loc[te_idx].to_numpy()
    base_y = df.loc[te_idx, "wet_dry_next"].to_numpy()
    ok = ~pd.isna(base_dates)

    for k in KS:
        # persistence: predict status(t-k)
        pers = np.empty(len(te_idx))
        for sid in HOLDOUT:
            m = base_sites == sid
            pers[m] = stale_lookup(sid, df.loc[te_idx[m], "Date"], k)[:, 0]
        all_preds[("Persistence", k)] = preds_frame(
            base_sites[ok], base_dates[ok], base_y[ok], pers[ok])

        # LR / XGBoost with stale obs features
        Xk = X_all.loc[te_idx].copy()
        for sid in HOLDOUT:
            m = base_sites == sid
            stale_vals = stale_lookup(sid, df.loc[te_idx[m], "Date"], k)
            for ci, c in enumerate(OBS_COLS):
                if c in Xk.columns:
                    Xk.loc[te_idx[m], c] = stale_vals[:, ci]
        Xk_t = tab_transform(Xk)
        for name, mdl in tab_models.items():
            prob = mdl.predict_proba(Xk_t)[:, 1]
            all_preds[(name, k)] = preds_frame(
                base_sites[ok], base_dates[ok], base_y[ok], prob[ok])

        # LSTMs with frozen window channels
        okh = (~pd.isna(th)).to_numpy()
        prob = lstm_predict(lstm_h, stale_sequences(Xh, feats, k))
        all_preds[("LSTM (HOBO only)", k)] = preds_frame(
            sh[okh], th[okh].to_numpy(), yh[okh], prob[okh])
        oka = (~pd.isna(ta)).to_numpy()
        prob = lstm_predict(lstm_a, stale_sequences(Xa, feats_a, k))
        all_preds[("LSTM (all sites)", k)] = preds_frame(
            sa[oka], ta[oka].to_numpy(), ya[oka], prob[oka])

        r = rgcn[k]
        all_preds[("RGCN", k)] = preds_frame(
            r["site"], r["date"], r["y"].round(), r["prob"])
        print(f"assembled k={k}", flush=True)

    MODELS = ["Persistence", "LR", "XGBoost", "LSTM (HOBO only)",
              "LSTM (all sites)", "RGCN"]

    # ---------------- report ----------------
    lines = ["# Staleness sweep + transition scoring (shared 5-site holdout, seed 42)",
             ""]
    lines.append("Models trained on fresh inputs (site-holdout protocol of "
                 "results/site_holdout_comparison.md); at evaluation the HELD-OUT "
                 "site's observation-derived inputs (wet/dry status, MaxDepth) "
                 "are aged so the latest available observation is k days old. "
                 "Neighboring sites' observations stay fresh — only the RGCN can "
                 "exploit them (graph propagation). Persistence = predict the "
                 "latest available status. Task: wet/dry at t+3.")
    lines.append("")
    lines.append("## Staleness sweep — pooled accuracy (dry recall)")
    lines.append("")
    lines.append("| Model | " + " | ".join(f"k={k}" for k in KS) + " |")
    lines.append("|---|" + "--:|" * len(KS))
    for m in MODELS:
        cells = []
        for k in KS:
            f = all_preds[(m, k)]
            mt = metrics(f["y"], f["prob"], f["pred"])
            dr = f" ({mt['DryRecall']:.2f})" if not np.isnan(mt["DryRecall"]) else ""
            cells.append(f"{mt['Accuracy']:.3f}{dr}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    n0 = {m: all_preds[(m, 0)].shape[0] for m in MODELS}
    lines.append("N per model: " + ", ".join(f"{m}={n}" for m, n in n0.items()) + ".")
    lines.append("")

    lines.append("## Transition scoring (fresh inputs, k=0)")
    lines.append("")
    lines.append(f"Observed status changes per site: "
                 f"{ {str(k)[-6:]: v for k, v in n_chg.items()} }. "
                 "'Transition' = target dates within ±3 days of a change; "
                 "'Stable' = all other labeled dates.")
    lines.append("")
    lines.append("| Model | N trans | Acc (trans) | Dry recall (trans) | "
                 "N stable | Acc (stable) |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for m in MODELS:
        f = all_preds[(m, 0)].copy()
        is_tr = f.apply(lambda r: r["date"] in adj.get(r["site"], set()), axis=1)
        mt_t = metrics(f.loc[is_tr, "y"], f.loc[is_tr, "prob"], f.loc[is_tr, "pred"])
        mt_s = metrics(f.loc[~is_tr, "y"], f.loc[~is_tr, "prob"], f.loc[~is_tr, "pred"])
        drt = (f"{mt_t['DryRecall']:.3f}"
               if not np.isnan(mt_t.get("DryRecall", np.nan)) else "—")
        lines.append(f"| {m} | {mt_t['N']} | {mt_t.get('Accuracy', float('nan')):.3f} | "
                     f"{drt} | {mt_s['N']} | {mt_s.get('Accuracy', float('nan')):.3f} |")
    lines.append("")
    lines.append("Notes: baselines' target dates use the released notebooks' "
                 "positional shift(-3) (≈3 calendar days on the near-daily HOBO "
                 "series); RGCN uses exact calendar t+3. Staleness for baselines "
                 "is applied per feature row/window; models were not retrained "
                 "on stale inputs (deployment-mismatch test). All single-seed "
                 "(42).")

    out = REPO / "results/staleness_transitions.md"
    out.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
