"""LSTM (all sites) on the four flagship splits: ph / q65 / q80 / site.

Mirrors the released lstm_all_sites protocol (Optuna hyperparams, ADASYN on
train only, train-only scaler, binaries unscaled) but assigns train/val by
each sequence's TARGET date so the splits match the flagship RGCN's:

  ph   — val = target date in the three 12-day phase blocks; sequences whose
         target date falls within guard_days=3 of a block are dropped (buffer)
  q65  — train = target date <= 2020-09-10, val after (no guard; train
         strictly precedes val)
  q80  — as q65 with cutoff 2020-09-28
  site — train = sequences from non-holdout reaches (all dates), val = the
         RGCN's 5 with-sensor holdout reaches (all dates); the LSTM sees the
         held-out sites' own observed history as input at eval time, matching
         the RGCN's with-sensor masking regime

Per (split, seed): per-sequence predictions CSV + metrics JSON under
results/flagship/lstm_all/. Summary table appended to lstm_summary.md.

Run:  CUDA_VISIBLE_DEVICES=<n> uv run python benchmarks/lstm_flagship_splits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from hja.data import (build_allsites_frame as build_central_df_allsites,  # noqa: E402
                      feature_frame, scale_train_only)
from hja.evaluation import metrics  # noqa: E402
from hja.models.lstm import SEQ_LEN, LSTMModel  # noqa: E402
from hja.splits import BLOCKS, CUTOFFS, GUARD_DAYS, in_blocks  # noqa: E402,F401
from site_holdout_baselines import HOLDOUT  # noqa: E402

OUT = REPO / "results/flagship/lstm_all"
SEEDS = (42, 43, 44)
# Optuna-selected released hyperparameters (lstm_all_sites.ipynb)
HP = dict(hidden=57, layers=2, dropout=0.15, lr=0.008, batch=64, epochs=15)


def make_sequences_dated(df: pd.DataFrame, feats: list[str]):
    """Per-site sliding windows; returns (X, y, sites, target_dates, label_is_hobo)."""
    X, y, sites, tdates, hobo = [], [], [], [], []
    for sid, g in df.groupby("NHDPlusID"):
        if len(g) <= SEQ_LEN:
            continue
        f = g[feats].values.astype(np.float32)
        lab = g["wet_dry_next"].values
        td = g["target_date"].values
        lh = g["label_is_hobo"].values
        for i in range(len(g) - SEQ_LEN):
            j = i + SEQ_LEN - 1
            X.append(f[i:i + SEQ_LEN])
            y.append(lab[j])
            sites.append(sid)
            tdates.append(td[j])
            hobo.append(lh[j])
    return (np.array(X, dtype=np.float32), np.array(y),
            np.array(sites), pd.DatetimeIndex(tdates), np.array(hobo))


def assign_split(split: str, tdates: pd.DatetimeIndex, sites: np.ndarray):
    """Returns an array of 'train' / 'val' / 'drop' per sequence."""
    td = pd.Series(tdates)
    if split == "ph":
        out = np.where(in_blocks(td), "val",
                       np.where(in_blocks(td, GUARD_DAYS), "drop", "train"))
    elif split in CUTOFFS:
        out = np.where(td <= pd.Timestamp(CUTOFFS[split]), "train", "val")
    elif split == "site":
        out = np.where(np.isin(sites, HOLDOUT), "val", "train")
    else:
        raise ValueError(split)
    return out


def prepare_frame() -> tuple[pd.DataFrame, list[str]]:
    """Released all-sites frame + per-row target dates / label sources.

    Shared by main() and lstm/lstm_all_sites.ipynb so the notebook runs the
    exact canonical preprocessing.
    """
    dfa = build_central_df_allsites()
    # Label dates: wet_dry_next is a 3-ROW shift within site (released
    # protocol); recover each label's actual date the same way. Rows dropped
    # by the trailing dropna leave the positional shift intact, so only the
    # last 3 rows per site need the calendar fallback.
    dfa["target_date"] = dfa.groupby("NHDPlusID")["Date"].shift(-3)
    dfa["target_date"] = dfa["target_date"].fillna(
        dfa["Date"] + pd.Timedelta(days=3))
    dfa["label_is_hobo"] = (dfa.groupby("NHDPlusID")["is_hobo"].shift(-3)
                            .fillna(0).astype(int))
    X_all, feats = feature_frame(dfa)
    # label_is_hobo is a scoring helper, NOT an input feature (feature_frame
    # picks up any numeric column outside DROP_COLS).
    feats = [c for c in feats if c != "label_is_hobo"]
    dfa = pd.concat([dfa, X_all.drop(columns=[c for c in X_all.columns
                                              if c in dfa.columns])], axis=1)
    print(f"all-sites frame: {len(dfa):,} rows, {len(feats)} features")
    return dfa, feats


def train_eval(split: str, seed: int, dfa: pd.DataFrame, feats: list[str],
               device: torch.device, use_adasyn: bool = True,
               suffix: str = "", extras: dict | None = None) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sequence pool over the raw (unscaled) frame; scaler is fit on the
    # training rows only, then the whole frame is transformed.
    tdates_row = dfa["target_date"]
    if split == "ph":
        row_train = ~in_blocks(tdates_row, GUARD_DAYS)
    elif split in CUTOFFS:
        row_train = tdates_row <= pd.Timestamp(CUTOFFS[split])
    else:
        row_train = ~dfa["NHDPlusID"].isin(HOLDOUT)
    tr_scaled, all_scaled = scale_train_only(
        dfa.loc[row_train, feats], dfa[feats], feats)
    dfs = dfa.assign(**{c: all_scaled[c].values for c in feats})
    # Released notebook zero-fills AFTER normalization — replicate.
    dfs[feats] = dfs[feats].fillna(0.0)

    X, y, sites, tdates, hobo = make_sequences_dated(dfs, feats)
    grp = assign_split(split, tdates, sites)
    tr, va = grp == "train", grp == "val"
    print(f"[{split} s{seed}] sequences: {tr.sum():,} train / {va.sum():,} val "
          f"/ {(grp == 'drop').sum():,} buffer")

    n, T, d = X[tr].shape
    if use_adasyn:
        X_res, y_res = ADASYN(random_state=seed).fit_resample(
            X[tr].reshape(n, T * d), y[tr].astype(int))
        X_res = X_res.reshape(-1, T, d).astype(np.float32)
    else:
        X_res, y_res = X[tr], y[tr].astype(int)
    idx = np.random.default_rng(seed).permutation(len(X_res))
    cut = int(0.8 * len(idx))
    tr_i, va_i = idx[:cut], idx[cut:]
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    loader = DataLoader(TensorDataset(to_t(X_res[tr_i]),
                                      to_t(y_res[tr_i]).reshape(-1, 1)),
                        batch_size=HP["batch"], shuffle=True)
    Xv = to_t(X_res[va_i]).to(device)
    yv = to_t(y_res[va_i]).reshape(-1, 1).to(device)

    model = LSTMModel(d, HP["hidden"], HP["layers"], HP["dropout"]).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(HP["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        print(f"  epoch {epoch+1:2d}/{HP['epochs']} val={vl:.4f}")
        if vl < best:
            best, best_state, patience = vl, model.state_dict(), 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    X_va = X[va]
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_va), 4096):
            probs.append(torch.sigmoid(
                model(to_t(X_va[i:i + 4096]).to(device))).cpu().numpy().ravel())
    prob = np.concatenate(probs)
    pred = (prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "site_id": sites[va], "target_date": tdates[va],
        "true_wetdry": y[va].astype(int), "label_is_hobo": hobo[va].astype(int),
        "pred_wetdry_prob": prob,
    })
    out.to_csv(OUT / f"preds_{split}_s{seed}{suffix}.csv", index=False)

    m = {"all": metrics(y[va].astype(int), prob, pred)}
    hb = hobo[va] > 0.5
    if hb.any():
        m["hobo_only"] = metrics(y[va][hb].astype(int), prob[hb], pred[hb])
    if (~hb).any():
        m["discretized_only"] = metrics(y[va][~hb].astype(int), prob[~hb], pred[~hb])
    (OUT / f"metrics_{split}_s{seed}{suffix}.json").write_text(json.dumps(m, indent=2))
    print(f"[{split} s{seed}] all: acc={m['all']['Accuracy']:.3f} "
          f"auc={m['all']['ROC-AUC']:.3f} dryF1={m['all']['DryF1']:.3f} "
          f"(N={m['all']['N']})")
    if extras is not None:
        extras.update(model=model, preds=out, feats=feats,
                      X_val=X_va, y_val=y[va], hobo_val=hobo[va])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-adasyn", action="store_true",
                    help="Skip ADASYN resampling (train on raw sequences).")
    args = ap.parse_args()
    use_adasyn = not args.no_adasyn
    suffix = "" if use_adasyn else "_noad"

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | ADASYN={'on' if use_adasyn else 'OFF'}")

    dfa, feats = prepare_frame()

    summary = {}
    for split in ("ph", "q65", "q80", "site"):
        for seed in SEEDS:
            summary[(split, seed)] = train_eval(split, seed, dfa, feats, device,
                                                use_adasyn, suffix)

    lines = [f"# LSTM (all sites) on the flagship splits"
             + ("" if use_adasyn else " — NO ADASYN"), "",
             "Optuna released hyperparams; "
             + ("ADASYN + scaler train-only; " if use_adasyn else
                "NO resampling (raw class balance, plain BCE); scaler train-only; ")
             + "splits assigned by sequence target date (ph guard=3d). "
             "Seeds 42/43/44.", "",
             "| Split | Scope | Acc (mean ± std) | AUC | Dry F1 | N |",
             "|---|---|--:|--:|--:|--:|"]
    for split in ("ph", "q65", "q80", "site"):
        for scope in ("all", "hobo_only", "discretized_only"):
            runs = [summary[(split, s)][scope] for s in SEEDS
                    if scope in summary[(split, s)]]
            if not runs:
                continue
            def ms(k):
                a = np.array([r[k] for r in runs], dtype=float)
                return f"{a.mean():.3f} ± {a.std():.3f}"
            lines.append(f"| {split} | {scope} | {ms('Accuracy')} | "
                         f"{ms('ROC-AUC')} | {ms('DryF1')} | {runs[0]['N']} |")
    (OUT / f"lstm_summary{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
