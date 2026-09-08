"""[Promoted from session scratchpad 2026-09-08 — results in
results/flagship/diet_panels.md. NOTE: diet_panel_onesided retrains
the canonical _1s LSTM outputs when run.]"""
"""Label-diet-controlled panel: ALL models trained on the RGCN's one-sided
(dry_only) label diet, q65 split, t+3, NO resampling for LR/XGB/LSTM
(decision 2026-09-07). Scored on identical matched (reach, date) rows next to
the canonical one-sided RGCN and persistence; RGCN-2s shown as reference."""
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hja.data import scale_train_only
from lstm_flagship_splits import prepare_frame, train_eval
from flagship_analysis import persistence_pred

CUT = pd.Timestamp("2020-09-10")
SEEDS = (42, 43, 44)
ROOT = str(Path(__file__).resolve().parents[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dfa, feats = prepare_frame(labels="dry_only")
dfa["NHDPlusID"] = pd.to_numeric(dfa["NHDPlusID"]).astype("int64")
lab = dfa["wet_dry_next"].notna()
tr = (dfa["target_date"] <= CUT) & lab
va = (dfa["target_date"] > CUT) & lab
print(f"one-sided labeled rows: {int(tr.sum()):,} train "
      f"(wet {int((dfa.loc[tr,'wet_dry_next']==1).sum()):,} / "
      f"dry {int((dfa.loc[tr,'wet_dry_next']==0).sum()):,}) / "
      f"{int(va.sum()):,} val", flush=True)

# ---------- LSTM (dry_only, no ADASYN), 3 seeds ----------
for seed in SEEDS:
    train_eval("q65", seed, dfa, feats, device, use_adasyn=False, suffix="_1s")

# ---------- LR / XGB (dry_only labeled rows, no resampling) ----------
y_tr = dfa.loc[tr, "wet_dry_next"].astype(int).to_numpy()
Xtr_s, Xall_s = scale_train_only(dfa.loc[tr, feats], dfa[feats], feats)
Xtr_s = Xtr_s.fillna(0.0)
Xva_s = Xall_s[va.to_numpy()].fillna(0.0)
Xtr_r = dfa.loc[tr, feats].fillna(0.0)
Xva_r = dfa.loc[va, feats].fillna(0.0)
tab = dfa.loc[va, ["NHDPlusID", "target_date", "wet_dry_next",
                   "label_is_hobo"]].rename(
    columns={"NHDPlusID": "site_id", "target_date": "date"}).reset_index(drop=True)
for seed in SEEDS:
    lr = LogisticRegression(max_iter=3000, random_state=seed).fit(Xtr_s, y_tr)
    tab[f"LR (one-sided)|{seed}"] = lr.predict_proba(Xva_s)[:, 1]
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                        random_state=seed).fit(Xtr_r, y_tr)
    tab[f"XGBoost (one-sided)|{seed}"] = xgb.predict_proba(Xva_r)[:, 1]
    print(f"tabular seed {seed} done", flush=True)

# ---------- LSTM preds, RGCN exports ----------
def merge_preds(base, path_fmt, name, datecol="target_date"):
    for seed, tag in [(42, ""), (43, "_s43"), (44, "_s44")]:
        d = pd.read_csv(path_fmt.format(seed=seed, tag=tag),
                        parse_dates=[datecol])
        if "has_true_label" in d.columns:
            pass
        d = d.rename(columns={datecol: "date",
                              "pred_wetdry_prob": f"{name}|{seed}"})
        d["site_id"] = pd.to_numeric(d["site_id"]).astype("int64")
        base = base.merge(d[["site_id", "date", f"{name}|{seed}"]],
                          on=["site_id", "date"])
    return base

m = merge_preds(tab, ROOT + "/results/flagship/lstm_all/preds_q65_s{seed}_1s.csv",
                "LSTM (one-sided)")
m = merge_preds(m, ROOT + "/data/retrain/flagship/predictions_flag_q65{tag}_stride1/"
                "train_val_predictions_day3.csv", "RGCN (canonical one-sided)",
                datecol="date")
m = merge_preds(m, ROOT + "/data/retrain/flagship/predictions_flag_q65_2s{tag}_stride1/"
                "train_val_predictions_day3.csv", "RGCN (two-sided, ref)",
                datecol="date")
m["y"] = m["wet_dry_next"].round().astype(int)
pp = persistence_pred(m["site_id"].to_numpy(), m["date"].to_numpy(), 3)
ok = ~np.isnan(pp)
m = m[ok].copy()
m["Persistence|det"] = pp[ok]
print(f"\nmatched rows: {len(m):,} (real HOBO {int(m['label_is_hobo'].sum()):,}, "
      f"wet {int(m['y'].sum()):,} / dry {int((m['y']==0).sum()):,})")

from sklearn.metrics import accuracy_score, roc_auc_score

def perclass(y, prob):
    labp = (np.asarray(prob) >= 0.5).astype(int)
    out = {"Acc": accuracy_score(y, labp),
           "AUC": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan}
    for cls, nm in [(1, "Wet"), (0, "Dry")]:
        tp = ((labp == cls) & (y == cls)).sum()
        pr = tp / max((labp == cls).sum(), 1)
        rc = tp / max((y == cls).sum(), 1)
        out[f"{nm}P"], out[f"{nm}R"] = pr, rc
        out[f"{nm}F1"] = 2 * pr * rc / max(pr + rc, 1e-12)
    return out

MODELS = ["RGCN (canonical one-sided)", "LSTM (one-sided)", "LR (one-sided)",
          "XGBoost (one-sided)", "Persistence", "RGCN (two-sided, ref)"]
for panel, mask in [("ALL MATCHED ROWS", np.ones(len(m), bool)),
                    ("REAL-HOBO ROWS ONLY", m["label_is_hobo"].to_numpy() > 0.5)]:
    sub = m[mask]
    print(f"\n=== {panel} (N={len(sub):,}) ===")
    print(f"{'Model':28s}" + "".join(f"{k:>16s}" for k in
          ["Acc", "AUC", "WetP", "WetR", "WetF1", "DryP", "DryR", "DryF1"]))
    for mod in MODELS:
        cols = [c for c in m.columns if c.startswith(mod + "|")]
        rows = [perclass(sub["y"].to_numpy(), sub[c].to_numpy()) for c in cols]
        cells = []
        for k in ["Acc", "AUC", "WetP", "WetR", "WetF1", "DryP", "DryR", "DryF1"]:
            a = np.array([r[k] for r in rows], dtype=float)
            cells.append(f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"
                         if len(rows) > 1 else f"{a[0]:.3f}")
        print(f"{mod:28s}" + "".join(f"{c:>16s}" for c in cells))
