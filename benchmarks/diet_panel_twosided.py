"""[Promoted from session scratchpad 2026-09-08 — results in
results/flagship/diet_panels.md. NOTE: diet_panel_onesided retrains
the canonical _1s LSTM outputs when run.]"""
"""5-way matched comparison on the two-sided diet, q65 split, t+3:
RGCN-2s, LSTM-all (existing preds), LR-all, XGB-all (trained here),
persistence — identical (reach, date) rows; canonical one-sided RGCN as a
reference row. All-rows + real-HOBO panels, full per-class metrics."""
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hja.data import scale_train_only
from lstm_flagship_splits import prepare_frame
from flagship_analysis import persistence_pred  # builds STATUS at import

CUT = pd.Timestamp("2020-09-10")
SEEDS = (42, 43, 44)
ROOT = str(Path(__file__).resolve().parents[1])

# ---------- all-sites frame (the LSTM/tabular two-sided diet) ----------
dfa, feats = prepare_frame()
dfa["NHDPlusID"] = pd.to_numeric(dfa["NHDPlusID"]).astype("int64")
tr = dfa["target_date"] <= CUT
va = dfa["target_date"] > CUT
print(f"tabular: {int(tr.sum()):,} train / {int(va.sum()):,} val rows")
y_tr = dfa.loc[tr, "wet_dry_next"].astype(int).to_numpy()
val_keys = dfa.loc[va, ["NHDPlusID", "target_date", "wet_dry_next",
                        "label_is_hobo"]].copy()

# ---------- train LR-all / XGB-all per seed ----------
tab_probs = {"LR (all-sites)": {}, "XGBoost (all-sites)": {}}
Xtr_s, Xall_s = scale_train_only(dfa.loc[tr, feats], dfa[feats], feats)
# Released protocol zero-fills AFTER normalization (see lstm_flagship_splits).
Xtr_s = Xtr_s.fillna(0.0)
Xva_s = Xall_s[va.to_numpy()].fillna(0.0)
Xtr_r = dfa.loc[tr, feats].fillna(0.0)
Xva_r = dfa.loc[va, feats].fillna(0.0)
for seed in SEEDS:
    Xr, yr = ADASYN(random_state=seed).fit_resample(Xtr_s, y_tr)
    lr = LogisticRegression(max_iter=3000, random_state=seed).fit(Xr, yr)
    tab_probs["LR (all-sites)"][seed] = lr.predict_proba(Xva_s)[:, 1]
    print(f"LR seed {seed} done", flush=True)
    Xr, yr = ADASYN(random_state=seed).fit_resample(Xtr_r, y_tr)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                        random_state=seed).fit(Xr, yr)
    tab_probs["XGBoost (all-sites)"][seed] = xgb.predict_proba(Xva_r)[:, 1]
    print(f"XGB seed {seed} done", flush=True)

tab = val_keys.rename(columns={"NHDPlusID": "site_id",
                               "target_date": "date"}).reset_index(drop=True)
for name, per_seed in tab_probs.items():
    for seed, p in per_seed.items():
        tab[f"{name}|{seed}"] = p

# ---------- RGCN exports (2s + canonical one-sided reference) ----------
def rgcn_preds(prefix):
    out = None
    for seed, tag in [(42, ""), (43, "_s43"), (44, "_s44")]:
        d = pd.read_csv(f"{ROOT}/data/retrain/flagship/{prefix}{tag}_stride1/"
                        "train_val_predictions_day3.csv",
                        usecols=["date", "site_id", "pred_wetdry_prob"],
                        parse_dates=["date"])
        d["site_id"] = d["site_id"].astype("int64")
        d = d[d["date"] > CUT].rename(
            columns={"pred_wetdry_prob": f"{prefix}|{seed}"})
        out = d if out is None else out.merge(d, on=["site_id", "date"])
    return out

r2s = rgcn_preds("predictions_flag_q65_2s")
r1s = rgcn_preds("predictions_flag_q65")

# ---------- LSTM-all preds ----------
lstm = None
for seed in SEEDS:
    d = pd.read_csv(f"{ROOT}/results/flagship/lstm_all/preds_q65_s{seed}.csv",
                    usecols=["site_id", "target_date", "pred_wetdry_prob"],
                    parse_dates=["target_date"])
    d["site_id"] = pd.to_numeric(d["site_id"]).astype("int64")
    d = d.rename(columns={"target_date": "date",
                          "pred_wetdry_prob": f"LSTM (all-sites)|{seed}"})
    lstm = d if lstm is None else lstm.merge(d, on=["site_id", "date"])

# ---------- matched join ----------
m = (tab.merge(lstm, on=["site_id", "date"])
        .merge(r2s, on=["site_id", "date"])
        .merge(r1s, on=["site_id", "date"]))
m["y"] = m["wet_dry_next"].round().astype(int)
pp = persistence_pred(m["site_id"].to_numpy(), m["date"].to_numpy(), 3)
ok = ~np.isnan(pp)
m = m[ok].copy()
m["Persistence|det"] = pp[ok]
print(f"\nmatched rows: {len(m):,} "
      f"(real HOBO {int(m['label_is_hobo'].sum()):,}, "
      f"wet {int(m['y'].sum()):,} / dry {int((m['y']==0).sum()):,})")

# ---------- metrics ----------
from sklearn.metrics import accuracy_score, roc_auc_score

def perclass(y, prob):
    lab = (np.asarray(prob) >= 0.5).astype(int)
    out = {"Acc": accuracy_score(y, lab),
           "AUC": roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan}
    for cls, nm in [(1, "Wet"), (0, "Dry")]:
        tp = ((lab == cls) & (y == cls)).sum()
        pr = tp / max((lab == cls).sum(), 1)
        rc = tp / max((y == cls).sum(), 1)
        out[f"{nm}P"], out[f"{nm}R"] = pr, rc
        out[f"{nm}F1"] = 2 * pr * rc / max(pr + rc, 1e-12)
    return out

MODELS = ["predictions_flag_q65_2s", "predictions_flag_q65",
          "LSTM (all-sites)", "LR (all-sites)", "XGBoost (all-sites)",
          "Persistence"]
LABEL = {"predictions_flag_q65_2s": "RGCN (two-sided)",
         "predictions_flag_q65": "RGCN (canonical one-sided)"}

for panel, mask in [("ALL MATCHED ROWS", np.ones(len(m), bool)),
                    ("REAL-HOBO ROWS ONLY", m["label_is_hobo"].to_numpy() > 0.5)]:
    sub = m[mask]
    print(f"\n=== {panel} (N={len(sub):,}) ===")
    hdr = f"{'Model':28s}" + "".join(f"{k:>16s}" for k in
          ["Acc", "AUC", "WetP", "WetR", "WetF1", "DryP", "DryR", "DryF1"])
    print(hdr)
    for mod in MODELS:
        cols = [c for c in m.columns if c.startswith(mod + "|")]
        rows = [perclass(sub["y"].to_numpy(), sub[c].to_numpy()) for c in cols]
        cells = []
        for k in ["Acc", "AUC", "WetP", "WetR", "WetF1", "DryP", "DryR", "DryF1"]:
            a = np.array([r[k] for r in rows], dtype=float)
            cells.append(f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}"
                         if len(rows) > 1 else f"{a[0]:.3f}")
        print(f"{LABEL.get(mod, mod):28s}" + "".join(f"{c:>16s}" for c in cells))
