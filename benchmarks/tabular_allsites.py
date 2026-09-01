"""XGB-all / LR-all: tabular baselines trained on the all-sites (HOBO +
discretized-discharge) diet, evaluated on the shared 5-site holdout at t+3.

Completes the diet x model factorial (LSTM already has both diets): if
XGB-all ~= XGB-HOBO, the "mixed training data makes the holdout OOD for
RGCN/LSTM-all" confound is empirically dismissed; if it drops, the
synchronized-diet table becomes the headline. Seeds 42/43/44.

Run:  uv run python benchmarks/tabular_allsites.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from site_holdout_baselines import (  # noqa: E402
    BINARY_COLS, HOLDOUT, build_central_df_allsites, feature_frame, metrics)

SEEDS = (42, 43, 44)


def main() -> int:
    df = build_central_df_allsites()
    X_all, feats = feature_frame(df)
    tr_idx = df.index[~df["NHDPlusID"].isin(HOLDOUT)]
    te_idx = df.index[df["NHDPlusID"].isin(HOLDOUT)]
    y_tr = df.loc[tr_idx, "wet_dry_next"].astype(int)
    y_te = df.loc[te_idx, "wet_dry_next"].astype(int).to_numpy()
    sites_te = df.loc[te_idx, "NHDPlusID"].to_numpy()
    print(f"all-sites diet: {len(tr_idx):,} train rows "
          f"({df.loc[tr_idx, 'is_hobo'].sum():,} HOBO) | "
          f"{len(te_idx)} holdout rows at {df.loc[te_idx,'NHDPlusID'].nunique()} reaches")

    cont = [c for c in feats if c not in BINARY_COLS]
    per_model = {"LR (all sites)": [], "XGBoost (all sites)": []}
    per_site_last = {}
    for seed in SEEDS:
        scaler = StandardScaler().fit(X_all.loc[tr_idx, cont])
        Xtr = X_all.loc[tr_idx].copy()
        Xte = X_all.loc[te_idx].copy()
        Xtr[cont] = scaler.transform(Xtr[cont])
        Xte[cont] = scaler.transform(Xte[cont])
        X_res, y_res = ADASYN(random_state=seed).fit_resample(Xtr, y_tr)
        for name, mdl in [
            ("LR (all sites)", LogisticRegression(max_iter=3000, random_state=seed)),
            ("XGBoost (all sites)", XGBClassifier(
                max_depth=3, learning_rate=0.1, n_estimators=100,
                random_state=seed, eval_metric="logloss")),
        ]:
            mdl.fit(X_res, y_res)
            prob = mdl.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = metrics(y_te, prob, pred)
            per_model[name].append(m)
            per_site_last[name] = {
                sid: metrics(y_te[sites_te == sid], prob[sites_te == sid],
                             pred[sites_te == sid]) for sid in HOLDOUT}
            print(f"seed {seed} {name}: acc={m['Accuracy']:.3f} "
                  f"auc={m['ROC-AUC']:.3f} dryF1={m['DryF1']:.3f}")

    lines = ["# Diet factorial addendum — LR/XGBoost trained on the all-sites diet", ""]
    lines.append("Same shared 5-reach holdout / t+3 protocol as "
                 "results/site_holdout_comparison_multiseed.md, but LR and "
                 "XGBoost trained on the HOBO + discretized-discharge row set "
                 "(the RGCN / LSTM-all-sites diet) instead of HOBO-only. "
                 "Seeds 42/43/44, mean ± std.")
    lines.append("")
    lines.append("| Model | N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for name, runs in per_model.items():
        def ms(k):
            a = np.array([r[k] for r in runs], dtype=float)
            return f"{a.mean():.3f} ± {a.std():.3f}"
        lines.append(f"| {name} | {runs[0]['N']} | {ms('Accuracy')} | "
                     f"{ms('ROC-AUC')} | {ms('WetF1')} | {ms('DryF1')} | "
                     f"{ms('DryRecall')} |")
    lines.append("")
    lines.append("HOBO-diet reference rows (from the multiseed table): "
                 "LR 0.839 ± 0.005, XGBoost 0.976 ± 0.002.")
    lines.append("")
    lines.append("## Per-site accuracy (seed 44)")
    lines.append("")
    lines.append("| Site | " + " | ".join(per_model.keys()) + " |")
    lines.append("|---|" + "--:|" * len(per_model))
    for sid in HOLDOUT:
        cells = [f"{per_site_last[n][sid]['Accuracy']:.3f}" for n in per_model]
        lines.append(f"| {str(sid)[-6:]} | " + " | ".join(cells) + " |")

    out = REPO / "results/site_holdout_diet_factorial.md"
    out.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
