# Plan: same-5-site holdout comparison across all models

Goal: put LR, XGBoost, LSTM, and the RGCN on the **identical** site-based
split so the paper's spatial-transferability table compares like with like
(currently: LR/XGB use a random 80/20 site split, LSTM has no site split at
all, RGCN has the new with-sensor holdout).

## The fixed holdout (do not change — RGCN checkpoints depend on it)

Five HOBO reaches, labels masked from RGCN training (chosen stratified by
dry-fraction; 2 intermittent = the real tests, 3 perennial):

| NHDPlusID | dry frac | HOBO labels | discharge obs |
|---|--:|--:|--:|
| 55000900097170 | 0.40 | 131 | 2 (spot only) |
| 55000900100137 | 0.68 | 114 | 2 |
| 55000900099610 | 0.00 | 116 | 2 |
| 55000900235848 | 0.00 | 126 | 2 |
| 55000900271029 | 0.00 | 126 | 3 |

All five have daily HOBO coverage Jun–Oct 2020 (what LR/XGB/LSTM need) and
no gauge records (comparison is wet/dry only; no discharge scoring here).

## Reference RGCN (already trained + scored)

`rgcn/config_consistph_strict_no7_sh.yml` → `data/retrain/best_model_consistph_strict_no7_sh.pt`
(30-day window, no 7-day lags, obs+drivers tail masking = zero post-t
information, site holdout). Held-out-site results
(`results/rgcn_eval_siteholdout_consistph_strict_no7_sh.md`, and per-class
F1 computed 2026-08-10):

| Slice | N | Acc | AUC | wet F1 | dry F1 | dry recall |
|---|--:|--:|--:|--:|--:|--:|
| pooled (h=1..3) | 613 | 0.951 | 0.989 | 0.969 | 0.885 | 0.885 |
| Day-3 only (t+3) | 201 | 0.960 | 0.992 | 0.975 | 0.907 | 0.907 |

Note on N: with stride 3, each calendar date is forecast at exactly ONE
horizon, so the t+3-only slice covers every third date (201 ≈ 613/3). For
full t+3 coverage of all ~600 site-days, add an eval-only stride-1 export
(generate all stride-1 windows at export time, keep day-3 outputs; no
retraining needed) — optional, see "open items".

## Baseline protocol (one script, do NOT edit released notebooks)

Create `benchmarks/site_holdout_baselines.py` that:

1. Rebuilds each model's `central_df` exactly as its notebook does
   (lr/xgb: HOBO rows + drivers + statics + degrees + MaxDepth ffill;
   LSTM HOBO-only: same but sequenced per site, SEQ_LEN=30).
2. Splits **by site**: train = the 17 other labeled reaches, test = the 5
   holdout reaches (match on NHDPlusID). No random site assignment.
3. Hygiene (differs from released notebooks — intentional fixes):
   - fit StandardScaler on TRAIN sites only (released code fits on all data);
   - do not z-score binary features (MaxDepth flags, wetdry_status) to match
     the RGCN pipeline convention;
   - ADASYN on train only (already the released behavior);
   - target = wetdry_status shifted −3 within site (released behavior).
4. Models, hyperparameters as released: LR (L2, C=1.0), XGBoost
   (max_depth=3, lr=0.1, n_estimators=100), LSTM (hidden 64, 2 layers,
   dropout 0.3, lr 1e-4, BCEWithLogits, early stopping on a train-site
   val subset — never on held-out sites).
5. Score on held-out sites at t+3: Accuracy, ROC-AUC (from probabilities!),
   wet F1, dry F1, dry recall — pooled + per site. Expected N: LR/XGB ~598,
   LSTM ~448 (30-day history requirement), RGCN Day-3 201 (or ~598 with the
   stride-1 export).
6. Write `results/site_holdout_comparison.md` with one table: LR, XGB,
   LSTM (HOBO-only), RGCN (strict_no7_sh Day-3 row + pooled row).

## Fairness caveats to state in the paper

- RGCN full-graph training still sees held-out sites' INPUT streams as
  unsupervised context for neighbors (labels never in any loss); classical
  models' held-out rows are absent from training entirely. Cuts both ways;
  disclose.
- All models receive the held-out site's own lagged status as an input at
  prediction time — this is the "with-sensor" regime by design (matches the
  released site-based-split semantics). It is NOT an ungauged test; that is
  what HJFlp is for (skipped for now; see rgcn_eval_hjflp_*.md).
- Class balance at held-out sites (79% wet overall, but 2 sites ~32–60% wet)
  differs from train; report per-site rows and AUC/F1, not accuracy alone.

## Open items

- [ ] Optional stride-1 eval-only export for the RGCN so its t+3 slice
      covers all ~600 site-days instead of every third one.
- [ ] Optional: LSTM (all sites) variant on the same split.
- [ ] Multi-seed replicates (3 seeds) of RGCN strict_no7_sh and the LSTM —
      single-seed noise at this val size is ~±0.02 (see f8b587f commit msg).
