# RGCN retrain trajectory — consistency variants onward (Aug 2026)

**Status: branch-history document.** Everything below is tentative / side
experimentation on branch `rgcn-retrain`, recorded so the reasoning trail
survives. It is NOT the paper's results set and is not intended to be merged
to `main`. Numbers are single-seed 42 unless marked otherwise; measured
single-seed noise on accuracy at these val sizes is ~±0.02.

## 0. Starting point (the consistency-variant comparison)

| Model | Split | Protocol | Acc | AUC | F1 |
|---|---|---|--:|--:|--:|
| retrain (unmasked) | phases | 28d, leaky tail | 0.953 | 0.982 | 0.972 |
| consistph (variant A) | phases | 30d, obs-masked tail | 0.945 | 0.977 | 0.967 |
| consistph_strict | phases | + tail drivers frozen | 0.946 | 0.975 | 0.968 |
| consistph_nolag | phases | + obs-lags removed | 0.529 | 0.624 | 0.647 |
| retrain (unmasked) | q80 | 28d, leaky tail | 0.950 | 0.986 | 0.966 |
| consist80 (variant A) | q80 | 30d, obs-masked tail | 0.951 | 0.982 | 0.968 |

HJFlp ungauged-site transfer (unobserved reaches, n=251): consistph AUC
0.519, no-lag AUC 0.530 — both chance.

Established at this point: honest forecast-tail masking costs ≤0.8 pts
(zero on q80) and flattens the horizon profile; future met is worth ~nothing
for wet/dry but a lot for discharge (NSE 0.93→0.71–0.82 when tail drivers
freeze); removing obs-lags collapses temporal skill AND leaves ungauged AUC
at chance (dry-default bias from zero-filled `HoboWetDry_lag_1=0` ≡ "dry
yesterday").

## 1. Re-framing spatial transfer (user correction, confirmed)

HJFlp's unobserved tier confounds "spatially new" with "observation-less".
The classical models' site-based split feeds held-out sites their own lagged
status — i.e. "new site WITH a sensor" — a regime the RGCN had never been
tested in. Also noted: even at fully-trained reaches, HJFlp visit agreement
tops out ~0.76 (measurement-convention / within-reach-heterogeneity ceiling).

## 2. With-sensor site holdout (the mechanism + result)

Mechanism (`site_holdout:` config, commit f8b587f): held-out reaches stay in
the graph and keep their obs-lag INPUTS, but their labels are NaN'd out of the
train/val loss. Scorer: `eval_holdout_sites.py`. Holdout = 5 reaches
stratified by dry-fraction — 097170 (0.40 dry), 100137 (0.68), 099610,
235848, 271029 (perennial) — 614 labels (~23%) masked; all 5 are HOBO reaches
with only 2–3 spot discharge obs (ungauged in discharge terms).

Result: strict_no7_sh pooled held-out acc 0.951 / AUC 0.989 / dry recall
0.885 (val-block-only 0.955) — matching seen-site performance. The RGCN DOES
transfer to spatially new sites given an observation stream. The failure is
confined to no-history sites.

Related singles: no-7d-lag runs scored 0.925/0.927 temporal (vs 0.945/0.946
with lag-7) but their sh twins hit 0.945/0.949 with FEWER training labels →
the "lag-7 cost" is inside single-seed noise; this run pair is where the
±0.02 noise figure was measured. Classification-tuned no-lag (λ_cls=1.0)
recovered temporal to 0.721/0.798 (the 0.529 collapse was partly
objective-tuning artifact) but HJFlp stayed at chance (0.539).

## 3. Shared 5-site holdout across all models

`benchmarks/site_holdout_baselines.py`: LR/XGB/LSTMs trained on the 17
non-holdout HOBO reaches (train-only scaler, binaries unscaled, ADASYN
train-only, released hyperparams), scored at t+3 on the same 5 reaches.
RGCN row via stride-1 eval export (`export_predictions --eval-stride 1`,
gives a true t+3 for every date; stride-3 tiles each date to one horizon).

Multi-seed (42/43/44), mean ± std:

| Model | Acc | AUC | Dry F1 |
|---|--:|--:|--:|
| XGBoost (HOBO diet) | 0.976 ± 0.002 | 0.984 | 0.944 |
| RGCN (strict_no7_sh) | 0.958 ± 0.009 | 0.982 | 0.900 |
| LSTM (all sites) | 0.915 ± 0.033 | 0.971 | 0.864 |
| LR (HOBO diet) | 0.839 ± 0.005 | 0.894 | 0.701 |
| LSTM (HOBO only) | 0.742 ± 0.015 | 0.890 | 0.513 |

LSTM got its first-ever site-based number (draft had none). LSTM-HOBO
collapses at the 68%-dry site (0.229) — static-feature reliance. The
LSTM-HOBO vs LSTM-all pair (+17 pts from the mixed diet) demonstrates the
data-volume story spatially. Temporal multi-seed alongside: variant A
0.945 ± 0.002, A-strict 0.951 ± 0.006 (strict ≥ A; masking cost within
noise of unmasked 0.953).

## 4. Staleness sweep + transition scoring (supremacy hopes refuted)

`benchmarks/staleness_transitions.py`: held-out site's obs inputs aged k days
(neighbors fresh); transition-adjacent (±3d of an observed change) vs stable
scoring. Findings (seed 42):

- XGBoost ≡ naive persistence within 0.4 pts at every k and at transitions.
- Network-rescue hypothesis FAILED: RGCN degrades FASTER than persistence
  under staleness (0.946→0.803 at k=28) — never trained on stale inputs;
  staleness-augmented training is the untried fix.
- LSTM (all sites) is flat under staleness (~0.94 at all k, dry recall ~1.0)
  and best at transitions (0.806 vs RGCN 0.568, persistence 0.545; N≈33).
- Ranking is regime-dependent: fresh+stable → XGB; stale/transitions →
  LSTM-all; RGCN = all-rounder with discharge + network-complete output.

## 5. Loss ablations + hyperparameter sensitivity (phases, consistph protocol)

14 single-factor runs + evaluator `benchmarks/rgcn_ablation_eval.py`
(results/rgcn_ablation_sweep.md):

- Defaults justified: 12/14 within ±0.02; wd=0 wrecks day-3 NSE (0.63),
  wd=1e-3 wrecks classification (0.929) — 1e-4 is a real sweet spot.
- Multitask synergy REFUTED in BOTH directions: wet/dry-only matches default
  classification (0.948 vs 0.945, dry recall 0.876 vs 0.791); discharge-only
  matches/beats default discharge (NSE d3 0.951 vs 0.925). Honest claim =
  consolidation (one model, both products, no cost), not regularization.
  Untrained heads produce garbage (neg. NSE) → heads genuinely share trunk.
- fpw is a monotone dry-recall dial: 0.752/0.791/0.860 at fpw 1/2/4, flat acc.
- dropout 0.3 looked like a winner (0.967) but did not replicate
  (0.957 ± 0.009 across 3 seeds); default retained. Pre-registered
  noise-band rule worked as designed.

## 6. Diet factorial (user's OOD confound — confirmed)

User hypothesis: holdout is in-distribution for HOBO-diet XGB/LR, OOD for
mixed-diet RGCN/LSTM. `benchmarks/tabular_allsites.py` (3 seeds):

- XGBoost (all-sites diet) 0.955 ± 0.003 vs HOBO-diet 0.976 ± 0.002: the
  mixed diet costs XGB ~2 pts — exactly erasing its edge over the RGCN
  (0.958 ± 0.009). Same-diet ⇒ statistical tie on accuracy.
- Diet effect is model-dependent in sign: HURTS XGB (−2), HELPS LSTM (+17).
- Full per-class, same diet: RGCN wins AUC decisively (0.982 vs 0.957),
  ties acc/dry-F1, trades recalls. Vs XGB-HOBO, AUCs tie (0.983/0.982) but
  XGB wins all thresholded metrics.

## 7. Persistence baseline (the sobering closer)

Horizon-matched persistence (day-h forecast uses status(d−h); verified
day-3-only uses only d−3):

| Split | Persistence | RGCN A-strict |
|---|--:|--:|
| phases val (N=774) | 0.970 (dry F1 0.902) | 0.951 ± 0.006 |
| phases val, day-3 only | 0.968 | 0.945 |
| site holdout t+3 | 0.974 | 0.958 ± 0.009 |
| phases transitions (N=57) | 0.596 | 0.49–0.56 (AUC < 0.5) |

Persistence beats every RGCN variant on thresholded accuracy on both splits;
persistence decays only 0.973→0.968 from day-1 to day-3. No model in the
study except LSTM-all (at transitions/staleness) beats it. RGCN's real value
vs persistence: calibrated probabilities (persistence has none — fatal for
the copula product), discharge, multi-horizon, all-793-reach coverage,
tunable threshold. Any revised paper needs a persistence row in every
classification table.

## 8. Flagship pivot (decided, partially executed)

User decision: make A-strict the flagship (deployment framing: no
weather-forecast dependency; wet/dry unaffected, discharge becomes the
"floor" with A-live quoted as the forecast-coupled ceiling). Identified gaps
at time of pivot-out:

- [ ] Feature-set coherence: recommend strict_no7 as THE flagship (spatial
      results already use it); needs strict_no7 temporal seeds 43/44.
- [ ] Rerun ablation suite on the flagship config (was run on A-live).
- [ ] Copula + calibration (§3.6 rerun): strict flagship vs a strict
      cls-only run; USE STRIDE-1 day-3 export — also fixes the released
      notebook's ρ bug (estimate_rho on stride-3 rows computes lag-3
      autocorrelation mislabeled as lag-1 → CIs too narrow).
- [ ] consist80-strict robustness row.
- [ ] (offered, undecided) multimodal-benefit recovery: label-scarcity sweep
      (multitask vs cls-only at 25/50/100% labels), calibration/copula
      comparison, RGCN impute_dry=False modality factorial.

## Key artifacts (all on branch rgcn-retrain)

Results: site_holdout_comparison{,_multiseed,_diet_factorial}.md,
rgcn_temporal_multiseed.md, staleness_transitions.md, rgcn_ablation_sweep.md,
rgcn_eval_consistph*(_s4x).md, rgcn_eval_siteholdout_*.md, rgcn_eval_hjflp_*.md,
site_holdout_seeds/*.json.
Code: rgcn/pipeline/{masking,eval_hjflp,eval_holdout_sites}.py, feature
ablation + site_holdout + eval-stride mechanisms in train/export,
benchmarks/{site_holdout_baselines,staleness_transitions,rgcn_ablation_eval,
tabular_allsites}.py. Configs: rgcn/config_consist*.yml, rgcn/ablations/.
Key commits: 0dba3e5, 78e58b3, db5a117, f8b587f, 87537b5, 3ce416d, ae78b49,
a4b9d1b, d2a9aac, c6e1c9a.
