# Flagship RGCN campaign — consolidated results (Sep 2026)

**Status: these are the candidate paper-revision results**, produced after the
2026-09-01 readjustment ("keep a subset of flagship RGCN results, ignore the
earlier site-based comparison work"). Everything predating this campaign is
branch history (`context/RGCN_RETRAIN_TRAJECTORY.md`, kept on the
`rgcn-retrain` branch only) and is superseded here.
Multi-seed = seeds 42/43/44 (mean ± std). Single-seed rows are seed 42.

## 1. The flagship model and protocol

RGCN **A-strict, no 7-day lags**:

- 30-day windows, stride 3, horizons t+1..t+3; date_start 1980-01-02.
- **Forecast-tail masking `obs+drivers`**: lag-1 obs features and MaxDepth are
  frozen at day t for the t+1..t+3 tail, AND tail GridMET drivers are frozen
  at day t (persistence weather). No post-t information of any kind — the
  deployment-honest information set, matching the LSTM's by construction.
- **35 input features** = 18 time-varying (11 drivers, lag-1 discharge +
  lag-1 wet/dry, 3 MaxDepth, month + day) + 17 statics. Lag-7 discharge and
  lag-7 wet/dry are excluded (`features.exclude_time`).
- Loss: multitask (λ_reg 1.0, λ_cls 0.5), weighted BCE with 2× dry penalty
  (fpw 2). hidden 64, dropout 0.1, lr 1e-3, wd 1e-4, batch 128, patience 20.
- Configs: `rgcn/flagship/config_{ph,q65,q80,sh}[_s43|_s44].yml`;
  ablations `rgcn/flagship/ablations/`.

## 2. Splits (all rebuilt for this campaign)

| Split | Definition | Guard | Val wet/dry labels |
|---|---|---|--:|
| ph | 3×12-day blocks: Jul 18–29 (drying), Sep 10–21 (peak dry), Oct 10–21 (rewetting) | 3 d (= max lag) | 784 |
| q65 | temporal cutoff 2020-09-10 (0.65 label quantile) | none needed (train strictly precedes val) | 968 |
| q80 | temporal cutoff 2020-09-28 (0.80) | none | 576 |
| site | ph backbone + 5-reach with-sensor holdout (labels NaN'd from loss; obs-lag inputs kept): 097170 (0.40 dry), 100137 (0.68), 099610, 235848, 271029 | 3 d | 614 masked labels |

The ph guard dropped from 7→3 days because the deepest remaining lag feature
is lag-1 (grid tiling makes guards 1–3 equivalent). Cutoff splits contain no
training window with any val-period date in its inputs (structurally clean on
the train-sees-val channel); ph retains a small residual (post-block training
windows include block dates among input days) documented for §2.4.

## 3. RGCN headline numbers (val, horizons pooled, Acc)

| Split | s42 / s43 / s44 | Mean ± std | AUC (s42) |
|---|---|---|--:|
| ph | 0.957 / 0.962 / 0.969 | **0.963 ± 0.005** | 0.985 |
| q65 | 0.950 / 0.968 / 0.969 | 0.962 ± 0.009 | 0.982 |
| q80 | 0.946 / 0.948 / 0.951 | 0.948 ± 0.002 | 0.972 |

Horizon profile is flat-to-rising (ph s42: 0.951 / 0.958 / 0.961 d1/d2/d3) —
no leaky-tail artifact. Discharge under strict masking (ph s42): NSE 0.860
(d1) → 0.757 (d3) — the honest "no weather forecast" floor.
Reports: `results/flagship/rgcn_eval_flag_*.md`.

## 4. Spatial transfer (with-sensor site holdout, flag_sh)

Held-out 5 reaches, pooled over the labeled season, all horizons
(`results/rgcn_eval_siteholdout_flag_sh*_stride1.md`):

| Seed | Acc | AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|
| 42 | 0.958 | 0.984 | 0.973 | 0.897 |
| 43 | 0.959 | 0.987 | 0.974 | 0.946 |
| 44 | 0.916 | 0.965 | 0.946 | 0.844 |
| **mean** | **0.944 ± 0.020** | 0.979 | 0.964 | 0.896 |

Transfer to spatially new reaches with an observation stream holds
(weakest site: 100137, the 68%-dry reach). Seed variance is larger here than
on temporal splits.

## 5. LSTM (all sites) on the same splits

`benchmarks/lstm_flagship_splits.py` — released Optuna hyperparams, splits
assigned by sequence target date (same blocks/guards/cutoffs); site split
drops the 5 reaches from training (same with-sensor regime). Two variants:
released protocol (ADASYN, train-only) and no-ADASYN.
(`results/flagship/lstm_all/lstm_summary{,_noad}.md`)

Key finding: **no-ADASYN is better nearly everywhere on the mixed all-sites
diet** — ph all-rows 0.934 ± 0.009 vs 0.908 ± 0.007, with better dry F1
(0.770 vs 0.718) and better q65 AUC (0.938 vs 0.870); only the site split
favors ADASYN (0.959 vs 0.938). The discretized-dry supply appears to make
synthetic oversampling unnecessary-to-harmful; a §2.2.5 protocol note is
needed whichever variant becomes the headline row.

## 6. Matched-set head-to-head at t+3 (the clean comparison table)

Identical (reach, date) rows for all models: inner join of RGCN stride-1
day-3 export and LSTM predictions, seeds paired
(`results/flagship/matched_headtohead.md`; HOBO-only sub-tables in the
file). Accuracy:

| Split | RGCN | LSTM (ADASYN) | LSTM (no ADASYN) |
|---|---|---|---|
| ph (N=651) | **0.958 ± 0.004** | 0.904 | 0.932 |
| q65 (N=928) | **0.961 ± 0.010** | 0.926 | 0.941 |
| q80 (N=537) | **0.942 ± 0.005** | 0.919 | 0.910 |
| site (N=449) | 0.928 ± 0.034 | **0.959** | 0.938 |

Dry-side (the paper's declared evaluation focus) is the stronger story —
ph: RGCN dry F1 0.879 / dry recall 0.949, ahead of both LSTM variants; same
pattern on q65/q80. The RGCN also wins AUC on every temporal split; the
site split favors the ADASYN-LSTM.

## 7. Persistence baseline

Held out of the main branch for now — the horizon-matched persistence
analysis lives on the `rgcn-retrain` branch
(`results/flagship/persistence_baseline.md` and the full sections 6–7
there).

## 8. Ablations + hyperparameter sensitivity (flagship, ph, seed 42)

`results/flagship/rgcn_ablation_sweep.md`, 16 rows. Noise band ±0.02
(multi-seed flagship 0.963 ± 0.005):

- Defaults justified: 13/14 hyperparameter rows within ~±0.01 of 0.957.
- **Multitask synergy refuted again on the flagship** (both directions):
  wet/dry-only 0.959 ≈ default 0.957; discharge-only NSE d3 0.776 ≥ 0.757.
  Honest claim = consolidation (one model, both products, no cost).
- **fpw is a monotone dry-recall dial**: 0.884 / 0.907 / 0.953 at fpw 1/2/4,
  accuracy flat — the loss-side equivalent of ADASYN, quantified.
- **No-statics: Acc 0.953 vs 0.957 (noise); discharge NSE d3 0.715 vs 0.757.**
  Statics are worth ~nothing for wet/dry and a little for discharge —
  the empirical resolution of the §3.4 narrative conflict.

## 9. Copula annual dry-day estimation (canonical: calibrated q65)

`results/flagship/copula_dryday_allsplits.md` (+ superseded ph-only
`copula_dryday.md`); shared implementation `src/hja/copula.py`. ρ fixed to
genuine lag-1 (consecutive-day pairs on the raw daily HOBO series over the
split's training dates — the released notebook's stride-3 rows computed
lag-3 mislabeled as lag-1). Each split scored with ITS OWN trained model
(seed 42), stride-1 day-3 export; per-site simulation seeds.

**Canonical (q65, adopted Sep 2026): Platt calibration + ρ ceiling 0.98,
scored on all 22 HOBO reaches with ≥20 val labels — coverage 18/22 (82%),
mean 95% CI width 27.9 days.** The calibrator is fit on the q65 model's own
training-period Day-3 predictions (579 HOBO-labeled stride-3 rows). The raw
method on the same 22 reaches covers 6/22 (27%) with 58.9-day intervals:

- The gain is almost entirely **probability calibration**: the honest
  model's mean p_dry at perennial reaches is ~4.5%, integrating to 6–25
  spurious dry days/yr; Platt drops it below 1% and all but one perennial
  reach (167692) is covered. (The released 7/8 was partly an artifact of the
  defective model's overconfidence.)
- The **ρ ceiling** keeps ρ ≈ 1.0 reaches' AR(1) intervals informative
  (e.g. 097170: raw CI [0, 365] → [85, 254]).
- Remaining misses are informative, not systematic: 097170 (true 349.8 vs
  [85, 254]) and 167704 (77.2 vs [43, 70]) are genuine under-predictions of
  dryness at the two most intermittent reaches; 096564 (10.1 vs [1, 8]) is a
  near-miss; 167692 is a site-specific miscalibration.
- Isotonic calibration scores similarly (17/22) but collapses perennial
  intervals to a degenerate [0, 0]; Platt avoids that and is canonical.
- Caveat: single-season 579-row calibration pool — a leave-site-out
  calibration check is the natural robustness follow-up (not run).

Diagnostic (raw method, original site selections): ph 2/8, q80 2/8, site
2/5 — the calibration failure mode is shared by every independently trained
flagship model, and the intermittent-reach successes include the spatially
held-out reaches (site 100137: true 249.7, mean 219.9; 097170: true 147.7,
mean 146.7 — an ungauged-product success).

## 10. Implications queued for the paper revision

1. Abstract/Results: RGCN 0.993 → ~0.95–0.96 on honest splits; N shrinks
   (7,453 → 784/968/576); AUC 0.882 (predicted-labels bug) → 0.97–0.985.
2. §2.3.4: 35 features incl. statics (not 20, not "no statics"), 30-day
   windows, plain PyTorch, tail-masking protocol (new methodological
   subsection). §2.4: the four splits + leakage rationale.
3. §3.4 rewrite around the no-statics ablation; drop "avoids static
   memorization by construction".
4. Table 8 → flagship sweep; §4.1 multitask claim → consolidation claim.
5. (Deferred) Persistence-baseline framing — held on the rgcn-retrain
   branch for now.
6. §3.6: canonical q65 copula = Platt + ρ-clip on all 22 HOBO val reaches,
   18/22 (82%) coverage; report raw 6/22 as the motivating comparison and
   the two intermittent under-predictions as residual limitation.
7. LSTM tables: decide ADASYN (released protocol) vs no-ADASYN (better,
   deviation) headline; matched-set table (§6) as the cross-model anchor.

## 11. Artifact index

- Configs: `rgcn/flagship/` (12 matrix + 15 ablations, generator-produced).
- Checkpoints/exports: `data/retrain/flagship/` (gitignored, regenerable);
  stride-1 day-3 exports per split/seed (`predictions_*_stride1/`).
- Code: `benchmarks/lstm_flagship_splits.py`, `flagship_ablation_eval.py`,
  `flagship_analysis.py`, `flagship_copula_all.py`,
  `flagship_queue_gpu{1,2}.sh`, `flagship_analysis_queue.sh`;
  `features.exclude_static` added to `rgcn/pipeline/{train,export_predictions}.py`.
- Results: this directory + `results/rgcn_eval_siteholdout_flag_sh*_stride1.md`.
- Invalid first-pass LSTM outputs (label_is_hobo feature leak):
  `results/flagship/lstm_all/_invalid_labelhobo_feature/` — do not use.

Single-seed noise on Acc at these val sizes remains ~±0.02; every headline
number above is 3-seed unless marked s42.
