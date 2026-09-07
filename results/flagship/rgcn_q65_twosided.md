# Two-sided label-imputation diagnostic — q65 RGCN retrain (2026-09-07)

**Question**: the released RGCN's label file was effectively two-sided
(discharge > 0.00014 → wet as well as ≤ → dry); the retrain switched to
dry-only imputation on 2026-07-01. Does the canonical (one-sided) choice
cost anything, and does two-sided explain the released model's dry-class
profile?

**Setup**: `rgcn/flagship/config_q65_2s*.yml` — flagship q65 protocol with
`imputation.wetdry: two_sided` (wet imputed where discharge > threshold and
no real HOBO label exists; fillna semantics matching the LSTM all-sites
frame). Identical cutoff (2020-09-10) and window grid. Training labels grow
~4.5k → 117,938 (97.5% wet / 2.5% dry vs the one-sided ~52/48); val grows
945 → 1,962 canonical Day-3 rows (908 real HOBO — byte-identical to the
canonical rows — plus 1,054 imputed, ~37 dry / ~1,017 wet). Seeds 42/43/44,
Day-3 stride-1 daily grid throughout.

## Results (mean ± std over seeds)

| Scope | N | Acc | AUC | Wet F1 | Dry P | Dry R | Dry F1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Two-sided, real-HOBO rows** | 908 | 0.957 ± 0.003 | 0.971 ± 0.001 | 0.973 ± 0.002 | 0.912 ± 0.003 | 0.880 ± 0.016 | 0.896 ± 0.008 |
| One-sided (canonical), same rows | 908 | 0.962 ± 0.011 | 0.986 ± 0.002 | 0.976 ± 0.007 | 0.878 ± 0.001 | 0.951 ± 0.059 | 0.912 ± 0.028 |
| Two-sided, all rows | 1,962 | 0.951 ± 0.005 | 0.954 ± 0.001 | 0.972 ± 0.003 | 0.802 ± 0.040 | 0.764 ± 0.011 | 0.782 ± 0.014 |
| Two-sided, imputed-only rows | 1,054 | 0.945 ± 0.011 | 0.900 ± 0.008 | 0.971 ± 0.006 | 0.204 ± 0.074 | 0.171 ± 0.046 | 0.181 ± 0.053 |

(One-sided reference from the canonical Day-3 daily-grid computation; its
all-rows panel is N = 945 with a 37-row all-dry imputed subset.)

## Findings

1. **One-sided stays canonical.** On the identical 908 real-label rows,
   two-sided training is not an improvement: accuracy within noise
   (0.957 vs 0.962), AUC clearly lower (0.971 vs 0.986), and the dry class
   trades recall for precision — dry recall drops 0.951 → 0.880 while
   precision rises 0.878 → 0.912. Under the paper's declared evaluation
   focus (dry recall: a missed dry event costs more than a false alarm),
   that trade is the wrong direction. The ~115k trivially-wet gauge labels
   dilute the dry class (2.5% of the loss) and push the classifier
   conservative on dry calls.
2. **It reproduces the released model's dry-class profile.** Two-sided dry
   precision is 0.912 — exactly the released paper's headline dry precision
   (0.912, with recall 0.765). Strong evidence that the released
   "high-precision / low-recall" dry profile was a product of the
   two-sided label diet, not the architecture.
3. **Imputed-row metrics restate discharge skill.** The imputed val subset
   (thresholded discharge, 37 dry among 1,054) is scored poorly on the dry
   side (P/R ≈ 0.20/0.17) — these labels are a deterministic function of
   same-day discharge, so this panel measures thresholded discharge
   forecasting, not sensor-verified wet/dry skill.
4. Seed variance tightens sharply under two-sided (±0.003 vs ±0.011) — the
   label mass stabilizes optimization but doesn't improve the anchor rows.

**Disposition**: diagnostic only; the canonical flagship remains one-sided
(dry) imputation. Configs/checkpoints retained under `*_2s` names; code
switch is `imputation.wetdry: two_sided` in any pipeline config.
