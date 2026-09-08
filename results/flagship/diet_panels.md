# Label-diet-controlled model panels — q65, t+3 (2026-09-07/08)

The experiments that motivated the canonical labeling decisions: (1) keep
**one-sided** (dry-only) discharge imputation, (2) train the canonical LSTM
(all sites) on the same one-sided diet with **no resampling**, (3) declare
the canonical evaluation universe **sensor-verified rows only**. Scripts:
`benchmarks/diet_panel_twosided.py`, `benchmarks/diet_panel_onesided.py`.
All learned models seeds 42/43/44 (mean ± std; LR/XGB deterministic without
ADASYN); identical matched (reach, date) rows per panel; persistence =
observed status as of d−3.

## Panel A — two-sided diet (all trained models on the released-style
two-sided labels; matched N = 1,959, real HOBO 857)

Real-HOBO rows only (N = 857):

| Model | Acc | AUC | Wet F1 | Dry P | Dry R | Dry F1 |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (two-sided) | 0.960 ± 0.002 | 0.970 | 0.975 | 0.914 | 0.893 | 0.903 |
| RGCN (canonical one-sided, cross-diet ref) | 0.963 ± 0.009 | 0.986 | 0.976 | 0.876 | 0.955 | 0.913 |
| LSTM (two-sided, ADASYN) | 0.938 ± 0.043 | 0.984 | 0.959 | 0.798 | 0.983 | 0.876 |
| LR (two-sided, ADASYN) | 0.640 ± 0.004 | 0.944 | 0.706 | 0.366 | 1.000 | 0.536 |
| XGBoost (two-sided, ADASYN) | 0.958 ± 0.003 | 0.966 | 0.973 | 0.870 | 0.938 | 0.903 |
| Persistence | 0.967 | 0.957 | 0.979 | 0.908 | 0.938 | 0.923 |

All-matched-rows panel (N = 1,959, ~56% discharge-imputed labels) in the
script output; on it the cross-diet one-sided RGCN craters to 0.527 —
demonstrating that two-sided all-rows metrics measure threshold agreement,
not wet/dry skill.

## Panel B — one-sided diet (all trained models on the canonical one-sided
labels, NO resampling; matched N = 948, real HOBO 908)

Real-HOBO rows only (N = 908):

| Model | Acc | AUC | Wet F1 | Dry P | Dry R | Dry F1 |
|---|--:|--:|--:|--:|--:|--:|
| RGCN (canonical) | 0.962 ± 0.011 | **0.986** | 0.976 | 0.878 | 0.951 | 0.912 |
| LSTM (one-sided) | 0.952 ± 0.012 | 0.978 | 0.969 | 0.835 | **0.968** | 0.895 |
| LR (one-sided) | 0.965 | 0.975 | 0.977 | 0.883 | 0.958 | 0.919 |
| XGBoost (one-sided) | **0.968** | 0.984 | 0.980 | 0.912 | 0.937 | 0.924 |
| Persistence | 0.966 | 0.953 | 0.978 | 0.907 | 0.938 | 0.919 |
| RGCN (two-sided, cross-diet ref) | 0.957 ± 0.003 | 0.971 | 0.973 | 0.912 | 0.880 | 0.896 |

## Findings

1. **Diet-controlled, the simple models catch up**: XGBoost (0.968) edges
   the RGCN (0.962) on accuracy and dry F1; even plain LR (0.965) matches
   it. LR's Panel-A collapse was an ADASYN-plus-diet artifact. The RGCN's
   robust measurable edges are ROC-AUC (0.986, best in both panels) and dry
   recall among accuracy-competitive models; the LSTM holds top dry recall.
   Replicates the shelved site-holdout diet factorial on the canonical
   temporal split.
2. **Two-sided training suppresses dry recall diet-wide**: the 2s RGCN's
   dry P/R (0.912/0.880) reproduces the released paper's dry-precision
   profile, and no hyperparameter setting recovers one-sided dry recall
   (`rgcn_ablation_sweep_q65_2s.md`: fpw 1x/2x/4x → dry recall
   0.740/0.766/0.844, all below the one-sided default's 0.870).
3. **Cross-diet scoring is invalid in both directions** (0.527 and 0.924
   artifacts) — published tables must be single-diet.
4. The one-sided training set is ~77% dry (2,322 wet / 7,675 dry of 9,997
   labeled targets), so minority oversampling would target the WET class —
   hence no resampling in the canonical protocol.
