# Diet factorial addendum — LR/XGBoost trained on the all-sites diet

Same shared 5-reach holdout / t+3 protocol as results/site_holdout_comparison_multiseed.md, but LR and XGBoost trained on the HOBO + discretized-discharge row set (the RGCN / LSTM-all-sites diet) instead of HOBO-only. Seeds 42/43/44, mean ± std.

| Model | N | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| LR (all sites) | 599 | 0.820 ± 0.000 | 0.980 ± 0.001 | 0.870 ± 0.000 | 0.708 ± 0.000 | 1.000 ± 0.000 |
| XGBoost (all sites) | 599 | 0.955 ± 0.003 | 0.957 ± 0.000 | 0.971 ± 0.002 | 0.899 ± 0.005 | 0.916 ± 0.000 |

HOBO-diet reference rows (from the multiseed table): LR 0.839 ± 0.005, XGBoost 0.976 ± 0.002.

## Per-site accuracy (seed 44)

| Site | LR (all sites) | XGBoost (all sites) |
|---|--:|--:|
| 097170 | 0.414 | 0.945 |
| 100137 | 0.703 | 0.928 |
| 099610 | 1.000 | 1.000 |
| 235848 | 1.000 | 0.919 |
| 271029 | 1.000 | 1.000 |