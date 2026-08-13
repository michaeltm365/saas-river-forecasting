# RGCN temporal headline — multi-seed (phases split, mean ± std)

Seeds 42/43/44 of the consistency variants (30-day window, tail masking). Val = 784 labels, 3 hydrologic-phase blocks. Single-seed eval reports: results/rgcn_eval_consistph*(_s4x).md.

| Variant | Accuracy | ROC-AUC | F1 |
|---|--:|--:|--:|
| Variant A (obs-masked tail, live drivers) | 0.945 ± 0.002 | 0.976 ± 0.001 | 0.967 ± 0.001 |
| A-strict (no post-t information) | 0.951 ± 0.006 | 0.973 ± 0.005 | 0.970 ± 0.003 |

Reference (single-seed 42): unmasked 28-day phases retrain = 0.953 acc.