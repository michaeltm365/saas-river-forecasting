# Site-holdout comparison — all models, same 5 held-out reaches

Split: train = 17 HOBO reaches, test = the 5 reaches held out of RGCN training (2 intermittent + 3 perennial; see context/SITE_HOLDOUT_BASELINES_PLAN.md). Task: wet/dry at t+3. All models receive the held-out site's own lagged status as an input (with-sensor regime). ROC-AUC from probabilities. LR/XGB/LSTM: scaler fit on train sites only; binaries unscaled; ADASYN train-only; released hyperparameters.

| Model | N | Wet frac | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 603 | 0.78 | 0.833 | 0.894 | 0.886 | 0.685 | 0.827 |
| XGBoost | 603 | 0.78 | 0.977 | 0.983 | 0.985 | 0.945 | 0.910 |
| LSTM (HOBO only) | 453 | 0.71 | 0.755 | 0.892 | 0.832 | 0.551 | 0.519 |
| LSTM (all sites) | 449 | 0.71 | 0.931 | 0.965 | 0.951 | 0.885 | 0.922 |
| RGCN (strict, with-sensor holdout) | 613 | 0.79 | 0.946 | 0.980 | 0.966 | 0.871 | 0.847 |

## Per-site accuracy / dry recall

| Site (dry frac) | Logistic Regression | XGBoost | LSTM (HOBO only) | LSTM (all sites) | RGCN (strict, with-sensor holdout) |
|---|--:|--:|--:|--:|--:|
| 097170 (0.40) | 0.411 / 1.00 | 0.938 / 0.87 | 0.525 / 1.00 | 0.898 / 0.88 | 0.954 / 0.92 |
| 100137 (0.68) | 0.779 / 0.71 | 0.947 / 0.94 | 0.229 / 0.20 | 0.926 / 0.95 | 0.816 / 0.79 |
| 099610 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 235848 (0.00) | 1.000 | 1.000 | 1.000 | 0.839 | 0.976 |
| 271029 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 | 0.976 |

Caveats: RGCN full-graph training sees held-out sites' input streams as unsupervised context (labels never in loss); baseline held-out rows are absent from training entirely. LSTM N is lower (30-day history requirement). RGCN N counts (site, date) pairs with a t+3 prediction from the stride-1 export.