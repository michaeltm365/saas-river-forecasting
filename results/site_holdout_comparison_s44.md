# Site-holdout comparison — all models, same 5 held-out reaches

Split: train = 17 HOBO reaches, test = the 5 reaches held out of RGCN training (2 intermittent + 3 perennial; see context/SITE_HOLDOUT_BASELINES_PLAN.md). Task: wet/dry at t+3. All models receive the held-out site's own lagged status as an input (with-sensor regime). ROC-AUC from probabilities. LR/XGB/LSTM: scaler fit on train sites only; binaries unscaled; ADASYN train-only; released hyperparameters.

| Model | N | Wet frac | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 603 | 0.78 | 0.841 | 0.894 | 0.891 | 0.706 | 0.865 |
| XGBoost | 603 | 0.78 | 0.973 | 0.982 | 0.983 | 0.937 | 0.895 |
| LSTM (HOBO only) | 453 | 0.71 | 0.722 | 0.891 | 0.814 | 0.452 | 0.397 |
| LSTM (all sites) | 449 | 0.71 | 0.944 | 0.974 | 0.962 | 0.898 | 0.853 |

## Per-site accuracy / dry recall

| Site (dry frac) | Logistic Regression | XGBoost | LSTM (HOBO only) | LSTM (all sites) |
|---|--:|--:|--:|--:|
| 097170 (0.40) | 0.411 / 1.00 | 0.922 / 0.83 | 0.525 / 1.00 | 0.816 / 0.69 |
| 100137 (0.68) | 0.823 / 0.78 | 0.947 / 0.94 | 0.048 / 0.00 | 0.914 / 0.96 |
| 099610 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 |
| 235848 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 |
| 271029 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 |

Caveats: RGCN full-graph training sees held-out sites' input streams as unsupervised context (labels never in loss); baseline held-out rows are absent from training entirely. LSTM N is lower (30-day history requirement). RGCN N counts (site, date) pairs with a t+3 prediction from the stride-1 export.