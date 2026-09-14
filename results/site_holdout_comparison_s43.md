# Site-holdout comparison — all models, same 5 held-out reaches

Split: train = 17 HOBO reaches, test = the 5 reaches held out of RGCN training (2 intermittent + 3 perennial; see context/SITE_HOLDOUT_BASELINES_PLAN.md). Task: wet/dry at t+3. All models receive the held-out site's own lagged status as an input (with-sensor regime). ROC-AUC from probabilities. LR/XGB/LSTM: scaler fit on train sites only; binaries unscaled; ADASYN train-only; released hyperparameters.

| Model | N | Wet frac | Accuracy | ROC-AUC | Wet F1 | Dry F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 603 | 0.78 | 0.844 | 0.894 | 0.893 | 0.713 | 0.880 |
| XGBoost | 603 | 0.78 | 0.978 | 0.987 | 0.986 | 0.949 | 0.910 |
| LSTM (HOBO only) | 453 | 0.71 | 0.751 | 0.887 | 0.830 | 0.535 | 0.496 |
| LSTM (all sites) | 449 | 0.71 | 0.869 | 0.975 | 0.900 | 0.808 | 0.961 |

## Per-site accuracy / dry recall

| Site (dry frac) | Logistic Regression | XGBoost | LSTM (HOBO only) | LSTM (all sites) |
|---|--:|--:|--:|--:|
| 097170 (0.40) | 0.411 / 1.00 | 0.938 / 0.87 | 0.525 / 1.00 | 0.806 / 0.90 |
| 100137 (0.68) | 0.841 / 0.80 | 0.956 / 0.94 | 0.205 / 0.16 | 0.951 / 1.00 |
| 099610 (0.00) | 1.000 | 1.000 | 1.000 | 1.000 |
| 235848 (0.00) | 1.000 | 1.000 | 1.000 | 0.774 |
| 271029 (0.00) | 1.000 | 1.000 | 1.000 | 0.840 |

Caveats: RGCN full-graph training sees held-out sites' input streams as unsupervised context (labels never in loss); baseline held-out rows are absent from training entirely. LSTM N is lower (30-day history requirement). RGCN N counts (site, date) pairs with a t+3 prediction from the stride-1 export.