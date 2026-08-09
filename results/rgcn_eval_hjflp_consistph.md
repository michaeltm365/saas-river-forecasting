# RGCN held-out-site evaluation — HJFlp single-visit observations

Checkpoint: `data/retrain/best_model_consistph.pt` | predictions: `data/retrain/predictions_consistph`

HJFlp visits: 472 across 204 reaches (2020-07-21 .. 2020-10-30). Visits by site tier: {'unobserved': 328, 'labeled': 95, 'no-labels': 49}.

| Horizon | Tier | N visits | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|---|--:|--:|--:|--:|--:|--:|
| All | unobserved | 251 | 0.34 | 0.649 | 0.519 | 0.154 | 0.939 |
| All | no-labels | 38 | 0.63 | 0.368 | 0.259 | 0.000 | 1.000 |
| All | labeled | 75 | 0.72 | 0.760 | 0.628 | 0.845 | 0.381 |
| All | ALL | 364 | 0.45 | 0.643 | 0.585 | 0.467 | 0.885 |
| Day 1 | unobserved | 74 | 0.31 | 0.716 | 0.458 | 0.222 | 0.980 |
| Day 1 | no-labels | 8 | 0.75 | 0.250 | 0.375 | 0.000 | 1.000 |
| Day 1 | labeled | 25 | 0.72 | 0.760 | 0.667 | 0.850 | 0.286 |
| Day 1 | ALL | 107 | 0.44 | 0.692 | 0.595 | 0.548 | 0.900 |
| Day 2 | unobserved | 95 | 0.35 | 0.611 | 0.544 | 0.098 | 0.903 |
| Day 2 | no-labels | 23 | 0.70 | 0.304 | 0.192 | 0.000 | 1.000 |
| Day 2 | labeled | 25 | 0.72 | 0.760 | 0.659 | 0.842 | 0.429 |
| Day 2 | ALL | 143 | 0.47 | 0.587 | 0.526 | 0.379 | 0.868 |
| Day 3 | unobserved | 82 | 0.37 | 0.634 | 0.515 | 0.167 | 0.942 |
| Day 3 | no-labels | 7 | 0.29 | 0.714 | 0.300 | 0.000 | 1.000 |
| Day 3 | labeled | 25 | 0.72 | 0.760 | 0.591 | 0.842 | 0.429 |
| Day 3 | ALL | 114 | 0.44 | 0.667 | 0.637 | 0.500 | 0.891 |

## Flow_Status = 0.5 (partial flow) — excluded from metrics above

| Horizon | N | Mean P(wet) | Frac predicted wet |
|---|--:|--:|--:|
| Day 1 | 25 | 0.202 | 0.120 |
| Day 2 | 46 | 0.179 | 0.130 |
| Day 3 | 36 | 0.163 | 0.083 |

Notes: 'unobserved' reaches have no obs.csv history, so their lag and
MaxDepth inputs are all-zero — predictions rely purely on meteorology,
statics, and network topology (true ungauged-site transfer). 'labeled'
reaches host HOBO sensors used in training and are not a spatial holdout.