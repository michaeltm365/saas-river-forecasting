# RGCN held-out-site evaluation — HJFlp single-visit observations

Checkpoint: `data/retrain/best_model_consistph_nolag_cls.pt` | predictions: `data/retrain/predictions_consistph_nolag_cls`

HJFlp visits: 472 across 204 reaches (2020-07-21 .. 2020-10-30). Visits by site tier: {'unobserved': 328, 'labeled': 95, 'no-labels': 49}.

| Horizon | Tier | N visits | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|---|--:|--:|--:|--:|--:|--:|
| All | unobserved | 251 | 0.34 | 0.514 | 0.539 | 0.408 | 0.527 |
| All | no-labels | 38 | 0.63 | 0.342 | 0.277 | 0.194 | 0.714 |
| All | labeled | 75 | 0.72 | 0.680 | 0.679 | 0.765 | 0.571 |
| All | ALL | 364 | 0.45 | 0.530 | 0.549 | 0.496 | 0.545 |
| Day 1 | unobserved | 74 | 0.31 | 0.500 | 0.585 | 0.393 | 0.490 |
| Day 1 | no-labels | 8 | 0.75 | 0.500 | 0.458 | 0.500 | 1.000 |
| Day 1 | labeled | 25 | 0.72 | 0.720 | 0.738 | 0.788 | 0.714 |
| Day 1 | ALL | 107 | 0.44 | 0.551 | 0.617 | 0.529 | 0.533 |
| Day 2 | unobserved | 95 | 0.35 | 0.568 | 0.595 | 0.468 | 0.581 |
| Day 2 | no-labels | 23 | 0.70 | 0.304 | 0.237 | 0.111 | 0.857 |
| Day 2 | labeled | 25 | 0.72 | 0.720 | 0.698 | 0.800 | 0.571 |
| Day 2 | ALL | 143 | 0.47 | 0.552 | 0.541 | 0.508 | 0.605 |
| Day 3 | unobserved | 82 | 0.37 | 0.463 | 0.446 | 0.353 | 0.500 |
| Day 3 | no-labels | 7 | 0.29 | 0.286 | 0.300 | 0.000 | 0.400 |
| Day 3 | labeled | 25 | 0.72 | 0.600 | 0.607 | 0.706 | 0.429 |
| Day 3 | ALL | 114 | 0.44 | 0.482 | 0.504 | 0.449 | 0.484 |

## Flow_Status = 0.5 (partial flow) — excluded from metrics above

| Horizon | N | Mean P(wet) | Frac predicted wet |
|---|--:|--:|--:|
| Day 1 | 25 | 0.522 | 0.600 |
| Day 2 | 46 | 0.318 | 0.370 |
| Day 3 | 36 | 0.372 | 0.361 |

Notes: 'unobserved' reaches have no obs.csv history, so their lag and
MaxDepth inputs are all-zero — predictions rely purely on meteorology,
statics, and network topology (true ungauged-site transfer). 'labeled'
reaches host HOBO sensors used in training and are not a spatial holdout.