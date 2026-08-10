# RGCN held-out-site evaluation — HJFlp single-visit observations

Checkpoint: `data/retrain/best_model_consistph_nolag.pt` | predictions: `data/retrain/predictions_consistph_nolag`

HJFlp visits: 472 across 204 reaches (2020-07-21 .. 2020-10-30). Visits by site tier: {'unobserved': 328, 'labeled': 95, 'no-labels': 49}.

| Horizon | Tier | N visits | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|---|--:|--:|--:|--:|--:|--:|
| All | unobserved | 251 | 0.34 | 0.566 | 0.530 | 0.435 | 0.606 |
| All | no-labels | 38 | 0.63 | 0.368 | 0.312 | 0.250 | 0.714 |
| All | labeled | 75 | 0.72 | 0.573 | 0.567 | 0.667 | 0.524 |
| All | ALL | 364 | 0.45 | 0.547 | 0.512 | 0.486 | 0.605 |
| Day 1 | unobserved | 74 | 0.31 | 0.581 | 0.606 | 0.475 | 0.569 |
| Day 1 | no-labels | 8 | 0.75 | 0.500 | 0.375 | 0.500 | 1.000 |
| Day 1 | labeled | 25 | 0.72 | 0.680 | 0.627 | 0.765 | 0.571 |
| Day 1 | ALL | 107 | 0.44 | 0.598 | 0.587 | 0.574 | 0.583 |
| Day 2 | unobserved | 95 | 0.35 | 0.600 | 0.561 | 0.472 | 0.645 |
| Day 2 | no-labels | 23 | 0.70 | 0.348 | 0.308 | 0.211 | 0.857 |
| Day 2 | labeled | 25 | 0.72 | 0.440 | 0.476 | 0.500 | 0.571 |
| Day 2 | ALL | 143 | 0.47 | 0.531 | 0.473 | 0.437 | 0.658 |
| Day 3 | unobserved | 82 | 0.37 | 0.512 | 0.438 | 0.355 | 0.596 |
| Day 3 | no-labels | 7 | 0.29 | 0.286 | 0.400 | 0.000 | 0.400 |
| Day 3 | labeled | 25 | 0.72 | 0.600 | 0.583 | 0.706 | 0.429 |
| Day 3 | ALL | 114 | 0.44 | 0.518 | 0.495 | 0.455 | 0.562 |

## Flow_Status = 0.5 (partial flow) — excluded from metrics above

| Horizon | N | Mean P(wet) | Frac predicted wet |
|---|--:|--:|--:|
| Day 1 | 25 | 0.433 | 0.440 |
| Day 2 | 46 | 0.277 | 0.304 |
| Day 3 | 36 | 0.298 | 0.306 |

Notes: 'unobserved' reaches have no obs.csv history, so their lag and
MaxDepth inputs are all-zero — predictions rely purely on meteorology,
statics, and network topology (true ungauged-site transfer). 'labeled'
reaches host HOBO sensors used in training and are not a spatial holdout.