# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65_2s_s44.pt` | split: `window_split_map_flagq652s.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 667 | 0.958 | 0.968 | 0.976 |
| Day 1 | Headwaters (<=2) | 380 | 0.947 | 0.952 | 0.968 |
| Day 1 | Tailwaters (>=3) | 287 | 0.972 | 1.000 | 0.985 |
| Day 2 | All | 660 | 0.962 | 0.974 | 0.979 |
| Day 2 | Headwaters (<=2) | 371 | 0.957 | 0.960 | 0.974 |
| Day 2 | Tailwaters (>=3) | 289 | 0.969 | 0.998 | 0.984 |
| Day 3 | All | 650 | 0.948 | 0.954 | 0.970 |
| Day 3 | Headwaters (<=2) | 366 | 0.932 | 0.930 | 0.958 |
| Day 3 | Tailwaters (>=3) | 284 | 0.968 | 0.998 | 0.983 |
| All Horizons | All | 1,977 | 0.956 | 0.965 | 0.975 |
| All Horizons | Headwaters (<=2) | 1,117 | 0.945 | 0.947 | 0.967 |
| All Horizons | Tailwaters (>=3) | 860 | 0.970 | 0.999 | 0.984 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 544 | 0.950 | 0.965 | 0.967 |
| 2 | 573 | 0.941 | 0.931 | 0.967 |
| 3 | 473 | 0.945 | 0.998 | 0.970 |
| 4 | 194 | 1.000 | nan | 1.000 |
| 5 | 82 | 1.000 | nan | 1.000 |
| 6 | 111 | 1.000 | nan | 1.000 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 194 | 38 |
| **Wet** | 49 | 1,696 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.886 | 0.673 | 0.5664 | 985.7 |
| Day 2 | 348 | 0.850 | 0.772 | 0.4532 | 748.4 |
| Day 3 | 350 | 0.377 | 0.301 | 1.5095 | 842.8 |
