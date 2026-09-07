# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65_2s.pt` | split: `window_split_map_flagq652s.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 667 | 0.964 | 0.965 | 0.980 |
| Day 1 | Headwaters (<=2) | 380 | 0.937 | 0.943 | 0.962 |
| Day 1 | Tailwaters (>=3) | 287 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 660 | 0.965 | 0.975 | 0.980 |
| Day 2 | Headwaters (<=2) | 371 | 0.941 | 0.961 | 0.964 |
| Day 2 | Tailwaters (>=3) | 289 | 0.997 | 0.999 | 0.998 |
| Day 3 | All | 650 | 0.952 | 0.952 | 0.973 |
| Day 3 | Headwaters (<=2) | 366 | 0.918 | 0.923 | 0.950 |
| Day 3 | Tailwaters (>=3) | 284 | 0.996 | 0.999 | 0.998 |
| All Horizons | All | 1,977 | 0.961 | 0.964 | 0.978 |
| All Horizons | Headwaters (<=2) | 1,117 | 0.932 | 0.942 | 0.959 |
| All Horizons | Tailwaters (>=3) | 860 | 0.998 | 0.999 | 0.999 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 544 | 0.945 | 0.960 | 0.964 |
| 2 | 573 | 0.920 | 0.923 | 0.955 |
| 3 | 473 | 0.996 | 0.998 | 0.998 |
| 4 | 194 | 1.000 | nan | 1.000 |
| 5 | 82 | 1.000 | nan | 1.000 |
| 6 | 111 | 1.000 | nan | 1.000 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 183 | 49 |
| **Wet** | 29 | 1,716 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.839 | 0.571 | 0.6737 | 1643.4 |
| Day 2 | 348 | 0.871 | 0.811 | 0.4213 | 1683.0 |
| Day 3 | 350 | 0.364 | 0.354 | 1.5255 | 2176.2 |
