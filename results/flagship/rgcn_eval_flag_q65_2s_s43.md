# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65_2s_s43.pt` | split: `window_split_map_flagq652s.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 667 | 0.964 | 0.966 | 0.980 |
| Day 1 | Headwaters (<=2) | 380 | 0.937 | 0.945 | 0.962 |
| Day 1 | Tailwaters (>=3) | 287 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 660 | 0.967 | 0.973 | 0.981 |
| Day 2 | Headwaters (<=2) | 371 | 0.943 | 0.957 | 0.966 |
| Day 2 | Tailwaters (>=3) | 289 | 0.997 | 0.999 | 0.998 |
| Day 3 | All | 650 | 0.955 | 0.951 | 0.975 |
| Day 3 | Headwaters (<=2) | 366 | 0.923 | 0.920 | 0.954 |
| Day 3 | Tailwaters (>=3) | 284 | 0.996 | 0.999 | 0.998 |
| All Horizons | All | 1,977 | 0.962 | 0.963 | 0.979 |
| All Horizons | Headwaters (<=2) | 1,117 | 0.935 | 0.941 | 0.961 |
| All Horizons | Tailwaters (>=3) | 860 | 0.998 | 0.999 | 0.999 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 544 | 0.943 | 0.965 | 0.962 |
| 2 | 573 | 0.927 | 0.910 | 0.959 |
| 3 | 473 | 0.996 | 0.998 | 0.998 |
| 4 | 194 | 1.000 | nan | 1.000 |
| 5 | 82 | 1.000 | nan | 1.000 |
| 6 | 111 | 1.000 | nan | 1.000 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 181 | 51 |
| **Wet** | 24 | 1,721 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.907 | 0.702 | 0.5120 | 1475.0 |
| Day 2 | 348 | 0.853 | 0.855 | 0.4487 | 1890.9 |
| Day 3 | 350 | 0.394 | 0.422 | 1.4889 | 1882.0 |
