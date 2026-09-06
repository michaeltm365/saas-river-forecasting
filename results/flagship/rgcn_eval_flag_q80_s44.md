# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q80_s44.pt` | split: `window_split_map_flagq80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.950 | 0.972 | 0.966 |
| Day 1 | Headwaters (<=2) | 167 | 0.946 | 0.968 | 0.963 |
| Day 1 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 2 | All | 192 | 0.953 | 0.969 | 0.969 |
| Day 2 | Headwaters (<=2) | 159 | 0.950 | 0.967 | 0.966 |
| Day 2 | Tailwaters (>=3) | 33 | 0.970 | 0.991 | 0.982 |
| Day 3 | All | 184 | 0.951 | 0.969 | 0.967 |
| Day 3 | Headwaters (<=2) | 153 | 0.948 | 0.967 | 0.963 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 0.991 | 0.981 |
| All Horizons | All | 576 | 0.951 | 0.969 | 0.967 |
| All Horizons | Headwaters (<=2) | 479 | 0.948 | 0.967 | 0.964 |
| All Horizons | Tailwaters (>=3) | 97 | 0.969 | 0.994 | 0.982 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.946 | 0.963 | 0.961 |
| 2 | 184 | 0.951 | 0.983 | 0.968 |
| 3 | 97 | 0.969 | 0.994 | 0.982 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 134 | 4 |
| **Wet** | 24 | 414 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.937 | 0.767 | 0.4563 | 978.5 |
| Day 2 | 294 | 0.873 | 0.848 | 0.4511 | 1002.9 |
| Day 3 | 296 | 0.395 | 0.367 | 1.6128 | 1055.8 |
