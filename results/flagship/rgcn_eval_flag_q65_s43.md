# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65_s43.pt` | split: `window_split_map_flagq65.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 332 | 0.964 | 0.988 | 0.976 |
| Day 1 | Headwaters (<=2) | 275 | 0.964 | 0.986 | 0.975 |
| Day 1 | Tailwaters (>=3) | 57 | 0.965 | 1.000 | 0.978 |
| Day 2 | All | 322 | 0.969 | 0.982 | 0.979 |
| Day 2 | Headwaters (<=2) | 265 | 0.970 | 0.982 | 0.980 |
| Day 2 | Tailwaters (>=3) | 57 | 0.965 | 0.991 | 0.978 |
| Day 3 | All | 314 | 0.971 | 0.982 | 0.981 |
| Day 3 | Headwaters (<=2) | 259 | 0.969 | 0.981 | 0.979 |
| Day 3 | Tailwaters (>=3) | 55 | 0.982 | 0.991 | 0.989 |
| All Horizons | All | 968 | 0.968 | 0.984 | 0.979 |
| All Horizons | Headwaters (<=2) | 799 | 0.967 | 0.983 | 0.978 |
| All Horizons | Tailwaters (>=3) | 169 | 0.970 | 0.995 | 0.982 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 486 | 0.967 | 0.977 | 0.977 |
| 2 | 313 | 0.968 | 0.997 | 0.980 |
| 3 | 169 | 0.970 | 0.995 | 0.982 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 228 | 4 |
| **Wet** | 27 | 709 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.916 | 0.714 | 0.4882 | 1209.0 |
| Day 2 | 348 | 0.860 | 0.834 | 0.4392 | 1342.4 |
| Day 3 | 350 | 0.398 | 0.375 | 1.4842 | 1579.2 |
