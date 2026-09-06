# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65_s44.pt` | split: `window_split_map_flagq65.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 332 | 0.967 | 0.986 | 0.978 |
| Day 1 | Headwaters (<=2) | 275 | 0.964 | 0.983 | 0.975 |
| Day 1 | Tailwaters (>=3) | 57 | 0.982 | 1.000 | 0.989 |
| Day 2 | All | 322 | 0.969 | 0.985 | 0.979 |
| Day 2 | Headwaters (<=2) | 265 | 0.970 | 0.984 | 0.980 |
| Day 2 | Tailwaters (>=3) | 57 | 0.965 | 0.998 | 0.978 |
| Day 3 | All | 314 | 0.971 | 0.985 | 0.981 |
| Day 3 | Headwaters (<=2) | 259 | 0.969 | 0.984 | 0.979 |
| Day 3 | Tailwaters (>=3) | 55 | 0.982 | 0.998 | 0.989 |
| All Horizons | All | 968 | 0.969 | 0.985 | 0.979 |
| All Horizons | Headwaters (<=2) | 799 | 0.967 | 0.983 | 0.978 |
| All Horizons | Tailwaters (>=3) | 169 | 0.976 | 0.999 | 0.985 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 486 | 0.967 | 0.982 | 0.977 |
| 2 | 313 | 0.968 | 0.992 | 0.980 |
| 3 | 169 | 0.976 | 0.999 | 0.985 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 228 | 4 |
| **Wet** | 26 | 710 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.893 | 0.650 | 0.5506 | 1043.7 |
| Day 2 | 348 | 0.880 | 0.789 | 0.4061 | 1074.9 |
| Day 3 | 350 | 0.388 | 0.322 | 1.4960 | 1176.8 |
