# RGCN retrain — evaluation report

Checkpoint: `data/retrain/correction_sep11/rgcn_availability_s44.pt` | split: `window_split_map_flagq65.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 332 | 0.967 | 0.984 | 0.978 |
| Day 1 | Headwaters (<=2) | 275 | 0.967 | 0.982 | 0.978 |
| Day 1 | Tailwaters (>=3) | 57 | 0.965 | 1.000 | 0.978 |
| Day 2 | All | 322 | 0.969 | 0.980 | 0.979 |
| Day 2 | Headwaters (<=2) | 265 | 0.974 | 0.980 | 0.982 |
| Day 2 | Tailwaters (>=3) | 57 | 0.947 | 0.994 | 0.967 |
| Day 3 | All | 314 | 0.965 | 0.981 | 0.976 |
| Day 3 | Headwaters (<=2) | 259 | 0.969 | 0.980 | 0.979 |
| Day 3 | Tailwaters (>=3) | 55 | 0.945 | 0.991 | 0.966 |
| All Horizons | All | 968 | 0.967 | 0.982 | 0.978 |
| All Horizons | Headwaters (<=2) | 799 | 0.970 | 0.980 | 0.980 |
| All Horizons | Tailwaters (>=3) | 169 | 0.953 | 0.995 | 0.970 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 486 | 0.967 | 0.978 | 0.977 |
| 2 | 313 | 0.974 | 0.988 | 0.984 |
| 3 | 169 | 0.953 | 0.995 | 0.970 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 228 | 4 |
| **Wet** | 28 | 708 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.886 | 0.659 | 0.5674 | 1953.6 |
| Day 2 | 348 | 0.869 | 0.826 | 0.4240 | 1692.6 |
| Day 3 | 350 | 0.399 | 0.363 | 1.4832 | 1843.2 |
