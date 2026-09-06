# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_ph_s44.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.970 | 0.981 | 0.981 |
| Day 1 | Headwaters (<=2) | 216 | 0.968 | 0.978 | 0.980 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.970 | 0.979 | 0.982 |
| Day 2 | Headwaters (<=2) | 216 | 0.968 | 0.977 | 0.980 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 1.000 | 0.989 |
| Day 3 | All | 257 | 0.969 | 0.981 | 0.981 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.979 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.969 | 0.980 | 0.981 |
| All Horizons | Headwaters (<=2) | 642 | 0.967 | 0.978 | 0.980 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.964 | 0.976 | 0.977 |
| 2 | 257 | 0.973 | 0.983 | 0.984 |
| 3 | 142 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 125 | 4 |
| **Wet** | 20 | 635 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.847 | 0.750 | 0.0906 | 1082.6 |
| Day 2 | 105 | 0.870 | 0.812 | 0.0769 | 812.2 |
| Day 3 | 109 | 0.776 | 0.786 | 0.0992 | 1653.0 |
