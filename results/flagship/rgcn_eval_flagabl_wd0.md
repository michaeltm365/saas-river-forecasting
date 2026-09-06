# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_wd0.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.958 | 0.988 | 0.975 |
| Day 1 | Headwaters (<=2) | 216 | 0.949 | 0.985 | 0.969 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.955 | 0.983 | 0.973 |
| Day 2 | Headwaters (<=2) | 216 | 0.949 | 0.981 | 0.969 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.965 | 0.985 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.962 | 0.983 | 0.976 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.959 | 0.985 | 0.975 |
| All Horizons | Headwaters (<=2) | 642 | 0.953 | 0.983 | 0.971 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.997 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.948 | 0.979 | 0.968 |
| 2 | 257 | 0.961 | 0.990 | 0.977 |
| 3 | 142 | 0.986 | 0.997 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 116 | 13 |
| **Wet** | 19 | 636 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.866 | 0.774 | 0.0848 | 594.8 |
| Day 2 | 105 | 0.870 | 0.835 | 0.0768 | 403.6 |
| Day 3 | 109 | 0.759 | 0.801 | 0.1028 | 1254.8 |
