# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_strict_no7.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.928 | 0.974 | 0.957 |
| Day 1 | Headwaters (<=2) | 216 | 0.912 | 0.969 | 0.947 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.924 | 0.973 | 0.956 |
| Day 2 | Headwaters (<=2) | 216 | 0.912 | 0.970 | 0.948 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.930 | 0.978 | 0.959 |
| Day 3 | Headwaters (<=2) | 210 | 0.919 | 0.975 | 0.951 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.927 | 0.975 | 0.957 |
| All Horizons | Headwaters (<=2) | 642 | 0.914 | 0.972 | 0.949 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.997 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.888 | 0.972 | 0.933 |
| 2 | 257 | 0.953 | 0.989 | 0.972 |
| 3 | 142 | 0.986 | 0.997 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 91 | 38 |
| **Wet** | 19 | 636 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.856 | 0.750 | 0.0879 | 513.1 |
| Day 2 | 105 | 0.866 | 0.813 | 0.0780 | 454.2 |
| Day 3 | 109 | 0.753 | 0.785 | 0.1042 | 1697.5 |
