# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_sh.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.962 | 0.978 | 0.977 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.975 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.958 | 0.982 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.981 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.961 | 0.985 | 0.976 |
| Day 3 | Headwaters (<=2) | 210 | 0.957 | 0.984 | 0.973 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.960 | 0.982 | 0.976 |
| All Horizons | Headwaters (<=2) | 642 | 0.955 | 0.980 | 0.972 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.997 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.956 | 0.977 | 0.972 |
| 2 | 257 | 0.953 | 0.990 | 0.972 |
| 3 | 142 | 0.986 | 0.997 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 119 | 10 |
| **Wet** | 21 | 634 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.848 | 0.724 | 0.0903 | 400.0 |
| Day 2 | 105 | 0.862 | 0.786 | 0.0792 | 407.3 |
| Day 3 | 109 | 0.748 | 0.764 | 0.1053 | 1299.6 |
