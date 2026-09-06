# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_lr3e3.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.962 | 0.986 | 0.977 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.984 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.958 | 0.983 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.981 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.965 | 0.986 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.962 | 0.984 | 0.976 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.962 | 0.985 | 0.977 |
| All Horizons | Headwaters (<=2) | 642 | 0.956 | 0.983 | 0.973 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.996 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.956 | 0.978 | 0.972 |
| 2 | 257 | 0.957 | 0.990 | 0.974 |
| 3 | 142 | 0.986 | 0.996 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 119 | 10 |
| **Wet** | 20 | 635 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.847 | 0.767 | 0.0906 | 873.0 |
| Day 2 | 105 | 0.867 | 0.838 | 0.0778 | 821.9 |
| Day 3 | 109 | 0.741 | 0.802 | 0.1067 | 1686.2 |
