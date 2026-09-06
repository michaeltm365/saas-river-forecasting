# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_l05c10.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.962 | 0.988 | 0.977 |
| Day 1 | Headwaters (<=2) | 216 | 0.958 | 0.986 | 0.974 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.962 | 0.983 | 0.977 |
| Day 2 | Headwaters (<=2) | 216 | 0.958 | 0.982 | 0.975 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.969 | 0.985 | 0.981 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.984 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.964 | 0.985 | 0.978 |
| All Horizons | Headwaters (<=2) | 642 | 0.961 | 0.984 | 0.976 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.996 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.956 | 0.978 | 0.972 |
| 2 | 257 | 0.969 | 0.992 | 0.981 |
| 3 | 142 | 0.979 | 0.996 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 119 | 10 |
| **Wet** | 18 | 637 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.819 | 0.696 | 0.0985 | 627.0 |
| Day 2 | 105 | 0.846 | 0.764 | 0.0836 | 536.8 |
| Day 3 | 109 | 0.754 | 0.747 | 0.1040 | 870.6 |
