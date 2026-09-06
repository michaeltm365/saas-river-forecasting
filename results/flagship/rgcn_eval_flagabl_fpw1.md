# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_fpw1.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.954 | 0.982 | 0.972 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.978 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.951 | 0.981 | 0.971 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.979 | 0.966 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.957 | 0.984 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.983 | 0.971 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.954 | 0.982 | 0.972 |
| All Horizons | Headwaters (<=2) | 642 | 0.947 | 0.980 | 0.967 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.996 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.943 | 0.976 | 0.964 |
| 2 | 257 | 0.953 | 0.989 | 0.972 |
| 3 | 142 | 0.986 | 0.996 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 114 | 15 |
| **Wet** | 21 | 634 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.866 | 0.774 | 0.0846 | 423.5 |
| Day 2 | 105 | 0.875 | 0.836 | 0.0753 | 424.0 |
| Day 3 | 109 | 0.761 | 0.804 | 0.1025 | 1384.4 |
