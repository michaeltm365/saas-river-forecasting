# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_strict_s44.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.958 | 0.979 | 0.975 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.976 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.962 | 0.977 | 0.977 |
| Day 2 | Headwaters (<=2) | 216 | 0.958 | 0.975 | 0.975 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.957 | 0.980 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.978 | 0.970 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.959 | 0.979 | 0.975 |
| All Horizons | Headwaters (<=2) | 642 | 0.955 | 0.977 | 0.972 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.996 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.951 | 0.974 | 0.969 |
| 2 | 257 | 0.961 | 0.986 | 0.976 |
| 3 | 142 | 0.979 | 0.996 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 120 | 9 |
| **Wet** | 23 | 632 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.845 | 0.764 | 0.0911 | 631.5 |
| Day 2 | 105 | 0.868 | 0.838 | 0.0774 | 411.8 |
| Day 3 | 109 | 0.751 | 0.804 | 0.1046 | 1143.7 |
