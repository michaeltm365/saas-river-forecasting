# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_s43.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.943 | 0.972 | 0.966 |
| Day 1 | Headwaters (<=2) | 216 | 0.935 | 0.971 | 0.961 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.939 | 0.975 | 0.964 |
| Day 2 | Headwaters (<=2) | 216 | 0.935 | 0.974 | 0.961 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 1.000 | 0.977 |
| Day 3 | All | 257 | 0.946 | 0.980 | 0.968 |
| Day 3 | Headwaters (<=2) | 210 | 0.938 | 0.979 | 0.963 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.943 | 0.975 | 0.966 |
| All Horizons | Headwaters (<=2) | 642 | 0.936 | 0.975 | 0.962 |
| All Horizons | Tailwaters (>=3) | 142 | 0.972 | 1.000 | 0.984 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.909 | 0.982 | 0.945 |
| 2 | 257 | 0.977 | 0.988 | 0.986 |
| 3 | 142 | 0.972 | 1.000 | 0.984 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 101 | 28 |
| **Wet** | 17 | 638 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.834 | 0.801 | 0.0942 | 301.7 |
| Day 2 | 105 | 0.692 | 0.605 | 0.1183 | 285.2 |
| Day 3 | 109 | 0.907 | 0.878 | 0.0639 | 640.9 |
