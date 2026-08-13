# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_s44.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.943 | 0.971 | 0.966 |
| Day 1 | Headwaters (<=2) | 216 | 0.940 | 0.971 | 0.963 |
| Day 1 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| Day 2 | All | 264 | 0.947 | 0.974 | 0.968 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.973 | 0.966 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 1.000 | 0.977 |
| Day 3 | All | 257 | 0.953 | 0.980 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.979 | 0.971 |
| Day 3 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| All Horizons | All | 784 | 0.948 | 0.975 | 0.969 |
| All Horizons | Headwaters (<=2) | 642 | 0.945 | 0.975 | 0.967 |
| All Horizons | Tailwaters (>=3) | 142 | 0.958 | 1.000 | 0.976 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.927 | 0.977 | 0.955 |
| 2 | 257 | 0.973 | 0.987 | 0.984 |
| 3 | 142 | 0.958 | 1.000 | 0.976 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 111 | 18 |
| **Wet** | 23 | 632 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.946 | 0.930 | 0.0537 | 708.9 |
| Day 2 | 105 | 0.774 | 0.673 | 0.1014 | 660.5 |
| Day 3 | 109 | 0.939 | 0.933 | 0.0517 | 1459.8 |
