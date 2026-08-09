# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consist80.pt` | split: `window_split_map_consist80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.955 | 0.981 | 0.970 |
| Day 1 | Headwaters (<=2) | 167 | 0.946 | 0.977 | 0.963 |
| Day 1 | Tailwaters (>=3) | 33 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 192 | 0.948 | 0.982 | 0.966 |
| Day 2 | Headwaters (<=2) | 159 | 0.943 | 0.980 | 0.962 |
| Day 2 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 3 | All | 184 | 0.951 | 0.982 | 0.967 |
| Day 3 | Headwaters (<=2) | 153 | 0.948 | 0.980 | 0.963 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 1.000 | 0.981 |
| All Horizons | All | 576 | 0.951 | 0.982 | 0.968 |
| All Horizons | Headwaters (<=2) | 479 | 0.946 | 0.979 | 0.963 |
| All Horizons | Tailwaters (>=3) | 97 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.936 | 0.972 | 0.954 |
| 2 | 184 | 0.962 | 0.994 | 0.975 |
| 3 | 97 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 130 | 8 |
| **Wet** | 20 | 418 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.964 | 0.886 | 0.3449 | 2265.3 |
| Day 2 | 294 | 0.930 | 0.908 | 0.3347 | 1854.6 |
| Day 3 | 296 | 0.949 | 0.906 | 0.4699 | 2366.2 |
