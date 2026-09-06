# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q80_s43.pt` | split: `window_split_map_flagq80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.945 | 0.978 | 0.963 |
| Day 1 | Headwaters (<=2) | 167 | 0.940 | 0.974 | 0.958 |
| Day 1 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 2 | All | 192 | 0.948 | 0.972 | 0.965 |
| Day 2 | Headwaters (<=2) | 159 | 0.950 | 0.971 | 0.966 |
| Day 2 | Tailwaters (>=3) | 33 | 0.939 | 0.991 | 0.964 |
| Day 3 | All | 184 | 0.951 | 0.972 | 0.967 |
| Day 3 | Headwaters (<=2) | 153 | 0.948 | 0.971 | 0.963 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 0.991 | 0.981 |
| All Horizons | All | 576 | 0.948 | 0.974 | 0.965 |
| All Horizons | Headwaters (<=2) | 479 | 0.946 | 0.972 | 0.962 |
| All Horizons | Tailwaters (>=3) | 97 | 0.959 | 0.994 | 0.976 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.946 | 0.964 | 0.961 |
| 2 | 184 | 0.946 | 0.993 | 0.964 |
| 3 | 97 | 0.959 | 0.994 | 0.976 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 134 | 4 |
| **Wet** | 26 | 412 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.935 | 0.806 | 0.4630 | 1131.1 |
| Day 2 | 294 | 0.804 | 0.794 | 0.5618 | 1159.0 |
| Day 3 | 296 | 0.403 | 0.434 | 1.6019 | 1390.3 |
