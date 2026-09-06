# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_lr3e4.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.954 | 0.984 | 0.972 |
| Day 1 | Headwaters (<=2) | 216 | 0.949 | 0.981 | 0.969 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.947 | 0.981 | 0.968 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.980 | 0.966 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 0.994 | 0.977 |
| Day 3 | All | 257 | 0.953 | 0.983 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.948 | 0.983 | 0.968 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.952 | 0.983 | 0.971 |
| All Horizons | Headwaters (<=2) | 642 | 0.947 | 0.981 | 0.968 |
| All Horizons | Tailwaters (>=3) | 142 | 0.972 | 0.997 | 0.984 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.940 | 0.977 | 0.963 |
| 2 | 257 | 0.957 | 0.990 | 0.974 |
| 3 | 142 | 0.972 | 0.997 | 0.984 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 113 | 16 |
| **Wet** | 22 | 633 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.879 | 0.822 | 0.0806 | 452.3 |
| Day 2 | 105 | 0.883 | 0.880 | 0.0730 | 366.5 |
| Day 3 | 109 | 0.769 | 0.832 | 0.1008 | 1000.5 |
