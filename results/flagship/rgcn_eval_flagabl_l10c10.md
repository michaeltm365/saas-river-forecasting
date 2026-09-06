# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_l10c10.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.954 | 0.984 | 0.972 |
| Day 1 | Headwaters (<=2) | 216 | 0.949 | 0.982 | 0.969 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.951 | 0.980 | 0.971 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.979 | 0.966 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.953 | 0.983 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.948 | 0.983 | 0.968 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.953 | 0.982 | 0.972 |
| All Horizons | Headwaters (<=2) | 642 | 0.947 | 0.981 | 0.968 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.997 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.940 | 0.976 | 0.963 |
| 2 | 257 | 0.957 | 0.991 | 0.974 |
| 3 | 142 | 0.979 | 0.997 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 113 | 16 |
| **Wet** | 21 | 634 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.851 | 0.740 | 0.0893 | 822.5 |
| Day 2 | 105 | 0.865 | 0.804 | 0.0782 | 571.6 |
| Day 3 | 109 | 0.761 | 0.778 | 0.1024 | 1310.6 |
