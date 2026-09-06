# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_ph.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.986 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.983 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.958 | 0.982 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.981 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.961 | 0.985 | 0.976 |
| Day 3 | Headwaters (<=2) | 210 | 0.957 | 0.984 | 0.973 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.957 | 0.985 | 0.974 |
| All Horizons | Headwaters (<=2) | 642 | 0.952 | 0.983 | 0.970 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.997 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.951 | 0.977 | 0.969 |
| 2 | 257 | 0.953 | 0.991 | 0.972 |
| 3 | 142 | 0.979 | 0.997 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 117 | 12 |
| **Wet** | 22 | 633 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.860 | 0.750 | 0.0865 | 382.7 |
| Day 2 | 105 | 0.871 | 0.814 | 0.0767 | 368.0 |
| Day 3 | 109 | 0.757 | 0.788 | 0.1032 | 1246.9 |
