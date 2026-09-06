# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_nostat.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.980 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.977 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.951 | 0.977 | 0.971 |
| Day 2 | Headwaters (<=2) | 216 | 0.944 | 0.974 | 0.966 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.957 | 0.980 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.977 | 0.970 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.953 | 0.979 | 0.971 |
| All Horizons | Headwaters (<=2) | 642 | 0.947 | 0.976 | 0.967 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.998 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.943 | 0.976 | 0.964 |
| 2 | 257 | 0.953 | 0.982 | 0.972 |
| 3 | 142 | 0.979 | 0.998 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 117 | 12 |
| **Wet** | 25 | 630 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.880 | 0.809 | 0.0803 | 492.7 |
| Day 2 | 105 | 0.892 | 0.874 | 0.0700 | 573.7 |
| Day 3 | 109 | 0.715 | 0.817 | 0.1119 | 4171.5 |
