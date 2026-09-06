# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_cls_only.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.954 | 0.982 | 0.972 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.980 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| Day 2 | All | 264 | 0.958 | 0.979 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.958 | 0.978 | 0.975 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 0.994 | 0.977 |
| Day 3 | All | 257 | 0.965 | 0.983 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.983 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.957 | 0.994 | 0.976 |
| All Horizons | All | 784 | 0.959 | 0.981 | 0.975 |
| All Horizons | Headwaters (<=2) | 642 | 0.960 | 0.980 | 0.975 |
| All Horizons | Tailwaters (>=3) | 142 | 0.958 | 0.996 | 0.976 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.948 | 0.977 | 0.968 |
| 2 | 257 | 0.977 | 0.986 | 0.986 |
| 3 | 142 | 0.958 | 0.996 | 0.976 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 119 | 10 |
| **Wet** | 22 | 633 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | -0.277 | -1.073 | 0.2617 | 1060.0 |
| Day 2 | 105 | -0.292 | -1.088 | 0.2424 | 905.5 |
| Day 3 | 109 | -0.269 | -1.060 | 0.2361 | 1284.3 |
