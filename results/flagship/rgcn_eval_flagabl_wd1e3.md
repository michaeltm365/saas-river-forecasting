# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_wd1e3.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.982 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.949 | 0.979 | 0.968 |
| Day 1 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| Day 2 | All | 264 | 0.951 | 0.978 | 0.970 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.977 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.938 | 0.994 | 0.965 |
| Day 3 | All | 257 | 0.953 | 0.981 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.957 | 0.981 | 0.973 |
| Day 3 | Tailwaters (>=3) | 47 | 0.936 | 0.994 | 0.964 |
| All Horizons | All | 784 | 0.952 | 0.980 | 0.971 |
| All Horizons | Headwaters (<=2) | 642 | 0.953 | 0.979 | 0.971 |
| All Horizons | Tailwaters (>=3) | 142 | 0.944 | 0.996 | 0.968 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.945 | 0.978 | 0.966 |
| 2 | 257 | 0.965 | 0.985 | 0.979 |
| 3 | 142 | 0.944 | 0.996 | 0.968 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 118 | 11 |
| **Wet** | 27 | 628 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.841 | 0.715 | 0.0923 | 273.8 |
| Day 2 | 105 | 0.871 | 0.791 | 0.0767 | 258.4 |
| Day 3 | 109 | 0.771 | 0.773 | 0.1003 | 956.7 |
