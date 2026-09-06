# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_ph_s43.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.962 | 0.987 | 0.977 |
| Day 1 | Headwaters (<=2) | 216 | 0.963 | 0.985 | 0.977 |
| Day 1 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| Day 2 | All | 264 | 0.958 | 0.982 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.963 | 0.980 | 0.977 |
| Day 2 | Tailwaters (>=3) | 48 | 0.938 | 0.994 | 0.965 |
| Day 3 | All | 257 | 0.965 | 0.984 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.983 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.957 | 0.994 | 0.976 |
| All Horizons | All | 784 | 0.962 | 0.984 | 0.977 |
| All Horizons | Headwaters (<=2) | 642 | 0.964 | 0.983 | 0.978 |
| All Horizons | Tailwaters (>=3) | 142 | 0.951 | 0.996 | 0.972 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.964 | 0.979 | 0.977 |
| 2 | 257 | 0.965 | 0.989 | 0.979 |
| 3 | 142 | 0.951 | 0.996 | 0.972 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 125 | 4 |
| **Wet** | 26 | 629 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.841 | 0.720 | 0.0925 | 487.8 |
| Day 2 | 105 | 0.869 | 0.784 | 0.0772 | 437.7 |
| Day 3 | 109 | 0.763 | 0.761 | 0.1020 | 1337.7 |
