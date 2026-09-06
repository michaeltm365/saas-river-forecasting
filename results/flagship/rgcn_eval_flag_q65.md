# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q65.pt` | split: `window_split_map_flagq65.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 332 | 0.958 | 0.985 | 0.972 |
| Day 1 | Headwaters (<=2) | 275 | 0.953 | 0.982 | 0.968 |
| Day 1 | Tailwaters (>=3) | 57 | 0.982 | 1.000 | 0.989 |
| Day 2 | All | 322 | 0.950 | 0.980 | 0.968 |
| Day 2 | Headwaters (<=2) | 265 | 0.943 | 0.979 | 0.963 |
| Day 2 | Tailwaters (>=3) | 57 | 0.982 | 0.987 | 0.989 |
| Day 3 | All | 314 | 0.943 | 0.982 | 0.962 |
| Day 3 | Headwaters (<=2) | 259 | 0.934 | 0.980 | 0.956 |
| Day 3 | Tailwaters (>=3) | 55 | 0.982 | 0.989 | 0.989 |
| All Horizons | All | 968 | 0.950 | 0.982 | 0.967 |
| All Horizons | Headwaters (<=2) | 799 | 0.944 | 0.981 | 0.962 |
| All Horizons | Tailwaters (>=3) | 169 | 0.982 | 0.992 | 0.989 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 486 | 0.928 | 0.978 | 0.950 |
| 2 | 313 | 0.968 | 0.995 | 0.980 |
| 3 | 169 | 0.982 | 0.992 | 0.989 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 207 | 25 |
| **Wet** | 23 | 713 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 349 | 0.938 | 0.760 | 0.4184 | 1355.8 |
| Day 2 | 348 | 0.865 | 0.848 | 0.4302 | 1501.9 |
| Day 3 | 350 | 0.396 | 0.392 | 1.4863 | 1754.4 |
