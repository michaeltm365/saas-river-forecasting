# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_q80.pt` | split: `window_split_map_flagq80.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 200 | 0.950 | 0.973 | 0.966 |
| Day 1 | Headwaters (<=2) | 167 | 0.946 | 0.970 | 0.963 |
| Day 1 | Tailwaters (>=3) | 33 | 0.970 | 1.000 | 0.982 |
| Day 2 | All | 192 | 0.953 | 0.970 | 0.969 |
| Day 2 | Headwaters (<=2) | 159 | 0.950 | 0.969 | 0.966 |
| Day 2 | Tailwaters (>=3) | 33 | 0.970 | 0.974 | 0.982 |
| Day 3 | All | 184 | 0.935 | 0.972 | 0.956 |
| Day 3 | Headwaters (<=2) | 153 | 0.928 | 0.972 | 0.950 |
| Day 3 | Tailwaters (>=3) | 31 | 0.968 | 0.981 | 0.981 |
| All Horizons | All | 576 | 0.946 | 0.972 | 0.964 |
| All Horizons | Headwaters (<=2) | 479 | 0.942 | 0.971 | 0.960 |
| All Horizons | Tailwaters (>=3) | 97 | 0.969 | 0.985 | 0.982 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 295 | 0.939 | 0.965 | 0.956 |
| 2 | 184 | 0.946 | 0.989 | 0.965 |
| 3 | 97 | 0.969 | 0.985 | 0.982 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 130 | 8 |
| **Wet** | 23 | 415 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 295 | 0.899 | 0.670 | 0.5788 | 1590.1 |
| Day 2 | 294 | 0.887 | 0.864 | 0.4254 | 1514.9 |
| Day 3 | 296 | 0.397 | 0.371 | 1.6099 | 1857.9 |
