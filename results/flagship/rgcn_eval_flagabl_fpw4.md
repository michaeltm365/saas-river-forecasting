# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_fpw4.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.962 | 0.985 | 0.977 |
| Day 1 | Headwaters (<=2) | 216 | 0.958 | 0.983 | 0.974 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.958 | 0.983 | 0.975 |
| Day 2 | Headwaters (<=2) | 216 | 0.958 | 0.982 | 0.975 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 0.994 | 0.977 |
| Day 3 | All | 257 | 0.965 | 0.985 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.985 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.957 | 0.994 | 0.976 |
| All Horizons | All | 784 | 0.962 | 0.985 | 0.977 |
| All Horizons | Headwaters (<=2) | 642 | 0.961 | 0.983 | 0.976 |
| All Horizons | Tailwaters (>=3) | 142 | 0.965 | 0.997 | 0.980 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.958 | 0.978 | 0.974 |
| 2 | 257 | 0.965 | 0.992 | 0.979 |
| 3 | 142 | 0.965 | 0.997 | 0.980 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 123 | 6 |
| **Wet** | 24 | 631 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.855 | 0.773 | 0.0882 | 802.1 |
| Day 2 | 105 | 0.874 | 0.841 | 0.0757 | 649.8 |
| Day 3 | 109 | 0.773 | 0.815 | 0.0998 | 1529.1 |
