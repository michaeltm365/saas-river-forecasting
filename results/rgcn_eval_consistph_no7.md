# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_no7.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.920 | 0.965 | 0.953 |
| Day 1 | Headwaters (<=2) | 216 | 0.907 | 0.962 | 0.945 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.920 | 0.972 | 0.954 |
| Day 2 | Headwaters (<=2) | 216 | 0.907 | 0.969 | 0.946 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 1.000 | 0.989 |
| Day 3 | All | 257 | 0.934 | 0.979 | 0.961 |
| Day 3 | Headwaters (<=2) | 210 | 0.924 | 0.977 | 0.955 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.925 | 0.972 | 0.956 |
| All Horizons | Headwaters (<=2) | 642 | 0.913 | 0.969 | 0.948 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.886 | 0.977 | 0.932 |
| 2 | 257 | 0.953 | 0.983 | 0.973 |
| 3 | 142 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 85 | 44 |
| **Wet** | 15 | 640 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.890 | 0.794 | 0.0770 | 277.0 |
| Day 2 | 105 | 0.586 | 0.465 | 0.1373 | 235.9 |
| Day 3 | 109 | 0.897 | 0.774 | 0.0671 | 941.7 |
