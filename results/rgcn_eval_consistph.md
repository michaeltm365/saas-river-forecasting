# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.947 | 0.973 | 0.968 |
| Day 1 | Headwaters (<=2) | 216 | 0.940 | 0.970 | 0.963 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.943 | 0.978 | 0.967 |
| Day 2 | Headwaters (<=2) | 216 | 0.935 | 0.976 | 0.961 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 1.000 | 0.989 |
| Day 3 | All | 257 | 0.946 | 0.982 | 0.968 |
| Day 3 | Headwaters (<=2) | 210 | 0.938 | 0.980 | 0.963 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.945 | 0.977 | 0.967 |
| All Horizons | Headwaters (<=2) | 642 | 0.938 | 0.975 | 0.962 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.914 | 0.979 | 0.948 |
| 2 | 257 | 0.973 | 0.988 | 0.984 |
| 3 | 142 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 102 | 27 |
| **Wet** | 16 | 639 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.932 | 0.864 | 0.0604 | 351.4 |
| Day 2 | 105 | 0.811 | 0.646 | 0.0926 | 370.1 |
| Day 3 | 109 | 0.925 | 0.855 | 0.0575 | 2246.9 |
