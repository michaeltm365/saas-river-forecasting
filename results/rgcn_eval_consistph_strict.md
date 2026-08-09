# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_strict.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.947 | 0.975 | 0.968 |
| Day 1 | Headwaters (<=2) | 216 | 0.940 | 0.971 | 0.963 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.947 | 0.973 | 0.969 |
| Day 2 | Headwaters (<=2) | 216 | 0.940 | 0.970 | 0.964 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.946 | 0.977 | 0.967 |
| Day 3 | Headwaters (<=2) | 210 | 0.938 | 0.975 | 0.962 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.946 | 0.975 | 0.968 |
| All Horizons | Headwaters (<=2) | 642 | 0.939 | 0.972 | 0.963 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.996 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.922 | 0.973 | 0.952 |
| 2 | 257 | 0.965 | 0.991 | 0.979 |
| 3 | 142 | 0.979 | 0.996 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 107 | 22 |
| **Wet** | 20 | 635 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.816 | 0.742 | 0.0993 | 363.4 |
| Day 2 | 105 | 0.850 | 0.823 | 0.0825 | 327.5 |
| Day 3 | 109 | 0.709 | 0.795 | 0.1131 | 3212.1 |
