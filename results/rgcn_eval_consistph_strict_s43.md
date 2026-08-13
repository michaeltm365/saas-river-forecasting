# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_strict_s43.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.947 | 0.964 | 0.968 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.963 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 0.957 | 1.000 | 0.976 |
| Day 2 | All | 264 | 0.947 | 0.964 | 0.968 |
| Day 2 | Headwaters (<=2) | 216 | 0.949 | 0.963 | 0.969 |
| Day 2 | Tailwaters (>=3) | 48 | 0.938 | 0.983 | 0.965 |
| Day 3 | All | 257 | 0.949 | 0.970 | 0.969 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.970 | 0.971 |
| Day 3 | Tailwaters (>=3) | 47 | 0.936 | 0.977 | 0.964 |
| All Horizons | All | 784 | 0.948 | 0.966 | 0.968 |
| All Horizons | Headwaters (<=2) | 642 | 0.949 | 0.966 | 0.968 |
| All Horizons | Tailwaters (>=3) | 142 | 0.944 | 0.987 | 0.968 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.938 | 0.971 | 0.961 |
| 2 | 257 | 0.965 | 0.978 | 0.979 |
| 3 | 142 | 0.944 | 0.987 | 0.968 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 115 | 14 |
| **Wet** | 27 | 628 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.815 | 0.667 | 0.0996 | 484.1 |
| Day 2 | 105 | 0.849 | 0.737 | 0.0828 | 363.9 |
| Day 3 | 109 | 0.744 | 0.717 | 0.1061 | 767.6 |
