# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_strict_no7_sh.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.947 | 0.971 | 0.968 |
| Day 1 | Headwaters (<=2) | 216 | 0.935 | 0.967 | 0.960 |
| Day 1 | Tailwaters (>=3) | 47 | 1.000 | 1.000 | 1.000 |
| Day 2 | All | 264 | 0.947 | 0.974 | 0.968 |
| Day 2 | Headwaters (<=2) | 216 | 0.940 | 0.972 | 0.964 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.953 | 0.979 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.948 | 0.976 | 0.968 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.994 | 0.988 |
| All Horizons | All | 784 | 0.949 | 0.975 | 0.969 |
| All Horizons | Headwaters (<=2) | 642 | 0.941 | 0.972 | 0.964 |
| All Horizons | Tailwaters (>=3) | 142 | 0.986 | 0.997 | 0.992 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.932 | 0.973 | 0.958 |
| 2 | 257 | 0.953 | 0.990 | 0.972 |
| 3 | 142 | 0.986 | 0.997 | 0.992 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 109 | 20 |
| **Wet** | 20 | 635 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.854 | 0.772 | 0.0886 | 740.3 |
| Day 2 | 105 | 0.868 | 0.837 | 0.0776 | 550.2 |
| Day 3 | 109 | 0.755 | 0.804 | 0.1037 | 2052.8 |
