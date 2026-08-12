# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_no7_sh.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.935 | 0.969 | 0.961 |
| Day 1 | Headwaters (<=2) | 216 | 0.926 | 0.966 | 0.955 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.947 | 0.979 | 0.969 |
| Day 2 | Headwaters (<=2) | 216 | 0.940 | 0.977 | 0.964 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 1.000 | 0.989 |
| Day 3 | All | 257 | 0.953 | 0.984 | 0.972 |
| Day 3 | Headwaters (<=2) | 210 | 0.948 | 0.982 | 0.968 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.945 | 0.977 | 0.967 |
| All Horizons | Headwaters (<=2) | 642 | 0.938 | 0.975 | 0.962 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 1.000 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.919 | 0.976 | 0.951 |
| 2 | 257 | 0.965 | 0.990 | 0.979 |
| 3 | 142 | 0.979 | 1.000 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 103 | 26 |
| **Wet** | 17 | 638 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.899 | 0.829 | 0.0735 | 543.1 |
| Day 2 | 105 | 0.621 | 0.511 | 0.1313 | 471.6 |
| Day 3 | 109 | 0.910 | 0.820 | 0.0630 | 1225.7 |
