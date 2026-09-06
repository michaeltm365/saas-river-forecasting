# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_sh_s44.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.947 | 0.975 | 0.967 |
| Day 1 | Headwaters (<=2) | 216 | 0.963 | 0.976 | 0.977 |
| Day 1 | Tailwaters (>=3) | 47 | 0.872 | 1.000 | 0.925 |
| Day 2 | All | 264 | 0.943 | 0.973 | 0.966 |
| Day 2 | Headwaters (<=2) | 216 | 0.963 | 0.975 | 0.977 |
| Day 2 | Tailwaters (>=3) | 48 | 0.854 | 0.983 | 0.914 |
| Day 3 | All | 257 | 0.953 | 0.975 | 0.971 |
| Day 3 | Headwaters (<=2) | 210 | 0.971 | 0.977 | 0.982 |
| Day 3 | Tailwaters (>=3) | 47 | 0.872 | 0.983 | 0.925 |
| All Horizons | All | 784 | 0.948 | 0.974 | 0.968 |
| All Horizons | Headwaters (<=2) | 642 | 0.966 | 0.976 | 0.979 |
| All Horizons | Tailwaters (>=3) | 142 | 0.866 | 0.988 | 0.921 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.958 | 0.975 | 0.974 |
| 2 | 257 | 0.977 | 0.980 | 0.986 |
| 3 | 142 | 0.866 | 0.988 | 0.921 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 123 | 6 |
| **Wet** | 35 | 620 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.855 | 0.734 | 0.0883 | 329.4 |
| Day 2 | 105 | 0.867 | 0.789 | 0.0779 | 215.7 |
| Day 3 | 109 | 0.761 | 0.760 | 0.1024 | 672.0 |
