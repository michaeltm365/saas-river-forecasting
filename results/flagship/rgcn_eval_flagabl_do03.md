# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_do03.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.958 | 0.983 | 0.975 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.980 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.955 | 0.980 | 0.973 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.978 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.958 | 1.000 | 0.977 |
| Day 3 | All | 257 | 0.965 | 0.983 | 0.979 |
| Day 3 | Headwaters (<=2) | 210 | 0.962 | 0.982 | 0.976 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| All Horizons | All | 784 | 0.959 | 0.982 | 0.975 |
| All Horizons | Headwaters (<=2) | 642 | 0.956 | 0.980 | 0.973 |
| All Horizons | Tailwaters (>=3) | 142 | 0.972 | 1.000 | 0.984 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.943 | 0.975 | 0.964 |
| 2 | 257 | 0.977 | 0.989 | 0.986 |
| 3 | 142 | 0.972 | 1.000 | 0.984 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 117 | 12 |
| **Wet** | 20 | 635 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.850 | 0.708 | 0.0898 | 495.5 |
| Day 2 | 105 | 0.860 | 0.762 | 0.0799 | 319.5 |
| Day 3 | 109 | 0.759 | 0.741 | 0.1029 | 1566.8 |
