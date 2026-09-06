# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_h128.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.977 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.976 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 0.936 | 1.000 | 0.964 |
| Day 2 | All | 264 | 0.947 | 0.976 | 0.968 |
| Day 2 | Headwaters (<=2) | 216 | 0.954 | 0.976 | 0.972 |
| Day 2 | Tailwaters (>=3) | 48 | 0.917 | 0.989 | 0.952 |
| Day 3 | All | 257 | 0.957 | 0.979 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.962 | 0.979 | 0.976 |
| Day 3 | Tailwaters (>=3) | 47 | 0.936 | 0.983 | 0.964 |
| All Horizons | All | 784 | 0.952 | 0.977 | 0.971 |
| All Horizons | Headwaters (<=2) | 642 | 0.956 | 0.977 | 0.973 |
| All Horizons | Tailwaters (>=3) | 142 | 0.930 | 0.991 | 0.960 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.943 | 0.974 | 0.964 |
| 2 | 257 | 0.977 | 0.982 | 0.986 |
| 3 | 142 | 0.930 | 0.991 | 0.960 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 117 | 12 |
| **Wet** | 26 | 629 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.853 | 0.744 | 0.0889 | 440.8 |
| Day 2 | 105 | 0.875 | 0.800 | 0.0755 | 293.4 |
| Day 3 | 109 | 0.768 | 0.773 | 0.1009 | 1059.9 |
