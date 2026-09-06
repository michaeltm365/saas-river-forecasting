# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_h32.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.981 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.954 | 0.984 | 0.971 |
| Day 1 | Tailwaters (>=3) | 47 | 0.936 | 0.965 | 0.964 |
| Day 2 | All | 264 | 0.951 | 0.976 | 0.970 |
| Day 2 | Headwaters (<=2) | 216 | 0.958 | 0.981 | 0.975 |
| Day 2 | Tailwaters (>=3) | 48 | 0.917 | 0.960 | 0.952 |
| Day 3 | All | 257 | 0.957 | 0.978 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.982 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.915 | 0.965 | 0.951 |
| All Horizons | All | 784 | 0.953 | 0.978 | 0.971 |
| All Horizons | Headwaters (<=2) | 642 | 0.960 | 0.982 | 0.975 |
| All Horizons | Tailwaters (>=3) | 142 | 0.923 | 0.963 | 0.956 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.956 | 0.978 | 0.972 |
| 2 | 257 | 0.965 | 0.993 | 0.979 |
| 3 | 142 | 0.923 | 0.963 | 0.956 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 122 | 7 |
| **Wet** | 30 | 625 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.864 | 0.744 | 0.0855 | 713.9 |
| Day 2 | 105 | 0.880 | 0.811 | 0.0739 | 448.4 |
| Day 3 | 109 | 0.751 | 0.788 | 0.1046 | 2471.5 |
