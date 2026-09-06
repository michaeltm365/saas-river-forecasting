# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_do00.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.951 | 0.982 | 0.970 |
| Day 1 | Headwaters (<=2) | 216 | 0.944 | 0.980 | 0.966 |
| Day 1 | Tailwaters (>=3) | 47 | 0.979 | 1.000 | 0.988 |
| Day 2 | All | 264 | 0.955 | 0.980 | 0.973 |
| Day 2 | Headwaters (<=2) | 216 | 0.949 | 0.978 | 0.969 |
| Day 2 | Tailwaters (>=3) | 48 | 0.979 | 0.994 | 0.989 |
| Day 3 | All | 257 | 0.957 | 0.983 | 0.974 |
| Day 3 | Headwaters (<=2) | 210 | 0.952 | 0.981 | 0.971 |
| Day 3 | Tailwaters (>=3) | 47 | 0.979 | 0.988 | 0.988 |
| All Horizons | All | 784 | 0.954 | 0.982 | 0.972 |
| All Horizons | Headwaters (<=2) | 642 | 0.949 | 0.979 | 0.968 |
| All Horizons | Tailwaters (>=3) | 142 | 0.979 | 0.995 | 0.988 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.938 | 0.976 | 0.961 |
| 2 | 257 | 0.965 | 0.988 | 0.979 |
| 3 | 142 | 0.979 | 0.995 | 0.988 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 115 | 14 |
| **Wet** | 22 | 633 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.891 | 0.772 | 0.0766 | 1086.1 |
| Day 2 | 105 | 0.896 | 0.857 | 0.0686 | 989.7 |
| Day 3 | 109 | 0.759 | 0.828 | 0.1028 | 2684.1 |
