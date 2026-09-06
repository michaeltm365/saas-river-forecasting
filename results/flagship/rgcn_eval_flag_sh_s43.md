# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flag_sh_s43.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.958 | 0.985 | 0.974 |
| Day 1 | Headwaters (<=2) | 216 | 0.963 | 0.983 | 0.977 |
| Day 1 | Tailwaters (>=3) | 47 | 0.936 | 1.000 | 0.964 |
| Day 2 | All | 264 | 0.955 | 0.981 | 0.972 |
| Day 2 | Headwaters (<=2) | 216 | 0.963 | 0.981 | 0.977 |
| Day 2 | Tailwaters (>=3) | 48 | 0.917 | 0.994 | 0.952 |
| Day 3 | All | 257 | 0.961 | 0.984 | 0.976 |
| Day 3 | Headwaters (<=2) | 210 | 0.967 | 0.984 | 0.979 |
| Day 3 | Tailwaters (>=3) | 47 | 0.936 | 0.994 | 0.964 |
| All Horizons | All | 784 | 0.958 | 0.983 | 0.974 |
| All Horizons | Headwaters (<=2) | 642 | 0.964 | 0.983 | 0.978 |
| All Horizons | Tailwaters (>=3) | 142 | 0.930 | 0.996 | 0.960 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.964 | 0.978 | 0.977 |
| 2 | 257 | 0.965 | 0.991 | 0.979 |
| 3 | 142 | 0.930 | 0.996 | 0.960 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 125 | 4 |
| **Wet** | 29 | 626 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.835 | 0.703 | 0.0940 | 707.1 |
| Day 2 | 105 | 0.865 | 0.767 | 0.0783 | 590.7 |
| Day 3 | 109 | 0.761 | 0.745 | 0.1024 | 1532.3 |
