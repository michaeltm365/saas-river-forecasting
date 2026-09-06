# RGCN retrain — evaluation report

Checkpoint: `data/retrain/flagship/best_model_flagabl_reg_only.pt` | split: `window_split_map_flagph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.829 | 0.500 | 0.906 |
| Day 1 | Headwaters (<=2) | 216 | 0.810 | 0.500 | 0.895 |
| Day 1 | Tailwaters (>=3) | 47 | 0.915 | 0.500 | 0.956 |
| Day 2 | All | 264 | 0.845 | 0.500 | 0.916 |
| Day 2 | Headwaters (<=2) | 216 | 0.829 | 0.500 | 0.906 |
| Day 2 | Tailwaters (>=3) | 48 | 0.917 | 0.500 | 0.957 |
| Day 3 | All | 257 | 0.833 | 0.500 | 0.909 |
| Day 3 | Headwaters (<=2) | 210 | 0.814 | 0.500 | 0.898 |
| Day 3 | Tailwaters (>=3) | 47 | 0.915 | 0.500 | 0.956 |
| All Horizons | All | 784 | 0.835 | 0.500 | 0.910 |
| All Horizons | Headwaters (<=2) | 642 | 0.818 | 0.500 | 0.900 |
| All Horizons | Tailwaters (>=3) | 142 | 0.915 | 0.500 | 0.956 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.800 | 0.500 | 0.889 |
| 2 | 257 | 0.844 | 0.500 | 0.916 |
| 3 | 142 | 0.915 | 0.500 | 0.956 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 0 | 129 |
| **Wet** | 0 | 655 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.852 | 0.739 | 0.0891 | 234.6 |
| Day 2 | 105 | 0.879 | 0.805 | 0.0743 | 209.3 |
| Day 3 | 109 | 0.776 | 0.781 | 0.0991 | 276.2 |
