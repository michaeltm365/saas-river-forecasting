# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_nolag_cls.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.734 | 0.799 | 0.823 |
| Day 1 | Headwaters (<=2) | 216 | 0.699 | 0.767 | 0.794 |
| Day 1 | Tailwaters (>=3) | 47 | 0.894 | 1.000 | 0.938 |
| Day 2 | All | 264 | 0.708 | 0.789 | 0.807 |
| Day 2 | Headwaters (<=2) | 216 | 0.676 | 0.756 | 0.780 |
| Day 2 | Tailwaters (>=3) | 48 | 0.854 | 1.000 | 0.914 |
| Day 3 | All | 257 | 0.720 | 0.807 | 0.813 |
| Day 3 | Headwaters (<=2) | 210 | 0.695 | 0.779 | 0.792 |
| Day 3 | Tailwaters (>=3) | 47 | 0.830 | 1.000 | 0.897 |
| All Horizons | All | 784 | 0.721 | 0.798 | 0.815 |
| All Horizons | Headwaters (<=2) | 642 | 0.690 | 0.768 | 0.789 |
| All Horizons | Tailwaters (>=3) | 142 | 0.859 | 1.000 | 0.917 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.662 | 0.709 | 0.774 |
| 2 | 257 | 0.732 | 0.891 | 0.811 |
| 3 | 142 | 0.859 | 1.000 | 0.917 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 84 | 45 |
| **Wet** | 174 | 481 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.825 | 0.816 | 0.0968 | 404.2 |
| Day 2 | 105 | 0.811 | 0.886 | 0.0927 | 383.9 |
| Day 3 | 109 | 0.790 | 0.805 | 0.0960 | 448.5 |
