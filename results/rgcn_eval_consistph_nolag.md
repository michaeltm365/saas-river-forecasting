# RGCN retrain — evaluation report

Checkpoint: `data/retrain/best_model_consistph_nolag.pt` | split: `window_split_map_consistph.csv`

## Wet/dry classification (val split)

| Horizon | Group | N | Accuracy | ROC-AUC | F1 |
|---|---|--:|--:|--:|--:|
| Day 1 | All | 263 | 0.532 | 0.625 | 0.648 |
| Day 1 | Headwaters (<=2) | 216 | 0.519 | 0.592 | 0.631 |
| Day 1 | Tailwaters (>=3) | 47 | 0.596 | 1.000 | 0.716 |
| Day 2 | All | 264 | 0.519 | 0.608 | 0.640 |
| Day 2 | Headwaters (<=2) | 216 | 0.505 | 0.572 | 0.625 |
| Day 2 | Tailwaters (>=3) | 48 | 0.583 | 1.000 | 0.706 |
| Day 3 | All | 257 | 0.537 | 0.637 | 0.653 |
| Day 3 | Headwaters (<=2) | 210 | 0.524 | 0.605 | 0.638 |
| Day 3 | Tailwaters (>=3) | 47 | 0.596 | 1.000 | 0.716 |
| All Horizons | All | 784 | 0.529 | 0.624 | 0.647 |
| All Horizons | Headwaters (<=2) | 642 | 0.516 | 0.590 | 0.631 |
| All Horizons | Tailwaters (>=3) | 142 | 0.592 | 1.000 | 0.713 |

## Stream-order breakdown (All Horizons, val)

| Order | N | Accuracy | ROC-AUC | F1 |
|--:|--:|--:|--:|--:|
| 1 | 385 | 0.506 | 0.576 | 0.625 |
| 2 | 257 | 0.529 | 0.632 | 0.641 |
| 3 | 142 | 0.592 | 1.000 | 0.713 |

## Confusion matrix (All Horizons, val)

| Observed \ Pred | Dry | Wet |
|---|--:|--:|
| **Dry** | 77 | 52 |
| **Wet** | 317 | 338 |

## Discharge regression (val split, linear CMS)

| Horizon | N | NSE | KGE | RMSE | MAPE% |
|---|--:|--:|--:|--:|--:|
| Day 1 | 107 | 0.885 | 0.801 | 0.0784 | 473.6 |
| Day 2 | 105 | 0.878 | 0.926 | 0.0746 | 434.5 |
| Day 3 | 109 | 0.872 | 0.789 | 0.0751 | 530.9 |
