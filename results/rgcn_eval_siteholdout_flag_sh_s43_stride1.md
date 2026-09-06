# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/flagship/best_model_flag_sh_s43.pt` | predictions: `data/retrain/flagship/predictions_flag_sh_s43`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 1,836 | 0.79 | 0.959 | 0.987 | 0.974 | 0.946 |
| Day 1 | 611 | 0.79 | 0.961 | 0.989 | 0.975 | 0.946 |
| Day 2 | 612 | 0.79 | 0.961 | 0.987 | 0.975 | 0.946 |
| Day 3 | 613 | 0.79 | 0.956 | 0.985 | 0.972 | 0.947 |
| site 55000900097170 | 390 | 0.60 | 0.967 | 0.986 | 0.972 | 0.981 |
| site 55000900100137 | 342 | 0.32 | 0.889 | 0.922 | 0.822 | 0.923 |
| site 55000900099610 | 348 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 378 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 378 | 1.00 | 0.937 | nan | 0.967 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 531 | 0.75 | 0.934 | 0.966 | 0.955 | 0.933 |
| Day 1 | 177 | 0.75 | 0.938 | 0.972 | 0.958 | 0.933 |
| Day 2 | 177 | 0.75 | 0.932 | 0.966 | 0.953 | 0.933 |
| Day 3 | 177 | 0.75 | 0.932 | 0.960 | 0.953 | 0.933 |
| site 55000900097170 | 108 | 0.36 | 0.972 | 0.952 | 0.960 | 1.000 |
| site 55000900100137 | 102 | 0.35 | 0.794 | 0.834 | 0.696 | 0.864 |
| site 55000900099610 | 105 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 108 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 108 | 1.00 | 0.898 | nan | 0.946 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.