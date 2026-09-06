# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/flagship/best_model_flag_sh.pt` | predictions: `data/retrain/flagship/predictions_flag_sh`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 1,836 | 0.79 | 0.958 | 0.984 | 0.973 | 0.897 |
| Day 1 | 611 | 0.79 | 0.962 | 0.991 | 0.976 | 0.907 |
| Day 2 | 612 | 0.79 | 0.959 | 0.985 | 0.974 | 0.900 |
| Day 3 | 613 | 0.79 | 0.951 | 0.978 | 0.969 | 0.885 |
| site 55000900097170 | 390 | 0.60 | 0.972 | 0.989 | 0.976 | 0.968 |
| site 55000900100137 | 342 | 0.32 | 0.845 | 0.920 | 0.773 | 0.850 |
| site 55000900099610 | 348 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 378 | 1.00 | 0.979 | nan | 0.989 | nan |
| site 55000900271029 | 378 | 1.00 | 0.984 | nan | 0.992 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 531 | 0.75 | 0.949 | 0.966 | 0.966 | 0.911 |
| Day 1 | 177 | 0.75 | 0.955 | 0.979 | 0.969 | 0.933 |
| Day 2 | 177 | 0.75 | 0.949 | 0.969 | 0.966 | 0.911 |
| Day 3 | 177 | 0.75 | 0.944 | 0.951 | 0.962 | 0.889 |
| site 55000900097170 | 108 | 0.36 | 0.963 | 0.962 | 0.947 | 0.986 |
| site 55000900100137 | 102 | 0.35 | 0.775 | 0.839 | 0.676 | 0.833 |
| site 55000900099610 | 105 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 108 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 108 | 1.00 | 1.000 | nan | 1.000 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.