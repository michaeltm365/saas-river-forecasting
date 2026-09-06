# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/flagship/best_model_flag_sh_s44.pt` | predictions: `data/retrain/flagship/predictions_flag_sh_s44`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 1,836 | 0.79 | 0.916 | 0.965 | 0.946 | 0.844 |
| Day 1 | 611 | 0.79 | 0.918 | 0.967 | 0.948 | 0.845 |
| Day 2 | 612 | 0.79 | 0.918 | 0.965 | 0.948 | 0.846 |
| Day 3 | 613 | 0.79 | 0.912 | 0.963 | 0.943 | 0.840 |
| site 55000900097170 | 390 | 0.60 | 0.951 | 0.988 | 0.960 | 0.917 |
| site 55000900100137 | 342 | 0.32 | 0.825 | 0.910 | 0.762 | 0.795 |
| site 55000900099610 | 348 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 378 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 378 | 1.00 | 0.802 | nan | 0.890 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 531 | 0.75 | 0.879 | 0.955 | 0.914 | 0.933 |
| Day 1 | 177 | 0.75 | 0.881 | 0.960 | 0.916 | 0.933 |
| Day 2 | 177 | 0.75 | 0.881 | 0.956 | 0.916 | 0.933 |
| Day 3 | 177 | 0.75 | 0.876 | 0.948 | 0.911 | 0.933 |
| site 55000900097170 | 108 | 0.36 | 0.972 | 0.941 | 0.960 | 1.000 |
| site 55000900100137 | 102 | 0.35 | 0.794 | 0.842 | 0.696 | 0.864 |
| site 55000900099610 | 105 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 108 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 108 | 1.00 | 0.630 | nan | 0.773 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.