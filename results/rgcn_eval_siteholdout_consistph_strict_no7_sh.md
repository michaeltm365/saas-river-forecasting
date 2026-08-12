# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/best_model_consistph_strict_no7_sh.pt` | predictions: `data/retrain/predictions_consistph_strict_no7_sh`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 613 | 0.79 | 0.951 | 0.989 | 0.969 | 0.885 |
| Day 1 | 205 | 0.79 | 0.941 | 0.985 | 0.963 | 0.860 |
| Day 2 | 207 | 0.78 | 0.952 | 0.990 | 0.969 | 0.889 |
| Day 3 | 201 | 0.79 | 0.960 | 0.992 | 0.975 | 0.907 |
| site 55000900097170 | 131 | 0.60 | 0.977 | 0.987 | 0.981 | 0.981 |
| site 55000900100137 | 114 | 0.32 | 0.816 | 0.920 | 0.734 | 0.821 |
| site 55000900099610 | 116 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 126 | 1.00 | 0.976 | nan | 0.988 | nan |
| site 55000900271029 | 126 | 1.00 | 0.976 | nan | 0.988 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 177 | 0.75 | 0.955 | 0.973 | 0.969 | 0.933 |
| Day 1 | 60 | 0.75 | 0.950 | 0.967 | 0.966 | 0.933 |
| Day 2 | 60 | 0.75 | 0.950 | 0.975 | 0.966 | 0.933 |
| Day 3 | 57 | 0.74 | 0.965 | 0.978 | 0.976 | 0.933 |
| site 55000900097170 | 36 | 0.36 | 0.972 | 0.933 | 0.960 | 1.000 |
| site 55000900100137 | 34 | 0.35 | 0.794 | 0.818 | 0.696 | 0.864 |
| site 55000900099610 | 35 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.