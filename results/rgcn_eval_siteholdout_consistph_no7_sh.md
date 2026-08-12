# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/best_model_consistph_no7_sh.pt` | predictions: `data/retrain/predictions_consistph_no7_sh`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 613 | 0.79 | 0.941 | 0.991 | 0.964 | 0.763 |
| Day 1 | 205 | 0.79 | 0.937 | 0.991 | 0.961 | 0.744 |
| Day 2 | 207 | 0.78 | 0.937 | 0.990 | 0.961 | 0.756 |
| Day 3 | 201 | 0.79 | 0.950 | 0.992 | 0.969 | 0.791 |
| site 55000900097170 | 131 | 0.60 | 0.863 | 0.993 | 0.895 | 0.679 |
| site 55000900100137 | 114 | 0.32 | 0.842 | 0.921 | 0.780 | 0.821 |
| site 55000900099610 | 116 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 126 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 126 | 1.00 | 1.000 | nan | 1.000 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 177 | 0.75 | 0.915 | 0.978 | 0.944 | 0.778 |
| Day 1 | 60 | 0.75 | 0.917 | 0.982 | 0.945 | 0.800 |
| Day 2 | 60 | 0.75 | 0.917 | 0.972 | 0.945 | 0.800 |
| Day 3 | 57 | 0.74 | 0.912 | 0.978 | 0.943 | 0.733 |
| site 55000900097170 | 36 | 0.36 | 0.778 | 0.973 | 0.750 | 0.696 |
| site 55000900100137 | 34 | 0.35 | 0.794 | 0.837 | 0.696 | 0.864 |
| site 55000900099610 | 35 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.