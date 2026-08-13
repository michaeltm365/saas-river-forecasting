# RGCN with-sensor site-holdout evaluation

Checkpoint: `data/retrain/best_model_consistph_strict_no7_sh.pt` | predictions: `data/retrain/predictions_consistph_strict_no7_sh`

Held-out reaches (labels masked from training loss; obs-lag inputs intact): [55000900097170, 55000900100137, 55000900099610, 55000900235848, 55000900271029]

## All labeled dates

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 613 | 0.79 | 0.946 | 0.980 | 0.966 | 0.847 |
| Day 3 | 613 | 0.79 | 0.946 | 0.980 | 0.966 | 0.847 |
| site 55000900097170 | 131 | 0.60 | 0.954 | 0.983 | 0.962 | 0.925 |
| site 55000900100137 | 114 | 0.32 | 0.816 | 0.890 | 0.747 | 0.795 |
| site 55000900099610 | 116 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 126 | 1.00 | 0.976 | nan | 0.988 | nan |
| site 55000900271029 | 126 | 1.00 | 0.976 | nan | 0.988 | nan |

## Val-block dates only

| Slice | N | Wet frac | Accuracy | ROC-AUC | F1 | Dry recall |
|---|--:|--:|--:|--:|--:|--:|
| Pooled (all horizons) | 177 | 0.75 | 0.938 | 0.952 | 0.958 | 0.867 |
| Day 3 | 177 | 0.75 | 0.938 | 0.952 | 0.958 | 0.867 |
| site 55000900097170 | 36 | 0.36 | 0.944 | 0.933 | 0.923 | 0.957 |
| site 55000900100137 | 34 | 0.35 | 0.735 | 0.769 | 0.640 | 0.773 |
| site 55000900099610 | 35 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900235848 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |
| site 55000900271029 | 36 | 1.00 | 1.000 | nan | 1.000 | nan |

Context: 'All labeled dates' overlaps the training period of *other* sites (standard site-based-split semantics — same era, new site). 'Val-block dates' additionally avoids any temporal overlap with training targets.