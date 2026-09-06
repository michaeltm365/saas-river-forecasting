# Annual dry-day estimation — flagship RGCN, Gaussian copula

**Canonical (q65)**: Platt-calibrated Day-3 dry probabilities (calibrator fit on training-period predictions) with a ρ ceiling of 0.98, scored on all HOBO reaches with ≥20 q65 validation labels. Raw-method coverage on the same reaches is reported for comparison. ph/q80/site sections use the raw uncalibrated method (diagnostic; original site selections). ρ = lag-1 autocorrelation from consecutive-day pairs of the raw daily HOBO series over each split's training dates; each split scored with its own seed-42 stride-1 Day-3 export; 10,000 sims/site with per-site seeds.

## q65 — CANONICAL (Platt + ρ-clip; all 22 HOBO val reaches)

| Site | N val | ρ (lag-1) | ρ used | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|--:|---|:--:|
| 55000900097170 | 48 | 1.000 | 0.980 | 81 | 349.8 | 169.8 | [85.0, 254.0] | NO |
| 55000900100137 | 40 | 1.000 | 0.980 | 69 | 328.5 | 346.5 | [284.0, 365.0] | yes |
| 55000900237668 | 48 | 0.927 | 0.927 | 64 | 319.4 | 319.1 | [287.0, 341.0] | yes |
| 55000900097169 | 42 | 1.000 | 0.980 | 74 | 304.2 | 333.2 | [271.0, 359.0] | yes |
| 55000900097164 | 41 | 1.000 | 0.980 | 83 | 258.2 | 265.5 | [203.0, 299.0] | yes |
| 55000900167704 | 52 | 0.000 | 0.000 | 84 | 77.2 | 55.8 | [43.0, 70.0] | NO |
| 55000900029021 | 43 | 0.000 | 0.000 | 84 | 25.5 | 26.5 | [17.0, 37.0] | yes |
| 55000900096564 | 36 | 0.000 | 0.000 | 66 | 10.1 | 4.0 | [1.0, 8.0] | NO |
| 55000900202948 | 43 | 0.000 | 0.000 | 77 | 0.0 | 1.3 | [0.0, 4.0] | yes |
| 55000900030173 | 43 | 0.000 | 0.000 | 77 | 0.0 | 1.4 | [0.0, 4.0] | yes |
| 55000900201811 | 43 | 0.000 | 0.000 | 84 | 0.0 | 1.3 | [0.0, 4.0] | yes |
| 55000900131558 | 42 | 0.000 | 0.000 | 79 | 0.0 | 2.1 | [0.0, 5.0] | yes |
| 55000900271635 | 42 | 0.000 | 0.000 | 77 | 0.0 | 2.7 | [0.0, 6.0] | yes |
| 55000900272714 | 41 | 0.000 | 0.000 | 78 | 0.0 | 1.3 | [0.0, 4.0] | yes |
| 55000900201800 | 41 | 0.000 | 0.000 | 74 | 0.0 | 1.4 | [0.0, 4.0] | yes |
| 55000900167054 | 41 | 0.000 | 0.000 | 73 | 0.0 | 1.5 | [0.0, 4.0] | yes |
| 55000900235848 | 41 | 0.000 | 0.000 | 84 | 0.0 | 1.4 | [0.0, 4.0] | yes |
| 55000900271029 | 41 | 0.000 | 0.000 | 84 | 0.0 | 1.6 | [0.0, 4.0] | yes |
| 55000900236525 | 40 | 0.000 | 0.000 | 84 | 0.0 | 2.0 | [0.0, 5.0] | yes |
| 55000900272209 | 40 | 0.000 | 0.000 | 80 | 0.0 | 2.1 | [0.0, 5.0] | yes |
| 55000900099610 | 39 | 0.000 | 0.000 | 75 | 0.0 | 1.4 | [0.0, 4.0] | yes |
| 55000900167692 | 35 | 0.000 | 0.000 | 80 | 0.0 | 15.3 | [8.0, 23.0] | NO |

Coverage: **18/22 (82%)**; mean 95% CI width 27.9 days.
Raw method on the same 22 reaches: 6/22 (27%) coverage, mean CI width 58.9 days — the improvement is almost entirely the probability calibration (mean perennial-reach p_dry drops from a few percent to <1%); the ρ ceiling keeps intervals at ρ≈1.0 reaches informative.

## ph — raw method, diagnostic  (top-8 by val label count (NOTE: 13 sites tied at 36 labels — selection arbitrary among ties))

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900097169 | 36 | 1.000 | 62 | 304.2 | 315.0 | [157.0, 365.0] | yes |
| 55000900097170 | 36 | 0.960 | 72 | 233.2 | 226.0 | [187.0, 257.0] | yes |
| 55000900030173 | 36 | 0.000 | 65 | 0.0 | 8.4 | [3.0, 14.0] | NO |
| 55000900201811 | 36 | 0.000 | 72 | 0.0 | 7.1 | [3.0, 13.0] | NO |
| 55000900131558 | 36 | 0.000 | 67 | 0.0 | 25.0 | [16.0, 35.0] | NO |
| 55000900167054 | 36 | 0.000 | 61 | 0.0 | 10.2 | [5.0, 17.0] | NO |
| 55000900272714 | 36 | 0.000 | 66 | 0.0 | 6.8 | [2.0, 12.0] | NO |
| 55000900271635 | 36 | 0.000 | 65 | 0.0 | 36.8 | [26.0, 48.0] | NO |

Coverage: 2/8 (25%).

## q80 — raw method, diagnostic  (top-8 by val label count)

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900097170 | 31 | 1.000 | 97 | 341.5 | 318.8 | [132.0, 365.0] | yes |
| 55000900237668 | 30 | 0.929 | 82 | 292.0 | 312.7 | [273.0, 343.0] | yes |
| 55000900167704 | 34 | 0.000 | 102 | 118.1 | 95.8 | [80.0, 112.0] | NO |
| 55000900029021 | 25 | 0.000 | 102 | 43.8 | 59.8 | [46.0, 74.0] | NO |
| 55000900202948 | 25 | 0.000 | 95 | 0.0 | 6.3 | [2.0, 12.0] | NO |
| 55000900030173 | 25 | 0.000 | 95 | 0.0 | 9.0 | [4.0, 15.0] | NO |
| 55000900201811 | 25 | 0.000 | 102 | 0.0 | 7.1 | [2.0, 13.0] | NO |
| 55000900131558 | 24 | 0.000 | 97 | 0.0 | 25.5 | [16.0, 35.0] | NO |

Coverage: 2/8 (25%).

## site — raw method, diagnostic  (all 5 holdout reaches)

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900100137 | 114 | 1.000 | 59 | 249.7 | 219.9 | [52.0, 365.0] | yes |
| 55000900097170 | 131 | 0.960 | 72 | 147.7 | 146.7 | [110.0, 187.0] | yes |
| 55000900099610 | 116 | 0.000 | 67 | 0.0 | 13.1 | [7.0, 20.0] | NO |
| 55000900235848 | 126 | 0.000 | 72 | 0.0 | 26.0 | [17.0, 36.0] | NO |
| 55000900271029 | 126 | 0.000 | 72 | 0.0 | 27.1 | [17.0, 37.0] | NO |

Coverage: 2/5 (40%).

Caveats: q65 val is late-season only (dry-biased 'typical year'); the q65 calibration pool is 579 HOBO train rows from a single season (a leave-site-out calibration check is the natural robustness follow-up); 'site' scores the flag_sh model at reaches whose labels were masked from its loss.

