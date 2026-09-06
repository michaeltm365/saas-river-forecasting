# Annual dry-day estimation — flagship RGCN, Gaussian copula, all splits

Seed-42 stride-1 day-3 predictions per split; ρ = genuine lag-1 autocorrelation of the raw daily HOBO label series on the split's TRAINING dates (consecutive-day pairs only). p_dry bootstrapped from the split's val-day predictions; observed dry d/yr = (1 - val wet fraction) x 365; 10,000 sims/site.

Caveats: ph val = 36 days across three phases; q65/q80 val = late season only (dry-biased 'typical year'); 'site' scores the flag_sh model at reaches whose labels were masked from its loss (with-sensor spatial regime).

## ph  (top-8 by val label count (NOTE: 13 sites tied at 36 labels — selection arbitrary among ties))

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900097169 | 36 | 1.000 | 62 | 304.2 | 315.8 | [164.0, 365.0] | yes |
| 55000900097170 | 36 | 0.960 | 72 | 233.2 | 225.8 | [186.0, 257.0] | yes |
| 55000900030173 | 36 | 0.000 | 65 | 0.0 | 8.3 | [3.0, 14.0] | NO |
| 55000900201811 | 36 | 0.000 | 72 | 0.0 | 7.1 | [2.0, 13.0] | NO |
| 55000900131558 | 36 | 0.000 | 67 | 0.0 | 24.8 | [16.0, 35.0] | NO |
| 55000900167054 | 36 | 0.000 | 61 | 0.0 | 10.2 | [5.0, 17.0] | NO |
| 55000900272714 | 36 | 0.000 | 66 | 0.0 | 6.8 | [2.0, 12.0] | NO |
| 55000900271635 | 36 | 0.000 | 65 | 0.0 | 36.9 | [26.0, 49.0] | NO |

Coverage: 2/8 (25%).

## q65  (top-8 by val label count)

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900097170 | 48 | 1.000 | 81 | 349.8 | 211.8 | [0.0, 365.0] | yes |
| 55000900237668 | 48 | 0.927 | 64 | 319.4 | 326.1 | [293.0, 350.0] | yes |
| 55000900167704 | 52 | 0.000 | 84 | 77.2 | 66.1 | [52.0, 81.0] | yes |
| 55000900029021 | 43 | 0.000 | 84 | 25.5 | 42.8 | [31.0, 55.0] | NO |
| 55000900202948 | 43 | 0.000 | 77 | 0.0 | 6.4 | [2.0, 12.0] | NO |
| 55000900030173 | 43 | 0.000 | 77 | 0.0 | 8.2 | [3.0, 14.0] | NO |
| 55000900201811 | 43 | 0.000 | 84 | 0.0 | 6.2 | [2.0, 11.0] | NO |
| 55000900131558 | 42 | 0.000 | 79 | 0.0 | 25.2 | [16.0, 35.0] | NO |

Coverage: 3/8 (38%).

## q80  (top-8 by val label count)

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900097170 | 31 | 1.000 | 97 | 341.5 | 318.3 | [132.0, 365.0] | yes |
| 55000900237668 | 30 | 0.929 | 82 | 292.0 | 312.9 | [273.0, 343.0] | yes |
| 55000900167704 | 34 | 0.000 | 102 | 118.1 | 95.9 | [80.0, 113.0] | NO |
| 55000900029021 | 25 | 0.000 | 102 | 43.8 | 60.0 | [46.0, 74.0] | NO |
| 55000900202948 | 25 | 0.000 | 95 | 0.0 | 6.3 | [2.0, 12.0] | NO |
| 55000900030173 | 25 | 0.000 | 95 | 0.0 | 8.9 | [4.0, 15.0] | NO |
| 55000900201811 | 25 | 0.000 | 102 | 0.0 | 7.1 | [2.0, 13.0] | NO |
| 55000900131558 | 24 | 0.000 | 97 | 0.0 | 25.4 | [16.0, 35.0] | NO |

Coverage: 2/8 (25%).

## site  (all 5 holdout reaches)

| Site | N val | ρ (lag-1) | ρ pairs | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|--:|---|:--:|
| 55000900100137 | 114 | 1.000 | 59 | 249.7 | 218.6 | [45.0, 365.0] | yes |
| 55000900097170 | 131 | 0.960 | 72 | 147.7 | 146.9 | [110.0, 188.0] | yes |
| 55000900099610 | 116 | 0.000 | 67 | 0.0 | 13.1 | [7.0, 20.0] | NO |
| 55000900235848 | 126 | 0.000 | 72 | 0.0 | 26.0 | [17.0, 36.0] | NO |
| 55000900271029 | 126 | 0.000 | 72 | 0.0 | 27.1 | [18.0, 38.0] | NO |

Coverage: 2/5 (40%).

