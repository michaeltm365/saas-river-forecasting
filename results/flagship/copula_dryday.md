# Annual dry-day estimation — flagship RGCN, Gaussian copula (AR1)

Flagship (A-strict no-lag-7) phases predictions, STRIDE-1 day-3 export (true daily grid), seed 42. Fixes the released ρ-estimation bug: ρ is now the genuine lag-1 autocorrelation (consecutive calendar days only; the stride-3 export computed lag-3 mislabeled as lag-1). p_dry bootstrapped from the 36 val-block days (3 hydrologic phases); observed dry days/yr = (1 - val wet fraction) x 365. 10,000 sims/site, top-8 HOBO reaches by val label count.

| Site | N val | ρ (lag-1) | True dry d/yr | Mean pred | 95% CI | In CI |
|---|--:|--:|--:|--:|---|:--:|
| 55000900097169 | 36 | 1.000 | 304.2 | 315.8 | [164.0, 365.0] | yes |
| 55000900097170 | 36 | 0.960 | 233.2 | 225.8 | [186.0, 257.0] | yes |
| 55000900030173 | 36 | 0.000 | 0.0 | 8.3 | [3.0, 14.0] | NO |
| 55000900201811 | 36 | 0.000 | 0.0 | 7.1 | [2.0, 13.0] | NO |
| 55000900131558 | 36 | 0.000 | 0.0 | 24.8 | [16.0, 35.0] | NO |
| 55000900167054 | 36 | 0.000 | 0.0 | 10.2 | [5.0, 17.0] | NO |
| 55000900272714 | 36 | 0.000 | 0.0 | 6.8 | [2.0, 12.0] | NO |
| 55000900271635 | 36 | 0.000 | 0.0 | 36.9 | [26.0, 49.0] | NO |

Coverage: 2/8 sites (25%) within the 95% interval.

Caveat: the val set is three 12-day blocks, so the bootstrap treats 36 days spanning drying/peak-dry/rewetting as a typical year; the wet-fraction x 365 'observed' value shares the same approximation.
