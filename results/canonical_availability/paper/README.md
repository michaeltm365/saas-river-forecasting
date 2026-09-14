# Canonical raw seed-42 copula

These outputs use the availability RGCN with raw probabilities, no Platt calibration, and no rho cap. The count includes only sensor-observed validation dates and aggregates rolling three-day-ahead forecasts. It is not an annual count or a whole-period forecast issued at its start.

- [Reach-level counts and intervals](raw_seed42_copula.csv)
- [Interval figure](raw_seed42_intervals.pdf)
- [Interval-inclusion map](raw_seed42_map.pdf)

Nominal 95% simulation intervals contain the observed count at 18 of 22 reaches. There are 908 observed dates across reaches. Mean interval width is 5.773 days.

Regenerate from the repository root with `uv run python benchmarks/paper_canonical_report.py`. The main notebook is `rgcn/rgcn_eval.ipynb`. The alternative calibrated implementation remains accessible in `benchmarks/observed_period_copula.py`.
