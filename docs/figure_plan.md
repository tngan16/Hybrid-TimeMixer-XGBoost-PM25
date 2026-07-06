# Figure plan

Every quantitative chart is generated from saved evidence by
`scripts/make_figures.py`; it is not an AI-generated illustration.

## Main text: eight figures

1. `01_data_overview.png` — timeline, distribution, diurnal pattern, and ACF.
2. `02_data_quality.png` — coverage, missingness, gaps, and longest outage.
3. `03_multistation_effectiveness.png` — hybrid gain over the strongest matched
   non-hybrid comparator by station and horizon.
4. `04_model_metrics.png` — MAE/RMSE across models and horizons.
5. `05_forecast_diagnostics.png` — representative trace and residual behavior.
6. `06_residual_ablation.png` — base, robust correction, rolling calibration.
7. `07_seed_variability.png` — variation across seeds 0–4.
8. `08_bootstrap_forest.png` — paired moving-block bootstrap intervals.

The original nine single-purpose EDA plots are retained as `S01`–`S09` under
`figures/supplementary/`. Horizon-level SHAP detail belongs in the supplement.

The conceptual architecture is the only manually drawn figure. Edit
`paper/figures/00_framework.drawio`, verify every arrow against the code, and
export it as `paper/figures/00_framework.pdf`.
