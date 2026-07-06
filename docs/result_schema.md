# Result artefact schema

All result files are generated; a clean clone may contain only this documentation.

| File | Unit of observation | Key fields |
|---|---|---|
| cleaning_audit.json | one run | raw issue counts, grid and gap totals |
| annual_coverage.csv | year | expected, observed, missing hours, coverage |
| gap_summary.csv | missing run | start, end, duration, class, inserted rows |
| split_report.json | temporal partition | dates, rows, valid count, coverage |
| run_config.json | experiment | all hyperparameters, features, window counts |
| experiment_metrics.csv | model × horizon | n, MAE, RMSE, sMAPE, R², Bias |
| predictions.csv | test origin | origin/target times, truth and each model forecast |
| high_pollution_metrics.csv | model × horizon | threshold, n_high, high MAE, recall, false alarms |
| bootstrap_comparison.csv | horizon | mean paired MAE difference and 95% interval |
| diebold_mariano.csv | horizon | statistic, raw p and Holm-adjusted p |
| ablation_metrics.csv | variant × horizon | point metrics |
| shap_summary.csv | hybrid feature × horizon | mean absolute SHAP contribution |
| seed_metrics.csv | seed × model × horizon | repeat-run performance |
| *_history.csv | epoch | training and validation loss |

## Multi-station confirmatory outputs

| File | Unit of observation | Purpose |
|---|---|---|
| multistation_seed_metrics.csv | station × seed × model × horizon | complete confirmatory scores |
| multistation_summary.csv | station × model × horizon | five-seed mean and standard deviation |
| station_effectiveness.csv | station × horizon | hybrid versus strongest non-hybrid comparator |
| rolling_calibration_diagnostics.csv | origin × horizon | available past errors, bias, shrinkage |

Each station/seed `predictions.csv` is the canonical evidence table for
rechecking metrics. Cross-station claims must come from the multi-station
tables, not from a single station folder.

`predictions.csv` is the canonical evidence table for rechecking metrics. Target times are stored separately for every horizon to avoid ambiguity. Negative `mean_diff` in the bootstrap file indicates lower hybrid absolute error than TimeMixer.

## Common-history sensitivity outputs

- `common_2021_multistation_seed_metrics.csv`: station, seed, model, and
  horizon scores after restricting all training histories to 2021--2022.
- `history_sensitivity_comparison.csv`: common-minus-available-history changes
  in MAE, RMSE, sMAPE, R2, and bias.
- `history_sensitivity_hybrid.csv`: the hybrid-only subset used in the
  manuscript sensitivity analysis.
