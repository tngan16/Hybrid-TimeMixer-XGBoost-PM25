# Hybrid TimeMixer–XGBoost for Multi-Horizon PM2.5 Forecasting

## Reviewer revision

This copy preserves the original project while implementing the requested
revision:

- two composite EDA figures in the main paper; nine detailed plots in
  `figures/supplementary/`;
- a manifest-driven multi-station runner;
- median de-biasing, pseudo-Huber residual XGBoost, correction clipping, and
  leakage-safe rolling calibration;
- held-out calibration gating per horizon, including an explicit zero-
  correction fallback when residual learning does not improve calibration MAE;
- origin-safe lag and shifted rolling predictors supplied to the residual
  learner in addition to TimeMixer forecasts and embeddings;
- DLinear as an additional strong simple comparator;
- five seeds (`0 1 2 3 4`) and a 30-epoch maximum;
- code-generated charts and an editable Draw.io architecture source.

The completed confirmatory run does not support a SOTA claim. The gated hybrid
improves TimeMixer at all pooled horizons, but wins 0/25 comparisons against
the strongest matched comparator. See `docs/sota_claim_policy.md`.

### Added reviewer station datasets

The repository now includes a manifest for the original Marylebone dataset plus
four reviewer stations:

- `MY1`: existing London Marylebone Road baseline dataset.
- `BIR_A4540`: Birmingham A4540, 2021-2025 coverage.
- `HP1`: London Honor Oak Park, with the 2016-2018 no-coverage block retained.
- `CHB`: Chilbolton, including the July 2018 instrument-regime change.
- `KC1`: London North Kensington, including the end-2017 instrument-regime change.

Raw station files live under `data/raw/stations/`, and `data/stations.csv`
drives preprocessing and multi-station comparison. The reader accepts both
`datetime, PM2.5, Status` and UK-Air `Date, Time, PM2.5, Status` CSV exports.

Run station-level preprocessing and the data-quality comparison table with:

```powershell
python scripts/preprocess_stations.py
```

This writes per-station artifacts to `data/processed/stations/<station_id>/`
and `results/station_data_quality_comparison.csv`.

### Run the revised confirmatory experiment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_full_study.ps1
```

This executes all configured stations with five seeds and 30 epochs. It can take
several hours on CPU; do not report quick-demo metrics.

### Run the common-history sensitivity analysis

The primary experiment uses all legitimately available training history at
each station. Because BIR_A4540 begins in 2021, run the pre-specified
common-history sensitivity after the primary experiment:

```powershell
.\run_common_history_sensitivity.ps1
```

This repeats all five stations and five seeds with training restricted to
2021-01-01 through 2022-12-31. It does not overwrite the primary artifacts.
The comparison is written to `results/history_sensitivity_comparison.csv` and
`results/history_sensitivity_hybrid.csv`.

Known station-specific limitations are: no HP1 observations in 2016--2018,
BIR_A4540 beginning in 2021, CHB 2024 coverage of approximately 73.44%, 561
negative CHB source readings, and an instrument-regime change at KC1.

This repository is a reproducible university research project for forecasting hourly PM2.5 across multiple UK monitoring stations. It contains auditable cleaning, origin-safe features, matched baselines, statistical comparison, eight main-paper figures, supplementary EDA, tests, a dashboard, and manuscript source.

## What the study predicts

- Target: hourly PM2.5 concentration in µg/m³.
- History: 168 hours.
- Direct horizons: 1, 6, 12, 24, and 48 hours.
- Exogenous weather/traffic/spatial data: deliberately not required.
- Models: persistence, leakage-safe seasonal-naive, direct XGBoost, LSTM, compact TimeMixer, and residual Hybrid TimeMixer–XGBoost.

The hybrid uses TimeMixer forecasts and multiscale embeddings plus calendar, missingness, and instrument-regime context. One XGBoost model learns the residual for each horizon.

## Verified dataset issues and policy

The supplied 2016–2025 CSV contains 87,672 rows, 8,087 missing/non-numeric readings, 238 negative measurements, a 1,656-hour (69-day) maximum invalid gap, approximately 77.81% valid coverage in 2020, and a TEOM FDMS-to-BAM transition at 2021-12-22 12:00 UTC.

Raw data is never overwritten. Negative values become missing only in the modeling field. Only bounded internal gaps of at most three hours are interpolated. Longer gaps remain missing, and windows that cross them are excluded and counted. Instrument type is retained as TEOM FDMS=0 and BAM=1; it is treated as a predictive regime marker, not a causal adjustment.

## Leakage controls

| Stage | Period | Use |
|---|---|---|
| Training | through 2022-12-31 | scalers, neural weights, direct XGBoost |
| Neural validation | 2023 | early stopping |
| Hybrid calibration | 2024-01-01 to 2024-06-30 | residual XGBoost |
| Locked test | after 2024-06-30 | final evaluation only |

Window construction uses legitimate past history across boundaries but requires every forecast target to stay within its partition. Scaling is fitted on training arrays only. All models are compared on the same finite test origins.

## Install

Python 3.10 or newer is required. From the repository root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,app]"
~~~

For CPU-only PyTorch, follow the current PyTorch installation guidance if the default wheel is unsuitable. GPU is optional.

## Run

Fast structural/smoke workflow:

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.un_quick_demo.ps1
~~~

Confirmatory 30-epoch, five-seed workflow:

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.un_full_study.ps1
~~~

Individual stages:

~~~powershell
python scripts/validate_environment.py
pytest
python scripts/run_cleaning.py
python scripts/preprocess_stations.py
python scripts/run_baselines.py
python scripts/run_experiment.py --epochs 30
python scripts/run_multistation.py --epochs 30 --seeds 0 1 2 3 4
.\run_common_history_sensitivity.ps1
python scripts/run_multiseed.py --epochs 30 --seeds 0 1 2 3 4
python scripts/make_figures.py
python scripts/export_framework_pdf.py
python scripts/export_latex_results.py
streamlit run app/streamlit_app.py
~~~

Do not submit metrics from the three-epoch quick run. It verifies execution only.

## Generated data products

- `pm25_annotated_raw.csv`: source rows plus quality reasons.
- `pm25_clean.csv`: complete hourly grid and gap flags.
- `pm25_model_ready.csv`: origin-safe model features.
- `gap_inventory.csv`, `annual_coverage.csv`, `cleaning_audit.json`: audit evidence.
- `predictions.csv`: origin time, target time, truth, and all forecasts.
- metric, high-event, bootstrap, Diebold–Mariano, ablation, SHAP, and multi-seed tables.
- trained neural states and scaler arrays.

See `docs/result_schema.md` for the exact meaning of each output.

## Repository map

~~~text
app/                 Streamlit research dashboard
data/raw/            immutable source export
data/processed/      generated audit, clean, and feature tables
docs/                architecture, protocols, cards, schemas, proposal
figures/             eight main-paper figures plus supplementary EDA
notebooks/           five guided audit/EDA/model/reporting notebooks
paper/               manuscript, bibliography, generated results fragment
results/             machine-readable experiment evidence
scripts/             stage and full-study runners
src/pm25forecast/    installable research package
tests/               cleaning, leakage, metric, model, and structure tests
~~~

## Paper workflow

The manuscript is `paper/main.tex`. It contains complete background, data, methods, protocol, data-quality results, discussion, limitations, reproducibility and appendices. Forecast performance is inserted from `paper/generated_results.tex` only after the experiment runs. This prevents unmeasured placeholder values from being presented as findings.

Compile from the paper directory after exporting results:

~~~powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
~~~

## Scientific interpretation

A lower hybrid point MAE is not enough to claim superiority. Read it with the paired 24-hour block-bootstrap interval, Holm-adjusted Diebold–Mariano test, high-event recall/false alarms, ablations, and five-seed variability. SHAP values describe the residual booster; they do not establish pollution causality.

## Scope and limitations

This remains a PM2.5-only, station-level forecasting design. Multi-station evaluation tests transferability within the selected network but does not establish generalization to other countries, estimate personal exposure, or replace official warnings.

## License and citation

Code is provided under the repository license. Source data remains subject to UK-Air terms and attribution. Update the author placeholders in `CITATION.cff` and `paper/main.tex` before submission.
