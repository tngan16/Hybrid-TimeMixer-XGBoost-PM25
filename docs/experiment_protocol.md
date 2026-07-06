# Confirmatory experiment protocol

## Research questions

1. Does the revised hybrid reduce horizon-specific MAE relative to the
   strongest matched baseline at each station?
2. Which component explains a gain: median de-biasing, robust residual
   learning, or leakage-safe rolling calibration?
3. Are gains consistent across horizons, stations, and five random seeds?
4. Does the method preserve signed bias and high-PM2.5 event performance?

At origin \(t\), models receive the preceding 168 hourly rows and directly
predict \(t+1,t+6,t+12,t+24,t+48\). Apply the same PM2.5 schema and cleaning
policy at every station. The runner reads required stations from
`data/stations.csv` and does not silently fall back to one station.

| Partition | Period | Permitted use |
|---|---|---|
| Training | through 2022-12-31 | scalers, neural models, direct XGBoost |
| Validation | 2023 | early stopping |
| Calibration | 2024-01-01 to 2024-06-30 | robust residual fitting |
| Locked test | after 2024-06-30 | past-error update and evaluation |

Rolling calibration at origin \(t\) may use only errors whose target timestamp
is no later than \(t\).

Compare persistence, seasonal-naive, direct XGBoost, DLinear, LSTM, TimeMixer,
static robust hybrid, and rolling-calibrated hybrid on a common finite-origin
mask. For each horizon: subtract the calibration median residual, fit
pseudo-Huber XGBoost to the centered residual, clip its correction to
calibration quantiles, then apply a shrunk 30-day rolling median of already
observable forecast errors. Ablations add these components sequentially.

The calibration interval is split chronologically: the first 70% fits the
residual booster and the last 30% selects a correction multiplier from
`0, 0.10, 0.25, 0.50, 0.75, 1`. A zero multiplier is an explicit fallback to
TimeMixer when residual correction harms held-out calibration MAE. Residual
inputs include origin-safe lags and shifted rolling summaries as well as the
TimeMixer forecast vector, embedding, calendar, instrument, missingness, and
quality context.

Run seeds `0 1 2 3 4`, maximum 30 epochs, early stopping patience seven.
Report mean and standard deviation. The reference is the strongest non-hybrid
comparator at the same station, horizon, and eligible origins—not TimeMixer by
default. “SOTA” is permitted only after all preregistered stations and
comparator implementations complete under the same protocol.

## Common-history sensitivity analysis

The primary analysis uses each station's available history through
2022-12-31. Because BIR_A4540 starts in 2021, repeat the full five-seed protocol
with `train_start=2021-01-01 00:00:00+00:00` for every station. The common
history applies to sequence windows, direct XGBoost rows, scalers, and the
training-derived high-PM2.5 threshold. Validation, residual calibration, and
locked-test dates remain unchanged.

Report `common minus available history` changes in MAE and RMSE. Do not select
between the two protocols according to locked-test performance; the available-
history experiment is primary and the common-history experiment is a
pre-specified robustness check.
