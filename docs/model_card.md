# Model card — Gated Hybrid TimeMixer–XGBoost

## Intended use

Research-grade hourly PM2.5 forecasting at five UK monitoring stations using
historical PM2.5 and metadata known at forecast origin. The system is not an
official warning service, an exposure model, or a replacement for monitored
observations.

## Components

TimeMixer uses a 168-hour sequence containing cleaned PM2.5, missingness,
instrument-regime, and cyclic calendar channels. It produces direct forecasts
for 1, 6, 12, 24, and 48 hours. Comparator models are persistence,
seasonal-naive, direct XGBoost, DLinear, and LSTM.

For each horizon, pseudo-Huber XGBoost predicts the de-biased TimeMixer
residual. A correction weight in [0,1] is selected only on held-out
calibration-validation data. Calibration-derived clipping and leakage-safe
rolling bias updates further limit harmful corrections. The locked test set is
never used to choose the gate.

## Evaluation status

The confirmatory evaluation contains five stations, seeds 0--4, and a
30-epoch maximum. The hybrid improves TimeMixer in pooled MAE at all five
horizons but wins 0/25 station--horizon comparisons against the strongest
matched non-hybrid comparator. It must not be described as SOTA or as the best
forecasting model in this benchmark.

## Risks

Performance can degrade after sensor changes, emissions-policy shifts,
unusual weather, or events absent from training. PM2.5-only inputs omit
important physical drivers. Station histories and coverage differ. Average
accuracy can conceal poor peak-event behavior. Feature attribution is not
causal.

## Monitoring

Track coverage, feature ranges, gate weights, rolling bias, MAE by horizon,
and high-event errors. Recalibrate after persistent drift, instrument change,
or sustained coverage change while preserving an untouched recent test period.
