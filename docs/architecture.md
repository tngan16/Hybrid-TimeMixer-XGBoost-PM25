# System architecture

The repository separates immutable source data, deterministic transformations,
model fitting, and reporting so each number is traceable to station, seed,
configuration, and code version.

1. `io.py` validates schema and preserves source text.
2. `quality.py` labels timestamp, missing, negative, duplicate, verification,
   and instrument-regime conditions.
3. `cleaning.py` creates an hourly UTC grid, interpolates only bounded 1–3 hour
   internal gaps, and leaves longer outages unresolved.
4. `features.py` creates origin-safe lags, shifted rolling summaries, and
   cyclic calendar fields.
5. `windows.py` constructs 168-hour inputs and direct 1/6/12/24/48-hour
   targets, rejecting and counting incomplete windows.
6. `pipeline.py` trains persistence, seasonal-naive, XGBoost, DLinear, LSTM,
   TimeMixer, static robust hybrid, and rolling-calibrated hybrid.
7. `run_multistation.py` applies the identical protocol to every manifest
   station and isolates station/seed artifacts.
8. Evaluation uses common test origins, high-event metrics, paired block
   bootstrap intervals, and Holm-adjusted Diebold–Mariano tests.
9. Reporting creates tables and every quantitative figure from saved evidence.

## Hybrid information path

TimeMixer emits direct forecasts and a multiscale embedding. At each horizon,
the calibration median removes systematic bias. Pseudo-Huber XGBoost predicts
the centered residual from base forecasts, the embedding, and origin-known
context. Calibration-quantile clipping limits unstable corrections. During
the locked test, a shrunk rolling median uses only errors whose target values
would already be available at the forecast origin.

## Failure behavior

Schema errors, duplicate window timestamps, irregular grids, missing manifest
files, empty splits, and zero eligible windows raise explicit errors. The full
study does not silently revert to the original one-station experiment.
