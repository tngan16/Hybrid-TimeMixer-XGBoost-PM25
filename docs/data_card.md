# Dataset card — London Marylebone Road PM2.5

## Identity and provenance

- File: `data/raw/London_Marylebone_PM25.csv`
- Station: London Marylebone Road (UK-Air site MY1)
- Geographic context: urban traffic/roadside monitoring location in central London
- Temporal resolution: hourly, stored and modeled in UTC
- Coverage in supplied export: 2016–2025
- Raw rows: 87,672
- Intended task: direct multi-horizon prediction of station-level PM2.5 concentration

The data remains subject to the source provider's terms. Users should retain source attribution and consult UK-Air metadata before redistribution.

## Known quality issues in the supplied file

- 8,087 missing or non-numeric readings marked as “No data” or equivalent.
- 238 negative measurements, treated as physically invalid for this forecasting target.
- Largest contiguous invalid interval: 1,656 hours (69 days).
- 2020 valid coverage: approximately 77.81% after invalid negatives are excluded.
- HP1 has no observed PM2.5 coverage during 2016--2018.
- BIR_A4540 begins in 2021 and therefore has a shorter primary training history.
- CHB has approximately 73.44% coverage in 2024 and 561 negative source values.
- KC1 contains a documented instrument-regime transition.

The confirmatory protocol reports station-specific eligible-window counts.
A separate common-history sensitivity experiment restricts every station's
training history to 2021-01-01 through 2022-12-31.
- Instrument transition from TEOM FDMS to BAM on 2021-12-22 12:00 UTC.

These values are verified by the generated `cleaning_audit.json`; they are not hard-coded into the pipeline.

## Cleaning contract

Raw rows are never modified. Parsing errors become explicit nulls. Negative measurements become missing while their original value and reason flag remain in the annotated table. After deterministic timestamp deduplication, an hourly grid is inserted. Only bounded missing runs no longer than three hours are interpolated. The 69-day gap and every other longer gap remain missing. Model windows crossing unresolved gaps are excluded.

## Fields of particular importance

- `pm25_raw`: source value/text retained for audit.
- `pm25_observed`: numeric, non-negative measurement before interpolation.
- `pm25_clean`: observed value plus permitted short-gap interpolation.
- `quality_issue`: primary quality label.
- `gap_id`, `gap_length_hours`, `gap_class`: missing-run metadata.
- `was_interpolated`, `long_gap_preserved`, `inserted_grid_row`: transformation flags.
- `instrument_name`, `instrument_type`: TEOM FDMS=0, BAM=1.

## Appropriate and inappropriate use

The dataset supports methods research and short-term station forecasting. It does not estimate personal exposure, prove causal effects, represent all of London, or support regulatory compliance decisions without expert validation. Instrument regime is confounded with time and should not be interpreted causally.
