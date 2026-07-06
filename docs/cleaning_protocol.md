# Detailed cleaning and processing protocol

## Principles

The policy is conservative, deterministic, reversible at the row level, and fixed before model evaluation. Missingness is evidence about the monitoring process; it is not erased for convenience.

## Procedure

1. Validate that `datetime`, `PM2.5` and `Status` exist and that the file has rows.
2. Preserve source filename, one-based CSV row number, raw timestamp text and raw PM2.5 value.
3. Parse timestamps as timezone-aware UTC. A failed parse receives `invalid_timestamp=True`.
4. Parse PM2.5 numerically. Source markers such as “No data” become null and receive `missing_or_non_numeric=True`.
5. Flag negative concentrations and replace them only in `pm25_observed`; the raw value stays intact.
6. Extract verification code and final parenthesized instrument name from Status. Map TEOM FDMS to 0, BAM to 1 and Reference Equivalent / Ref.eq to 2. Unknown names remain null and are listed in the audit.
7. Sort by timestamp and source row. For duplicate timestamps, retain the last source row while keeping the duplicate count in the raw audit.
8. Reindex from minimum to maximum time at exactly one-hour intervals. Mark rows created by reindexing.
9. Forward/backward propagate instrument regime across grid-only rows; this is metadata propagation, not PM2.5 imputation.
10. Identify contiguous missing runs, assign IDs and classify 1–3 h, 4–24 h, 25–168 h and >7 d gaps.
11. Apply linear interpolation only when a gap is internal, bounded on both sides and at most three hours. No edge extrapolation is allowed.
12. Retain all longer missing values. Sequence windows containing them are rejected, with counts saved in `run_config.json`.
13. Screen high PM2.5 episodes with reversible flags instead of deleting them. The pipeline records global IQR high/extreme flags, rolling robust-spike flags and temporal-jump flags. These flags are audit metadata and optional model context; the concentration value remains unchanged.
14. Create calendar, lag and rolling features. Every rolling feature is shifted by one hour. Direct XGBoost may use the current origin measurement, but never a value after the origin.
15. Fit standardization mean and standard deviation on training windows only and reuse them unchanged for all later periods.

## Validation checks

- unique, monotonically increasing hourly timestamps;
- no negative values in `pm25_observed` or `pm25_clean`;
- all interpolated rows belong to gaps of at most three hours;
- all long-gap rows remain null;
- outlier flags do not remove or winsorise pollution spikes;
- target timestamps are strictly after forecast origins;
- all target timestamps remain within their assigned temporal partition;
- preprocessing audit totals reconcile with row-level flags.

## Sensitivity analysis

For a final dissertation, rerun with short-gap limits 0, 1, 3 and 6 hours. The primary analysis remains three hours. Report whether model rankings change, rather than selecting the limit that produces the lowest test error.
