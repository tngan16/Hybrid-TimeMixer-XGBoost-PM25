"""Canonical column names and validation helpers for UK-Air PM2.5 data."""
RAW_REQUIRED=("PM2.5","Status")
DATETIME_ALTERNATIVES=(("datetime",),("Date","Time"))
INSTRUMENT_CODES={
    "TEOM FDMS":0,
    "BAM":1,
    "REF.EQ":2,
    "REF EQ":2,
    "REFERENCE EQUIVALENT":2,
    "REFERENCE EQUIVALENT MEASUREMENT":2,
}
MISSING_MARKERS=("No data","No Data","NO DATA","N/A","NA","")
LAGS=(1,2,3,6,12,24,48,72,168)
ROLLING_WINDOWS=(3,6,12,24,48,168)
SEQUENCE_FEATURES=(
    "pm25_clean","pm25_missing_mask","instrument_type",
    "outlier_iqr_high","outlier_iqr_extreme","rolling_spike_outlier",
    "temporal_jump_flag",
    "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos",
)
