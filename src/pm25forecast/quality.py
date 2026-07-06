"""Measurement-quality flags and audit summaries."""
import numpy as np
import pandas as pd
from .schema import INSTRUMENT_CODES

def annotate_measurements(frame):
    x=frame.copy(); status=x["Status"].fillna("").astype(str)
    x["verification_status"]=status.str.extract(r"^\s*([VPNS])",expand=False).astype("string")
    x["instrument_name"]=status.str.extract(r"\(([^()]*)\)\s*$",expand=False).str.strip().str.upper().astype("string")
    x["instrument_type"]=x["instrument_name"].map(INSTRUMENT_CODES).astype("Int64")
    x["invalid_timestamp"]=x["datetime"].isna(); x["negative_pm25"]=x["PM2.5"].lt(0).fillna(False); x["missing_or_non_numeric"]=x["PM2.5"].isna(); x["duplicate_timestamp"]=x["datetime"].notna()&x["datetime"].duplicated(keep=False)
    x["pm25_observed"]=x["PM2.5"].mask(x["negative_pm25"])
    x["quality_issue"]=np.select([x.invalid_timestamp,x.negative_pm25,x.missing_or_non_numeric,x.duplicate_timestamp],["invalid_timestamp","negative_measurement","missing_or_non_numeric","duplicate_timestamp"],default="valid")
    previous=x["instrument_name"].ffill().shift(); x["instrument_change"]=x["instrument_name"].notna()&previous.notna()&x["instrument_name"].ne(previous)
    return x

def flag_outliers(frame,value="pm25_clean"):
    """Flag suspicious PM2.5 spikes without removing or winsorising them."""
    x=frame.copy()
    series=x[value].astype(float)
    valid=series.dropna()
    if valid.empty:
        x["outlier_iqr_high"]=False
        x["outlier_iqr_extreme"]=False
        x["rolling_spike_outlier"]=False
        x["temporal_jump_flag"]=False
        x["outlier_reason"]=""
        return x
    q1,q3=valid.quantile([0.25,0.75])
    iqr=q3-q1
    high=q3+1.5*iqr if iqr>0 else q3
    extreme=q3+3.0*iqr if iqr>0 else q3
    history=series.shift(1)
    rolling_median=history.rolling(24,min_periods=12).median()
    rolling_mad=(history-rolling_median).abs().rolling(24,min_periods=12).median()
    robust_z=(series-rolling_median).abs()/(1.4826*rolling_mad.replace(0,np.nan))
    diff=series.diff().abs()
    diff_q=diff.dropna().quantile(0.995) if diff.notna().any() else np.inf
    x["outlier_iqr_high"]=series.gt(high).fillna(False)
    x["outlier_iqr_extreme"]=series.gt(extreme).fillna(False)
    x["rolling_spike_outlier"]=robust_z.gt(6).fillna(False)
    x["temporal_jump_flag"]=diff.gt(diff_q).fillna(False)
    reasons=[]
    for _,row in x[["outlier_iqr_high","outlier_iqr_extreme","rolling_spike_outlier","temporal_jump_flag"]].iterrows():
        reason=[name for name,flag in row.items() if bool(flag)]
        reasons.append(";".join(reason))
    x["outlier_reason"]=reasons
    return x

def annual_coverage(frame,value="pm25_observed"):
    x=frame.dropna(subset=["datetime"]).copy(); x["year"]=x.datetime.dt.year
    out=x.groupby("year").agg(expected_hours=("datetime","size"),observed_hours=(value,"count")); out["missing_hours"]=out.expected_hours-out.observed_hours; out["coverage"]=out.observed_hours/out.expected_hours
    return out.reset_index()

def raw_audit(frame):
    return {"raw_rows":int(len(frame)),"invalid_timestamps":int(frame.invalid_timestamp.sum()),"missing_or_non_numeric":int(frame.missing_or_non_numeric.sum()),"negative_values":int(frame.negative_pm25.sum()),"duplicate_rows":int(frame.duplicate_timestamp.sum()),"unknown_instruments":sorted(frame.loc[frame.instrument_type.isna()&frame.instrument_name.notna(),"instrument_name"].dropna().unique().tolist()),"instrument_changes":[str(v) for v in frame.loc[frame.instrument_change,"datetime"]]}

def outlier_audit(frame):
    return {
        "outlier_iqr_high":int(frame.outlier_iqr_high.sum()),
        "outlier_iqr_extreme":int(frame.outlier_iqr_extreme.sum()),
        "rolling_spike_outliers":int(frame.rolling_spike_outlier.sum()),
        "temporal_jump_flags":int(frame.temporal_jump_flag.sum()),
    }
