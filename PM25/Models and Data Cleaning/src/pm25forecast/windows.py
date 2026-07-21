"""Leakage-safe sequence construction for direct multi-horizon forecasting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .schema import SEQUENCE_FEATURES

@dataclass
class WindowSet:
    X: np.ndarray
    y: np.ndarray
    origin_time: pd.Series
    target_time: pd.DataFrame
    context: pd.DataFrame
    feature_names: list[str]
    rejected_missing_input: int
    rejected_missing_target: int

def make_windows(frame, cfg, feature_cols=None, origin_start=None, origin_end=None, target_end=None):
    """Create direct-forecast samples with explicit temporal boundaries.

    Passing the complete hourly timeline allows the first validation/test
    origins to use legitimate earlier history. The target_end argument ensures
    no target crosses into a later partition.
    """
    frame=frame.sort_values("datetime").reset_index(drop=True)
    if frame.datetime.duplicated().any():
        raise ValueError("Window input contains duplicated timestamps")
    delta=frame.datetime.diff().dropna()
    if len(delta) and not delta.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Window input must be a complete hourly grid")
    columns=list(feature_cols or SEQUENCE_FEATURES)
    missing=[c for c in [*columns,"pm25_clean","datetime"] if c not in frame]
    if missing:
        raise ValueError(f"Missing window columns: {missing}")
    values=(
        frame[columns]
        .apply(pd.to_numeric,errors="coerce")
        .to_numpy(dtype=np.float32,na_value=np.nan)
    )
    target=pd.to_numeric(
        frame.pm25_clean,errors="coerce"
    ).to_numpy(dtype=np.float32,na_value=np.nan)
    max_h=max(cfg.horizons)
    lower=pd.Timestamp(origin_start) if origin_start is not None else None
    upper=pd.Timestamp(origin_end) if origin_end is not None else None
    last_target=pd.Timestamp(target_end) if target_end is not None else None
    ends=np.arange(cfg.input_length-1,len(frame)-max_h,dtype=np.int64)
    origins=frame.datetime.iloc[ends].reset_index(drop=True)
    boundary=np.ones(len(ends),dtype=bool)
    if lower is not None:
        boundary &= origins.ge(lower).to_numpy()
    if upper is not None:
        boundary &= origins.le(upper).to_numpy()
    if last_target is not None:
        final_times=frame.datetime.iloc[ends+max_h].reset_index(drop=True)
        boundary &= final_times.le(last_target).to_numpy()
    ends=ends[boundary]

    # O(n) missing-window detection replaces the previous Python loop.
    invalid_row=~np.isfinite(values).all(axis=1)
    cumulative=np.concatenate([[0],np.cumsum(invalid_row,dtype=np.int64)])
    starts=ends-cfg.input_length+1
    input_invalid=(cumulative[ends+1]-cumulative[starts])>0
    bad_input=int(input_invalid.sum())

    target_idx=ends[:,None]+np.asarray(cfg.horizons,dtype=np.int64)[None,:]
    outputs=target[target_idx]
    target_invalid=~np.isfinite(outputs).all(axis=1)
    bad_target=int((~input_invalid & target_invalid).sum())
    eligible=~input_invalid & ~target_invalid
    ends=ends[eligible]
    starts=starts[eligible]
    outputs=outputs[eligible]
    target_idx=target_idx[eligible]
    if not len(ends):
        raise ValueError("No eligible windows after boundary and missingness checks")

    windows=np.lib.stride_tricks.sliding_window_view(
        values,cfg.input_length,axis=0
    )
    windows=np.moveaxis(windows,-1,1)
    X=windows[starts]
    times=frame.datetime.iloc[ends].reset_index(drop=True)
    target_times=frame.datetime.iloc[target_idx.ravel()].to_numpy().reshape(
        len(ends),len(cfg.horizons)
    )
    context=frame.iloc[ends].reset_index(drop=True)
    return WindowSet(
        np.asarray(X,np.float32), np.asarray(outputs,np.float32),
        pd.Series(times,name="datetime"),
        pd.DataFrame(target_times,columns=[f"target_h{h}" for h in cfg.horizons]),
        context, columns,
        bad_input, bad_target,
    )
