from dataclasses import replace
import numpy as np
import pandas as pd
from pm25forecast.config import Config
from pm25forecast.schema import SEQUENCE_FEATURES
from pm25forecast.windows import make_windows

def hourly(n=300):
    dt=pd.date_range("2024-01-01",periods=n,freq="h",tz="UTC")
    x=pd.DataFrame({"datetime":dt,"pm25_clean":np.arange(n,dtype=float)})
    x["pm25_missing_mask"]=0; x["instrument_type"]=0
    # Keep the synthetic fixture aligned with the canonical model schema.
    # New binary quality channels default to zero (no issue) in this test.
    for c in SEQUENCE_FEATURES:
        if c not in x:
            x[c]=0.0
    return x

def test_target_alignment_and_partition_boundary():
    cfg=replace(Config(),input_length=24,horizons=(1,6,12))
    x=hourly(); start=x.datetime.iloc[100]; end=x.datetime.iloc[199]
    w=make_windows(x,cfg,origin_start=start,target_end=end)
    assert w.origin_time.min()>=start
    assert w.target_time.max().max()<=end
    assert np.allclose(w.y[:,0],w.context.pm25_clean.to_numpy()+1)
    assert np.allclose(w.y[:,-1],w.context.pm25_clean.to_numpy()+12)
    assert w.X.shape[1:]==(24,len(SEQUENCE_FEATURES))

def test_windows_crossing_missing_values_are_rejected():
    cfg=replace(Config(),input_length=24,horizons=(1,6))
    x=hourly(); x.loc[50,"pm25_clean"]=np.nan
    w=make_windows(x,cfg)
    assert w.rejected_missing_input>0
    assert w.rejected_missing_target>0
