import numpy as np
import pandas as pd
from pm25forecast.features import build_features

def frame(n=200):
    dt=pd.date_range("2024-01-01",periods=n,freq="h",tz="UTC")
    return pd.DataFrame({"datetime":dt,"pm25_clean":np.arange(n,dtype=float)})

def test_lags_and_rolling_statistics_are_origin_safe():
    x=build_features(frame())
    i=180
    assert x.loc[i,"pm25_lag_24"]==156
    assert x.loc[i,"pm25_mean_3"]==np.mean([177,178,179])
    assert x.loc[i,"pm25_max_24"]==179
    # Changing the future must not change features at the current origin.
    changed=frame(); changed.loc[i+1:,"pm25_clean"]=99999
    y=build_features(changed)
    cols=[c for c in x if c.startswith("pm25_") and c!="pm25_clean"]
    pd.testing.assert_series_equal(x.loc[i,cols],y.loc[i,cols])

def test_calendar_encodings_are_bounded():
    x=build_features(frame())
    for col in ["hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos"]:
        assert x[col].between(-1,1).all()
