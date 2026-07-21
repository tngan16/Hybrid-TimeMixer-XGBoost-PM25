import numpy as np
import pandas as pd
from pm25forecast.config import Config
from pm25forecast.data import split
from pm25forecast.baselines import seasonal_naive

def test_seasonal_naive_uses_only_past_references():
    dt=pd.date_range("2021-01-01",periods=200,freq="h",tz="UTC")
    full=pd.DataFrame({"datetime":dt,"pm25_clean":np.arange(200.)})
    context=full.iloc[[100]]
    p=seasonal_naive(full,context,[1,6,12,24,48])
    assert np.all(p[0]==[77,82,88,100,100])

def test_three_way_compatibility_split_has_no_overlap():
    cfg=Config(train_end="2021-01-03 00:00:00+00:00",
               calibration_start="2021-01-04 00:00:00+00:00",
               val_end="2021-01-05 00:00:00+00:00")
    x=pd.DataFrame({"datetime":pd.date_range("2021-01-01",periods=150,freq="h",tz="UTC"),
                    "pm25_clean":1.0})
    tr,va,te=split(x,cfg)
    assert tr.datetime.max()<va.datetime.min()<te.datetime.min()
