import pandas as pd
import numpy as np
from pm25forecast.splits import chronological_split
from pm25forecast.config import Config
from pm25forecast.pipeline import _numeric_context

def test_chronological_splits_do_not_overlap():
    cfg=Config(); dt=pd.date_range("2022-12-30", "2024-07-02",freq="h",tz="UTC")
    x=pd.DataFrame({"datetime":dt,"pm25_clean":1.0})
    train,val,test=chronological_split(x,cfg)
    assert train.datetime.max()<val.datetime.min()<test.datetime.min()

def test_nullable_context_converts_to_finite_float_matrix():
    frame=pd.DataFrame({
        "a":pd.Series([1,pd.NA],dtype="Int64"),
        "b":["2.5","bad"],
    })
    matrix=_numeric_context(frame,["a","b"])
    assert matrix.dtype==float
    assert np.isfinite(matrix).all()
    assert matrix.tolist()==[[1.0,2.5],[0.0,0.0]]
