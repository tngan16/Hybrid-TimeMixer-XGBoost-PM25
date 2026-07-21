import numpy as np
import pandas as pd
from pm25forecast.baselines import _numeric_matrix

def test_xgboost_features_are_coerced_from_object_to_float():
    frame=pd.DataFrame({"lag":["1.0","2.5",None],"instrument":pd.Series([0,1,1],dtype="Int64")})
    matrix=_numeric_matrix(frame,["lag","instrument"])
    assert all(np.issubdtype(dtype,np.number) for dtype in matrix.dtypes)
    assert matrix.loc[0,"lag"]==1.0
    assert np.isnan(matrix.loc[2,"lag"])
