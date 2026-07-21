"""Naive and gradient-boosted baseline models."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def persistence(context, horizons):
    values=pd.to_numeric(context["pm25_clean"],errors="coerce").to_numpy(dtype=float)
    return np.repeat(values[:,None],len(horizons),axis=1)

def seasonal_naive(full_data, context, horizons, period=24):
    series=pd.to_numeric(full_data.set_index("datetime")["pm25_clean"],errors="coerce")
    rows=[]
    for origin in pd.to_datetime(context["datetime"],utc=True):
        row=[]
        for h in horizons:
            cycles=int(np.ceil(h/period))
            reference=origin+pd.Timedelta(hours=h-cycles*period)
            row.append(series.get(reference,np.nan))
        rows.append(row)
    return np.asarray(rows,dtype=float)

def _numeric_matrix(frame, features):
    """Return a finite-compatible float matrix regardless of pandas extension dtypes."""
    converted=frame.loc[:,features].apply(pd.to_numeric,errors="coerce")
    return converted.astype("float64")

def _model(cfg,seed):
    return XGBRegressor(
        n_estimators=cfg.xgb_estimators,max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,min_child_weight=cfg.xgb_min_child_weight,
        reg_alpha=.01,reg_lambda=1.0,objective="reg:squarederror",
        random_state=seed,n_jobs=-1,
    )

def fit_xgb_direct(train,features,horizons,cfg):
    X=_numeric_matrix(train,features)
    target=pd.to_numeric(train["pm25_clean"],errors="coerce").astype("float64")
    models=[]
    for h in horizons:
        y=target.shift(-h)
        ok=X.notna().all(axis=1)&y.notna()
        if not ok.any(): raise ValueError(f"No eligible XGBoost rows for horizon {h}")
        model=_model(cfg,cfg.seed+h)
        model.fit(X.loc[ok].to_numpy(dtype=np.float32),y.loc[ok].to_numpy(dtype=np.float32))
        models.append(model)
    return models

def predict_xgb_direct(models,frame,features):
    X=_numeric_matrix(frame,features)
    # XGBoost handles NaN natively; only the dtype must be numeric.
    matrix=X.to_numpy(dtype=np.float32)
    return np.column_stack([model.predict(matrix) for model in models])
