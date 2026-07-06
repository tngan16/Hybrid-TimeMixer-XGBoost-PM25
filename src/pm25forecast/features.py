"""Origin-safe deterministic feature engineering.

Every lag or rolling statistic is shifted so it uses observations available at
or before the forecast origin. No target-side value enters the predictors.
"""
import numpy as np
from .schema import LAGS,ROLLING_WINDOWS

def add_calendar_features(frame):
    x=frame.copy(); dt=x.datetime; x["hour"]=dt.dt.hour; x["day_of_week"]=dt.dt.dayofweek; x["month"]=dt.dt.month; x["day_of_year"]=dt.dt.dayofyear; x["is_weekend"]=dt.dt.dayofweek.ge(5).astype("int8")
    for name,period,value in [("hour",24,dt.dt.hour),("dow",7,dt.dt.dayofweek),("doy",365.25,dt.dt.dayofyear)]: x[f"{name}_sin"]=np.sin(2*np.pi*value/period); x[f"{name}_cos"]=np.cos(2*np.pi*value/period)
    return x

def add_lag_features(frame,lags=LAGS):
    x=frame.copy()
    for lag in lags: x[f"pm25_lag_{lag}"]=x.pm25_clean.shift(lag)
    return x

def add_rolling_features(frame,windows=ROLLING_WINDOWS):
    x=frame.copy(); history=x.pm25_clean.shift(1)
    for window in windows:
        roll=history.rolling(window,min_periods=window); x[f"pm25_mean_{window}"]=roll.mean(); x[f"pm25_std_{window}"]=roll.std(); x[f"pm25_min_{window}"]=roll.min(); x[f"pm25_max_{window}"]=roll.max(); x[f"pm25_median_{window}"]=roll.median()
    x["pm25_diff_1"]=x.pm25_clean.diff(1); x["pm25_diff_24"]=x.pm25_clean.diff(24); x["pm25_ratio_24"]=x.pm25_clean/(x.pm25_clean.shift(24).abs()+1e-6)
    return x

def build_features(frame): return add_rolling_features(add_lag_features(add_calendar_features(frame)))
