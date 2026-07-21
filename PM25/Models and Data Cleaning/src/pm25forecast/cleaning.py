"""Deterministic cleaning policy with gap-level auditability."""
import pandas as pd

def complete_hourly_grid(frame):
    x=frame.dropna(subset=["datetime"]).sort_values(["datetime","source_row"]).drop_duplicates("datetime",keep="last").copy()
    index=pd.date_range(x.datetime.min(),x.datetime.max(),freq="h",tz="UTC"); x=x.set_index("datetime").reindex(index); x.index.name="datetime"; x=x.reset_index(); x["inserted_grid_row"]=x.source_row.isna()
    x["instrument_name"]=x.instrument_name.ffill().bfill(); x["instrument_type"]=x.instrument_type.ffill().bfill().astype("Int64"); x["instrument_change"]=x.instrument_change.fillna(False).astype(bool)
    return x

def describe_gaps(frame,value="pm25_observed"):
    x=frame.copy(); missing=x[value].isna(); group=missing.ne(missing.shift(fill_value=False)).cumsum(); length=missing.groupby(group).transform("sum")
    x["is_missing"]=missing; x["gap_id"]=group.where(missing).astype("Int64"); x["gap_length_hours"]=length.where(missing,0).astype(int)
    x["gap_class"]=pd.cut(x.gap_length_hours,[-1,0,3,24,168,float("inf")],labels=["observed","short_1_3h","medium_4_24h","long_25_168h","very_long_gt_7d"]).astype("string")
    return x

def gap_table(frame):
    x=frame.loc[frame.is_missing]
    if x.empty: return pd.DataFrame(columns=["gap_id","start","end","hours","gap_class","inserted_rows"])
    return x.groupby("gap_id",as_index=False).agg(start=("datetime","min"),end=("datetime","max"),hours=("gap_length_hours","max"),gap_class=("gap_class","first"),inserted_rows=("inserted_grid_row","sum")).sort_values(["hours","start"],ascending=[False,True])

def interpolate_short_internal_gaps(frame,limit=3,value="pm25_observed"):
    if limit<0: raise ValueError("Interpolation limit must be non-negative")
    x=describe_gaps(frame,value); candidate=x[value].interpolate(method="linear",limit_area="inside"); allowed=x.is_missing&x.gap_length_hours.le(limit)
    x["pm25_clean"]=x[value]; x.loc[allowed,"pm25_clean"]=candidate.loc[allowed]; x["was_interpolated"]=allowed&x.pm25_clean.notna(); x["long_gap_preserved"]=x.is_missing&x.gap_length_hours.gt(limit); x["pm25_missing_mask"]=x.pm25_clean.isna().astype("int8")
    return x
