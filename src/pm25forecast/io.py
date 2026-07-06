"""Schema-aware input/output routines; raw files are never overwritten."""
from pathlib import Path
import os
import time
import pandas as pd
from .schema import DATETIME_ALTERNATIVES, MISSING_MARKERS, RAW_REQUIRED

def _datetime_source(frame):
    for columns in DATETIME_ALTERNATIVES:
        if all(column in frame.columns for column in columns):
            if columns == ("datetime",):
                return frame["datetime"].copy()
            return (frame["Date"].str.strip()+" "+frame["Time"].str.strip()).astype("string")
    expected=["datetime or Date+Time", *RAW_REQUIRED]
    raise ValueError(f"Missing required columns {expected}; found {list(frame.columns)}")

def read_uk_air_csv(path):
    """Read a UK-Air export while retaining exact source text and row identity."""
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"UK-Air CSV not found: {path}")
    # Read as text first: pandas must not erase 'No data' before it is audited.
    frame=pd.read_csv(path,dtype="string",keep_default_na=False)
    missing=[c for c in RAW_REQUIRED if c not in frame.columns]
    if missing: raise ValueError(f"Missing required columns {missing}; found {list(frame.columns)}")
    if frame.empty: raise ValueError("The UK-Air CSV contains no observations")
    frame=frame.copy(); frame["source_file"]=path.name
    frame["source_row"]=range(2,len(frame)+2)
    frame["datetime_raw"]=_datetime_source(frame)
    frame["pm25_raw"]=frame["PM2.5"].copy()
    frame["status_raw"]=frame["Status"].copy()
    frame["datetime"]=pd.to_datetime(frame["datetime_raw"],utc=True,errors="coerce")
    cleaned=frame["pm25_raw"].str.strip().replace(list(MISSING_MARKERS),pd.NA)
    frame["PM2.5"]=pd.to_numeric(cleaned,errors="coerce")
    return frame

def atomic_csv(frame,path):
    """Write through a sibling temporary file to avoid half-written outputs."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp,index=False)
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            try:
                if path.exists():
                    path.unlink()
                    tmp.replace(path)
                    return
            except PermissionError:
                time.sleep(0.5*(attempt+1))
