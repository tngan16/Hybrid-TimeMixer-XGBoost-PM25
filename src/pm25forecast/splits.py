"""Chronological partition utilities and leakage assertions."""
import pandas as pd

def research_partitions(frame,cfg):
    x=frame.sort_values("datetime")
    train_end=pd.Timestamp(cfg.train_end); cal_start=pd.Timestamp(cfg.calibration_start); val_end=pd.Timestamp(cfg.val_end)
    parts={
        "train":x.loc[x.datetime.le(train_end)].copy(),
        "neural_validation":x.loc[x.datetime.gt(train_end)&x.datetime.lt(cal_start)].copy(),
        "hybrid_calibration":x.loc[x.datetime.ge(cal_start)&x.datetime.le(val_end)].copy(),
        "test":x.loc[x.datetime.gt(val_end)].copy(),
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("Every temporal partition must contain rows")
    previous=None
    for name,part in parts.items():
        if previous is not None and not previous.datetime.max()<part.datetime.min():
            raise AssertionError(f"Chronological overlap before {name}")
        previous=part
    return parts

def chronological_split(frame,cfg):
    """Compatibility API returning train, combined validation, and test."""
    parts=research_partitions(frame,cfg)
    validation=pd.concat([parts["neural_validation"],parts["hybrid_calibration"]]).sort_values("datetime")
    return parts["train"],validation,parts["test"]

def split_report(*parts):
    """Summarise either a partition mapping or three legacy data frames."""
    if len(parts)==1 and isinstance(parts[0],dict): items=parts[0].items()
    elif len(parts)==3: items=zip(("train","validation","test"),parts)
    else: raise TypeError("split_report expects a mapping or train, validation, test")
    rows=[]
    for name,part in items:
        rows.append({"split":name,"start":str(part.datetime.min()),
                     "end":str(part.datetime.max()),"rows":int(len(part)),
                     "valid_pm25":int(part.pm25_clean.notna().sum()),
                     "coverage":float(part.pm25_clean.notna().mean())})
    return rows
