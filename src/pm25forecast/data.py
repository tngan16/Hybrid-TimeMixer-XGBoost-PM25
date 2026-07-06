"""Backward-compatible data API. New code should import focused modules directly."""
from .preprocessing import preprocess
from .io import read_uk_air_csv
from .quality import annotate_measurements
from .cleaning import complete_hourly_grid,interpolate_short_internal_gaps
from .features import build_features
from .splits import chronological_split as split
from .windows import make_windows

def clean_uk_air(path,short_gap_limit=3):
    raw=read_uk_air_csv(path); return interpolate_short_internal_gaps(complete_hourly_grid(annotate_measurements(raw)),short_gap_limit)
def add_features(frame): return build_features(frame)
