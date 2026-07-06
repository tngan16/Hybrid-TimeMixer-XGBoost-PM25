"""Preprocess every station in the manifest and compare data-quality issues."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from pm25forecast.config import Config
from pm25forecast.preprocessing import preprocess


def _load_manifest(path):
    stations=pd.read_csv(path)
    required={"station_id","station_name","site_type","path"}
    missing=required-set(stations.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return stations


def _station_summary(station,audit,gaps,coverage):
    longest=int(gaps.hours.max()) if len(gaps) else 0
    low_coverage_years=coverage.loc[
        coverage.coverage.lt(0.80),["year","coverage"]
    ].to_dict("records")
    return {
        "station_id":station.station_id,
        "station_name":station.station_name,
        "site_type":station.site_type,
        "path":station.path,
        "raw_rows":audit["raw_rows"],
        "missing_or_non_numeric":audit["missing_or_non_numeric"],
        "negative_values":audit["negative_values"],
        "inserted_grid_rows":audit["inserted_grid_rows"],
        "unresolved_missing":audit["unresolved_missing"],
        "largest_gap_hours":longest,
        "outlier_iqr_high":audit["outlier_iqr_high"],
        "outlier_iqr_extreme":audit["outlier_iqr_extreme"],
        "rolling_spike_outliers":audit["rolling_spike_outliers"],
        "temporal_jump_flags":audit["temporal_jump_flags"],
        "unknown_instruments": ";".join(audit["unknown_instruments"]),
        "instrument_change_count":len(audit["instrument_changes"]),
        "low_coverage_years":json.dumps(low_coverage_years),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default="data/stations.csv")
    parser.add_argument("--short-gap-limit",type=int,default=Config().short_gap_limit)
    args=parser.parse_args()

    manifest=ROOT/args.manifest
    stations=_load_manifest(manifest)
    rows=[]
    out_root=ROOT/"data/processed/stations"
    for station in stations.itertuples(index=False):
        raw=ROOT/station.path
        if not raw.exists():
            raise FileNotFoundError(f"Missing dataset for {station.station_id}: {raw}")
        output=out_root/station.station_id
        _,audit,gaps,coverage=preprocess(raw,output,args.short_gap_limit)
        rows.append(_station_summary(station,audit,gaps,coverage))

    results=ROOT/"results"
    results.mkdir(exist_ok=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(results/"station_data_quality_comparison.csv",index=False)
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
