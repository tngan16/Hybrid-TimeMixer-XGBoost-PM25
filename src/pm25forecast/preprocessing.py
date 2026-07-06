"""End-to-end preprocessing orchestration with machine-readable audit output."""
from pathlib import Path
import json
from .io import atomic_csv,read_uk_air_csv
from .quality import annotate_measurements,annual_coverage,flag_outliers,outlier_audit,raw_audit
from .cleaning import complete_hourly_grid,gap_table,interpolate_short_internal_gaps
from .features import build_features

def preprocess(raw_path,output_dir,short_gap_limit=3):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); raw=read_uk_air_csv(raw_path); annotated=annotate_measurements(raw); audit=raw_audit(annotated); grid=complete_hourly_grid(annotated); clean=interpolate_short_internal_gaps(grid,short_gap_limit); clean=flag_outliers(clean); features=build_features(clean); gaps=gap_table(clean); coverage=annual_coverage(clean)
    audit.update({"complete_hourly_rows":int(len(clean)),"inserted_grid_rows":int(clean.inserted_grid_row.sum()),"interpolated_values":int(clean.was_interpolated.sum()),"unresolved_missing":int(clean.pm25_clean.isna().sum()),"largest_gap_hours":int(gaps.hours.max()) if len(gaps) else 0,"short_gap_limit":int(short_gap_limit),**outlier_audit(clean)})
    atomic_csv(annotated,output/"pm25_annotated_raw.csv"); atomic_csv(clean,output/"pm25_clean.csv"); atomic_csv(features,output/"pm25_model_ready.csv"); atomic_csv(gaps,output/"gap_inventory.csv"); atomic_csv(coverage,output/"annual_coverage.csv"); (output/"cleaning_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return features,audit,gaps,coverage
