"""Fail-fast environment, dataset, and writable-output checks."""
from pathlib import Path
import importlib, platform, sys
ROOT=Path(__file__).resolve().parents[1]
PACKAGES=["numpy","pandas","sklearn","scipy","matplotlib","seaborn","torch","xgboost"]
print("Python:",sys.version.replace("\n"," ")); print("Platform:",platform.platform())
if sys.version_info<(3,10): raise SystemExit("Python 3.10 or newer is required")
failed=[]
for name in PACKAGES:
    try:
        module=importlib.import_module(name); print(f"{name}: {getattr(module,'__version__','installed')}")
    except Exception as exc: failed.append((name,str(exc)))
if failed: raise SystemExit("Missing/broken dependencies: "+str(failed))
import torch
print("CUDA available:",torch.cuda.is_available())
if torch.cuda.is_available(): print("CUDA device:",torch.cuda.get_device_name(0))
import pandas as pd
manifest=ROOT/"data/stations.csv"
if not manifest.exists():
    raise SystemExit(f"Station manifest missing: {manifest}")
stations=pd.read_csv(manifest)
manifest_required={"station_id","path"}
if not manifest_required.issubset(stations):
    raise SystemExit(
        f"Station manifest missing columns: {manifest_required-set(stations)}"
    )
for station in stations.itertuples(index=False):
    raw=ROOT/station.path
    if not raw.exists():
        raise SystemExit(f"Dataset missing for {station.station_id}: {raw}")
    sample=pd.read_csv(raw,nrows=5)
    measurement_required={"PM2.5","Status"}
    if not measurement_required.issubset(sample):
        raise SystemExit(
            f"{station.station_id} schema missing "
            f"{measurement_required-set(sample)}"
        )
    has_timestamp=(
        "datetime" in sample.columns
        or {"Date","Time"}.issubset(sample.columns)
    )
    if not has_timestamp:
        raise SystemExit(
            f"{station.station_id} requires datetime or Date + Time columns"
        )
    print(f"{station.station_id}: dataset and schema ready")
for folder in [ROOT/"data/processed",ROOT/"results",ROOT/"figures"]:
    folder.mkdir(parents=True,exist_ok=True)
    probe=folder/".write_test"; probe.write_text("ok",encoding="utf-8"); probe.unlink()
print("Environment, dataset schema, and output directories are ready.")
