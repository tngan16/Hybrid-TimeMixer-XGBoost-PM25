"""Data access layer for the PM2.5 research dashboard.

The sample project uses a dashboard-specific pipeline module. This project
keeps the same idea, but every function reads this repository's UK-Air PM2.5
data and generated experiment artifacts rather than the sample NSW/weather
dataset.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PAPER_FIGURES_DIR = ROOT / "paper" / "figures"


@st.cache_data(show_spinner=False)
def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_station_manifest() -> pd.DataFrame:
    return read_csv(DATA_DIR / "stations.csv")


@st.cache_data(show_spinner=False)
def load_station_raw(station_id: str) -> pd.DataFrame:
    manifest = load_station_manifest()
    if manifest.empty or station_id not in set(manifest["station_id"]):
        return pd.DataFrame()

    rel_path = manifest.loc[manifest["station_id"].eq(station_id), "path"].iloc[0]
    path = ROOT / rel_path
    df = read_csv(path)
    if df.empty:
        return df

    cols = {c.lower().strip(): c for c in df.columns}
    if "datetime" in cols:
        df["datetime"] = pd.to_datetime(df[cols["datetime"]], errors="coerce")
    elif "date" in cols and "time" in cols:
        # UK-Air hour-ending files may contain 24:00. Handle it without
        # changing the source columns.
        date = pd.to_datetime(df[cols["date"]], errors="coerce")
        time_text = df[cols["time"]].astype(str).str.strip()
        is_24 = time_text.str.startswith("24")
        safe_time = time_text.mask(is_24, "00:00")
        df["datetime"] = pd.to_datetime(
            date.dt.strftime("%Y-%m-%d") + " " + safe_time,
            errors="coerce",
        )
        df.loc[is_24 & df["datetime"].notna(), "datetime"] += pd.Timedelta(days=1)
    else:
        df["datetime"] = pd.NaT

    pm_col = next((c for c in df.columns if c.lower().replace(" ", "") in {"pm2.5", "pm25"}), None)
    if pm_col is not None:
        df["pm25_raw"] = df[pm_col]
        df["pm25"] = pd.to_numeric(df[pm_col], errors="coerce")
        df.loc[df["pm25"] < 0, "pm25"] = pd.NA
    else:
        df["pm25_raw"] = pd.NA
        df["pm25"] = pd.NA

    return df.sort_values("datetime")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return read_csv(RESULTS_DIR / "multistation_seed_metrics.csv")


@st.cache_data(show_spinner=False)
def load_effectiveness() -> pd.DataFrame:
    return read_csv(RESULTS_DIR / "station_effectiveness.csv")


@st.cache_data(show_spinner=False)
def load_ablation() -> pd.DataFrame:
    return read_csv(RESULTS_DIR / "multistation_ablation.csv")


@st.cache_data(show_spinner=False)
def load_gate() -> pd.DataFrame:
    return read_csv(RESULTS_DIR / "multistation_residual_gate.csv")


@st.cache_data(show_spinner=False)
def load_quality_audit() -> dict:
    return read_json(RESULTS_DIR / "cleaning_audit.json")


def available_figures() -> list[Path]:
    """Return paper-first generated figures, with root figures as fallback."""
    candidates = []
    for directory in [PAPER_FIGURES_DIR, FIGURES_DIR]:
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.png")))
    # Deduplicate by filename while keeping paper figures first.
    seen = set()
    out = []
    for p in candidates:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out

