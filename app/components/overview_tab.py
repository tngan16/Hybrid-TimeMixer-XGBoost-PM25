"""Dataset overview tab, adapted from the sample dashboard structure."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.data_pipeline import load_station_manifest, load_station_raw
from visuals.charts import show_metric_cards


def render_overview() -> None:
    st.title("Dataset Analytics")
    st.caption("UK-Air PM2.5 station files used by this research project.")

    manifest = load_station_manifest()
    if manifest.empty:
        st.error("data/stations.csv was not found.")
        return

    st.subheader("Station manifest")
    st.dataframe(manifest, use_container_width=True)

    station_id = st.selectbox("Inspect station", manifest["station_id"].tolist())
    df = load_station_raw(station_id)
    if df.empty:
        st.warning("Station file is missing or empty.")
        return

    observed = df["pm25"].notna() if "pm25" in df else pd.Series(False, index=df.index)
    negative_count = pd.to_numeric(df.get("pm25_raw", pd.Series(dtype=object)), errors="coerce").lt(0).sum()
    show_metric_cards(
        {
            "Rows": f"{len(df):,}",
            "Observed PM2.5": f"{int(observed.sum()):,}",
            "Missing/invalid": f"{int((~observed).sum()):,}",
            "Negative raw values": f"{int(negative_count):,}",
        }
    )

    if "datetime" in df and df["datetime"].notna().any():
        daily = df.dropna(subset=["datetime"]).set_index("datetime")["pm25"].resample("D").mean()
        st.subheader("Daily PM2.5 profile")
        st.line_chart(daily)

    st.subheader("Raw preview")
    st.dataframe(df.head(200), use_container_width=True)

