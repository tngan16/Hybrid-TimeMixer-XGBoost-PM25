"""Small plotting helpers for the PM2.5 research dashboard."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def show_metric_cards(metrics: dict[str, object]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def show_model_mae_by_horizon(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("Run the multi-station experiment to generate metrics.")
        return
    pooled = (
        metrics.groupby(["model", "horizon"], as_index=False)["MAE"]
        .mean()
        .pivot(index="horizon", columns="model", values="MAE")
        .sort_index()
    )
    st.line_chart(pooled)
    st.dataframe(pooled.round(3), use_container_width=True)


def show_station_effectiveness(effectiveness: pd.DataFrame) -> None:
    if effectiveness.empty:
        st.info("No station effectiveness table found.")
        return
    heat = effectiveness.pivot(index="station_id", columns="horizon", values="improvement_percent")
    st.dataframe(heat.round(2), use_container_width=True)
    st.caption("Positive values mean the hybrid beats the strongest non-hybrid comparator; negative values mean it does not.")

