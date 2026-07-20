"""Model performance tab."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.data_pipeline import load_effectiveness, load_gate, load_metrics
from visuals.charts import show_metric_cards, show_model_mae_by_horizon, show_station_effectiveness


def render_performance() -> None:
    st.title("Performance Dashboard")
    st.caption("Measured results from the five-station, five-seed, 30-epoch gated residual study.")

    metrics = load_metrics()
    effectiveness = load_effectiveness()
    gates = load_gate()

    if metrics.empty:
        st.warning("results/multistation_seed_metrics.csv is missing. Run scripts/run_multistation.py first.")
        return

    runs = metrics[["station_id", "seed"]].drop_duplicates().shape[0]
    best = metrics.groupby("model")["MAE"].mean().sort_values().head(1)
    hybrid_mae = metrics.loc[metrics["model"].eq("hybrid"), "MAE"].mean()
    tm_mae = metrics.loc[metrics["model"].eq("timemixer"), "MAE"].mean()
    show_metric_cards(
        {
            "Station-seed runs": f"{runs}",
            "Best pooled model": best.index[0] if len(best) else "n/a",
            "Hybrid pooled MAE": f"{hybrid_mae:.3f}",
            "TimeMixer pooled MAE": f"{tm_mae:.3f}",
        }
    )

    st.subheader("Pooled MAE by horizon")
    show_model_mae_by_horizon(metrics)

    st.subheader("Hybrid vs strongest matched comparator")
    show_station_effectiveness(effectiveness)

    if not gates.empty:
        st.subheader("Residual gate behavior")
        gate_summary = gates.groupby("horizon")["correction_gate_alpha"].agg(["mean", "std", "min", "max"]).round(3)
        st.dataframe(gate_summary, use_container_width=True)
        st.caption("A gate of zero means the residual correction was rejected on held-out calibration.")

    with st.expander("Raw metrics table"):
        st.dataframe(metrics, use_container_width=True)

