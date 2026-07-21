"""About tab for the research dashboard."""
from __future__ import annotations

import streamlit as st


def render_about() -> None:
    st.title("About This Study")
    st.markdown(
        """
        This dashboard follows the presentation style of the provided PM2.5-weather
        sample project, but it uses this repository's own UK-Air PM2.5 dataset and
        generated experiment outputs.

        **Research question.** Can a conservative XGBoost residual layer improve a
        compact TimeMixer PM2.5 forecaster across several horizons and stations?

        **Forecast target.** Hourly PM2.5 concentration.

        **Stations.** MY1, BIR_A4540, HP1, CHB, and KC1.

        **Horizons.** 1, 6, 12, 24, and 48 hours.

        **Important interpretation.** The gated hybrid improves TimeMixer at pooled
        horizons, but it does not beat the strongest non-hybrid comparator in the
        completed 25 station-horizon comparisons. The project therefore reports a
        careful negative/partial result rather than a state-of-the-art claim.
        """
    )

    st.subheader("Model components")
    st.markdown(
        """
        - **TimeMixer base:** learns multiscale sequence structure.
        - **XGBoost residual:** learns horizon-specific correction from TimeMixer
          forecasts, embeddings, lags, rolling summaries, missingness, calendar,
          and instrument context.
        - **Calibration gate:** selects how much residual correction to apply using
          held-out calibration data; zero gate falls back to TimeMixer.
        - **Rolling calibration:** adjusts recent residual bias using only errors
          already observable at the forecast origin.
        """
    )

