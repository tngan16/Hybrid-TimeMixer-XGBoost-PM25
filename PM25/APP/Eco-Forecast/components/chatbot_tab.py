"""Static Q&A tab inspired by the sample project's assistant page."""
from __future__ import annotations

import streamlit as st


QA = {
    "What does the model predict?": "Hourly PM2.5 concentration at horizons 1, 6, 12, 24, and 48 hours.",
    "Does the hybrid beat all baselines?": "No. It improves TimeMixer in pooled results, but wins 0/25 comparisons against the strongest matched non-hybrid comparator.",
    "Why use a gate?": "The gate prevents the residual XGBoost layer from applying corrections when held-out calibration shows that correction is harmful.",
    "Is this an official forecast service?": "No. It is a university research prototype and should not replace official air-quality information.",
}


def render_chatbot() -> None:
    st.title("Research Q&A")
    st.caption("A lightweight explanation tab for presentation/demo use.")

    question = st.selectbox("Choose a question", list(QA))
    st.info(QA[question])

    with st.expander("Suggested presentation summary"):
        st.markdown(
            """
            This project evaluates a gated TimeMixer-XGBoost framework for
            multi-horizon PM2.5 forecasting. The key lesson is that residual
            learning can improve a TimeMixer base model, but careful calibration
            is necessary because a residual layer can also over-correct. The final
            claim is intentionally conservative and evidence-based.
            """
        )

