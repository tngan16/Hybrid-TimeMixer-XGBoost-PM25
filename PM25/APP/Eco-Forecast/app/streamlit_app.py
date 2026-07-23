"""Main router for the PM2.5 research dashboard.

The structure mirrors the supplied DAP391m sample project:
one router, separate component tabs, a data pipeline module, and a figure
gallery. The data and results remain this repository's UK-Air PM2.5 artifacts.
"""
from __future__ import annotations

import streamlit as st

from components.about_tab import render_about
from components.chatbot_tab import render_chatbot
from components.figures_tab import render_figures
from components.forecast_tab import render_forecast
from components.overview_tab import render_overview
from components.performance_tab import render_performance


st.set_page_config(page_title="PM2.5 Research Dashboard", page_icon="🌫️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1280px;}
    div[data-testid="stMetric"] {
        background: #f7f9fb;
        border: 1px solid #e6eaf0;
        padding: 0.8rem;
        border-radius: 0.8rem;
    }
    .small-note {color: #5b6470; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("PM2.5 Dashboard")
st.sidebar.caption("Hybrid TimeMixer-XGBoost research artifacts")

MENU = {
    "Dataset Analytics": render_overview,
    "Performance Dashboard": render_performance,
    "Forecast Demo + AI": render_forecast,
    "Generated Figures": render_figures,
    "Research Q&A": render_chatbot,
    "About": render_about,
}

choice = st.sidebar.radio("Navigation", list(MENU), index=1)

st.title("Gated TimeMixer-XGBoost PM2.5 Research Lab")
st.caption("UK-Air PM2.5-only study · five stations · five seeds · 30 epochs · not an official warning service")

MENU[choice]()

st.warning(
    "Research prototype: station concentration is not personal exposure, and this app must not replace official air-quality information."
)
