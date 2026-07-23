"""Prototype forecast tab with optional AI explanation service."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from pipeline.data_pipeline import load_metrics, load_station_manifest, load_station_raw


HORIZONS = [1, 6, 12, 24, 48]


def _latest_station_value(station_id: str) -> tuple[float | None, pd.Timestamp | None]:
    frame = load_station_raw(station_id)
    if frame.empty or "pm25" not in frame:
        return None, None
    clean = frame.dropna(subset=["datetime", "pm25"]).sort_values("datetime")
    if clean.empty:
        return None, None
    row = clean.iloc[-1]
    return float(row["pm25"]), row["datetime"]


def _station_hourly_medians(station_id: str) -> pd.Series:
    frame = load_station_raw(station_id)
    if frame.empty or "pm25" not in frame:
        return pd.Series(dtype=float)
    clean = frame.dropna(subset=["datetime", "pm25"]).copy()
    if clean.empty:
        return pd.Series(dtype=float)
    clean["hour"] = clean["datetime"].dt.hour
    return clean.groupby("hour")["pm25"].median()


def _best_model_for_horizon(horizon: int) -> str:
    metrics = load_metrics()
    if metrics.empty:
        return "historical baseline"
    subset = metrics.loc[metrics["horizon"].eq(horizon)]
    if subset.empty:
        return "historical baseline"
    return subset.groupby("model")["MAE"].mean().sort_values().index[0]


def _prototype_forecast(
    station_id: str,
    current_pm25: float,
    horizon: int,
    current_hour: int,
) -> dict:
    """Small demo forecast based on persistence plus historical hourly pattern.

    This is intentionally lightweight for the dashboard. It is not the trained
    research model; the trained-model evidence remains in the Performance tab.
    """
    medians = _station_hourly_medians(station_id)
    target_hour = (current_hour + horizon) % 24
    if medians.empty or current_hour not in medians.index or target_hour not in medians.index:
        adjustment = 0.0
    else:
        adjustment = float(medians.loc[target_hour] - medians.loc[current_hour])

    # Conservative shrinkage for longer horizons to avoid exaggerated demo values.
    shrink = {1: 0.70, 6: 0.55, 12: 0.40, 24: 0.25, 48: 0.15}[horizon]
    forecast = max(0.0, current_pm25 + shrink * adjustment)

    if forecast < 12:
        category = "Low"
    elif forecast < 35:
        category = "Moderate"
    else:
        category = "High"

    return {
        "forecast": forecast,
        "target_hour": target_hour,
        "hourly_adjustment": adjustment,
        "risk_level": category,
        "best_research_model": _best_model_for_horizon(horizon),
    }


def _fallback_explanation(station_id: str, current_pm25: float, horizon: int, result: dict) -> str:
    direction = "higher" if result["hourly_adjustment"] > 0 else "lower or similar"
    return (
        f"For station {station_id}, the demo forecast is mainly driven by the current "
        f"PM2.5 value ({current_pm25:.1f} µg/m³) and the historical median pattern "
        f"for the target hour. The target-hour pattern is {direction} than the current "
        f"hour, so the predicted value is {result['forecast']:.1f} µg/m³. The risk "
        f"level is labelled {result['risk_level']}. In the research results, "
        f"{result['best_research_model']} is the strongest matched model for the "
        f"{horizon}h horizon, while this tab is only a small interactive prototype."
    )


def _call_ai_service(prompt: str, api_key: str | None, model: str | None) -> str | None:
    """Call OpenAI-compatible chat completions if credentials are supplied."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model or os.getenv("OPENAI_MODEL")
    if not api_key or not model:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain PM2.5 forecast demo results for a university project. "
                    "Be concise, cautious, and do not claim this is an official warning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 180,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        st.info(f"AI service unavailable, using local explanation instead. Detail: {exc}")
        return None


def render_forecast() -> None:
    st.title("Forecast Demo + AI Explanation")
    st.caption(
        "Small interactive prototype for presentation use. It is not the locked-test research model "
        "and not an official air-quality warning service."
    )

    manifest = load_station_manifest()
    if manifest.empty:
        st.warning("data/stations.csv is missing.")
        return

    station_ids = manifest["station_id"].dropna().astype(str).tolist()
    col_left, col_right = st.columns([0.42, 0.58])

    with col_left:
        st.subheader("Input")
        station_id = st.selectbox("Station", station_ids, index=0)
        horizon = st.selectbox("Forecast horizon", HORIZONS, index=1, format_func=lambda h: f"{h} hours")

        latest_value, latest_time = _latest_station_value(station_id)
        default_pm25 = float(latest_value) if latest_value is not None else 10.0
        current_pm25 = st.number_input(
            "Current PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=round(default_pm25, 2),
            step=0.5,
        )
        current_hour = st.slider("Current hour of day", 0, 23, int(latest_time.hour) if latest_time is not None else 12)

        with st.expander("Optional AI service settings"):
            st.caption("Leave blank to use the local explanation. For a real API call, set both fields.")
            api_key = st.text_input("OpenAI API key", type="password", value="")
            model = st.text_input("Model name", value=os.getenv("OPENAI_MODEL", ""))

    result = _prototype_forecast(station_id, current_pm25, horizon, current_hour)

    with col_right:
        st.subheader("Prediction output")
        m1, m2, m3 = st.columns(3)
        m1.metric("Demo forecast", f"{result['forecast']:.2f} µg/m³")
        m2.metric("Risk label", result["risk_level"])
        m3.metric("Target hour", f"{result['target_hour']:02d}:00")

        st.write(
            {
                "station_id": station_id,
                "horizon_hours": horizon,
                "current_pm25": round(current_pm25, 3),
                "demo_forecast_pm25": round(result["forecast"], 3),
                "best_research_model_for_horizon": result["best_research_model"],
            }
        )

        prompt = (
            f"Station: {station_id}. Current PM2.5: {current_pm25:.2f} µg/m3. "
            f"Horizon: {horizon} hours. Demo forecast: {result['forecast']:.2f} µg/m3. "
            f"Risk label: {result['risk_level']}. Best research model for this horizon: "
            f"{result['best_research_model']}. Explain this for a presentation slide."
        )
        explanation = _call_ai_service(prompt, api_key.strip() or None, model.strip() or None)
        st.subheader("AI explanation")
        st.info(explanation or _fallback_explanation(station_id, current_pm25, horizon, result))

    st.warning(
        "This tab is a presentation prototype. It demonstrates user input, prediction output, "
        "and AI-style explanation, but the formal paper results are evaluated in the Performance Dashboard."
    )
