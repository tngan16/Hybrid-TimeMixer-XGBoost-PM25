"""Interactive explorer for generated research artefacts (not a live forecast service)."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/processed/pm25_clean.csv"; RESULTS=ROOT/"results"; FIGURES=ROOT/"figures"
st.set_page_config(page_title="PM2.5 Forecast Research Lab",page_icon="🌫️",layout="wide")
st.title("Hybrid TimeMixer–XGBoost PM2.5 Research Lab")
st.caption("London Marylebone Road · historical PM2.5-only experiment · not an official warning service")
st.sidebar.header("Interpretation")
st.sidebar.info("All performance values are read from generated result files. If a panel is absent, run the corresponding research script; the app never substitutes demo metrics.")
@st.cache_data
def csv(path,**kwargs): return pd.read_csv(path,**kwargs)
@st.cache_data
def js(path): return json.loads(path.read_text(encoding="utf-8"))
quality,patterns,models,events,diagnostics,gallery=st.tabs(["Data quality","Temporal patterns","Models","High events","Diagnostics","Figures"])
with quality:
    if DATA.exists():
        x=csv(DATA,parse_dates=["datetime"]); audit=js(RESULTS/"cleaning_audit.json") if (RESULTS/"cleaning_audit.json").exists() else {}
        cols=st.columns(5); cols[0].metric("Hourly grid",f"{len(x):,}"); cols[1].metric("Raw no-data",f"{audit.get('missing_or_non_numeric',0):,}"); cols[2].metric("Negative",f"{audit.get('negative_values',0):,}"); cols[3].metric("Interpolated",f"{audit.get('interpolated_values',0):,}"); cols[4].metric("Largest gap",f"{audit.get('largest_gap_hours',0):,} h")
        st.subheader("Observed coverage by year")
        cov=x.groupby(x.datetime.dt.year).pm25_observed.apply(lambda s:s.notna().mean()*100).rename("coverage_percent")
        st.bar_chart(cov)
        c1,c2=st.columns([2,1]); c1.line_chart(x.set_index("datetime").pm25_clean.resample("D").mean(),height=350)
        c2.dataframe(x.groupby("instrument_name").pm25_clean.agg(["count","mean","median","std"]).round(2),width="stretch")
        gap=RESULTS/"gap_summary.csv"
        if gap.exists(): st.dataframe(csv(gap).head(25),width="stretch")
    else: st.info("Run scripts/run_cleaning.py first.")
with patterns:
    if DATA.exists():
        x=csv(DATA,parse_dates=["datetime"]); x=x.assign(hour=x.datetime.dt.hour,weekday=x.datetime.dt.day_name(),month=x.datetime.dt.month)
        c1,c2,c3=st.columns(3); c1.line_chart(x.groupby("hour").pm25_clean.mean()); c2.line_chart(x.groupby("month").pm25_clean.mean()); c3.line_chart(x.groupby("weekday").pm25_clean.mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
        st.subheader("Distribution below the 99.5th percentile")
        capped=x.loc[x.pm25_clean.le(x.pm25_clean.quantile(.995)),"pm25_clean"]
        histogram=(pd.cut(capped,bins=50,include_lowest=True)
                   .value_counts(sort=False)
                   .rename_axis("pm25_range").reset_index(name="hours"))
        histogram["pm25_range"]=histogram["pm25_range"].astype(str)
        st.bar_chart(histogram,x="pm25_range",y="hours")
with models:
    mp=RESULTS/"experiment_metrics.csv"
    if mp.exists():
        m=csv(mp); metric=st.selectbox("Metric",["MAE","RMSE","sMAPE","R2","Bias"]); st.line_chart(m.pivot(index="horizon",columns="model",values=metric)); st.dataframe(m.sort_values(["horizon",metric]),width="stretch")
    else: st.info("Run scripts/run_experiment.py to generate measured model results.")
    pp=RESULTS/"predictions.csv"
    if pp.exists():
        p=csv(pp,parse_dates=["origin_time"]); h=st.select_slider("Horizon (hours)",[1,6,12,24,48],value=24); n=st.slider("Most recent origins",100,3000,500)
        columns=[c for c in [f"y_h{h}",f"persistence_h{h}",f"xgboost_h{h}",f"timemixer_h{h}",f"hybrid_h{h}"] if c in p]
        st.line_chart(p.tail(n).set_index("origin_time")[columns])
with events:
    hp=RESULTS/"high_pollution_metrics.csv"
    if hp.exists():
        h=csv(hp); st.dataframe(h,width="stretch")
        c1,c2=st.columns(2); c1.bar_chart(h.pivot(index="horizon",columns="model",values="event_recall")); c2.bar_chart(h.pivot(index="horizon",columns="model",values="false_alarm_rate"))
    else: st.info("High-event metrics are created by the full experiment.")
with diagnostics:
    for name,title in [("bootstrap_comparison.csv","Paired bootstrap: hybrid minus TimeMixer"),("diebold_mariano.csv","Diebold–Mariano tests"),("ablation_metrics.csv","Hybrid ablations"),("shap_summary.csv","Residual feature contributions")]:
        p=RESULTS/name
        st.subheader(title)
        if p.exists(): st.dataframe(csv(p),width="stretch")
        else: st.caption("Not generated yet.")
with gallery:
    figs=sorted(FIGURES.glob("*.png"))
    if not figs: st.info("Run scripts/make_figures.py.")
    for i in range(0,len(figs),2):
        for col,p in zip(st.columns(2),figs[i:i+2]): col.image(str(p),caption=p.stem,width="stretch")
st.warning("Research prototype: station concentration is not personal exposure, and this app must not replace official air-quality information.")
