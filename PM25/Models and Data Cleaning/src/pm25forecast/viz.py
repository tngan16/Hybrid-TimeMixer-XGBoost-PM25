"""All paper charts are reproducibly generated here; no image AI is used."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid",context="notebook")


def _save(fig,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight")
    plt.close(fig)


def data_quality_figures(data,figdir):
    """Two main EDA composites plus detailed supplementary plots."""
    figdir=Path(figdir); supplement=figdir/"supplementary"
    x=data.copy(); x["datetime"]=pd.to_datetime(x.datetime,utc=True)
    x["year"]=x.datetime.dt.year; x["month"]=x.datetime.dt.month
    x["hour"]=x.datetime.dt.hour; x["dow"]=x.datetime.dt.dayofweek
    daily=x.set_index("datetime").pm25_clean.resample("D").mean()
    counts=x.groupby("year").size()
    years=counts[counts.ge(24*300)].index
    coverage=(x[x.year.isin(years)].groupby("year").pm25_observed
              .apply(lambda s:s.notna().mean()*100))
    missing=(x[x.year.isin(years)]
             .assign(missing=lambda f:f.pm25_observed.isna().astype(int))
             .pivot_table(index="year",columns="month",values="missing",
                          aggfunc="mean")*100)
    mask=x.is_missing; group=mask.ne(mask.shift()).cumsum()
    gaps=x.loc[mask].groupby(group[mask]).agg(
        hours=("gap_length_hours","max"),start=("datetime","min")
    )
    lags=[1,2,3,6,12,24,48,72,168,336,720]
    ac=[x.pm25_clean.autocorr(lag) for lag in lags]
    excerpt=x.set_index("datetime").loc["2020-09":"2021-01"]

    fig,axs=plt.subplots(2,2,figsize=(14,9))
    axs[0,0].plot(daily.index,daily,lw=.6,color="#17365D")
    axs[0,0].set(title="(a) Daily PM2.5",ylabel="ug/m3")
    sns.histplot(x.pm25_clean,bins=60,ax=axs[0,1],color="#4E79A7")
    axs[0,1].set(title="(b) Concentration distribution",
                 xlim=(0,x.pm25_clean.quantile(.995)))
    x.groupby("hour").pm25_clean.mean().plot(
        ax=axs[1,0],marker="o",title="(c) Mean diurnal profile"
    )
    axs[1,1].stem(lags,ac)
    axs[1,1].set(title="(d) Selected-lag autocorrelation",
                 xlabel="Lag (hours)",ylabel="Correlation",ylim=(-.1,1))
    _save(fig,figdir/"01_data_overview.png")

    fig,axs=plt.subplots(2,2,figsize=(14,9))
    bars=axs[0,0].bar(
        coverage.index,coverage,
        color=np.where(coverage<80,"#E15759","#4E79A7")
    )
    axs[0,0].axhline(80,color="black",ls="--",lw=1)
    axs[0,0].bar_label(bars,fmt="%.0f",fontsize=8)
    axs[0,0].set(title="(a) Annual observed coverage",ylabel="%")
    sns.heatmap(missing,cmap="Reds",ax=axs[0,1],
                cbar_kws={"label":"Missing (%)"})
    axs[0,1].set(title="(b) Monthly missingness")
    sns.histplot(gaps.hours.clip(upper=200),bins=40,
                 ax=axs[1,0],color="#E15759")
    axs[1,0].set(title="(c) Missing-gap duration",
                 xlabel="Hours (clipped at 200)")
    axs[1,1].plot(excerpt.index,excerpt.pm25_observed,label="Observed",lw=.8)
    axs[1,1].plot(excerpt.index,excerpt.pm25_clean,label="Clean",lw=.8)
    axs[1,1].set(title="(d) Long gap is not interpolated",ylabel="ug/m3")
    axs[1,1].legend()
    _save(fig,figdir/"02_data_quality.png")

    # Detailed EDA belongs to the supplement.
    details=[
        ("S01_full_timeline.png",lambda ax:ax.plot(daily.index,daily,lw=.7)),
        ("S02_annual_coverage.png",lambda ax:ax.bar(coverage.index,coverage)),
        ("S04_gap_duration.png",
         lambda ax:sns.histplot(gaps.hours.clip(upper=200),bins=40,ax=ax)),
        ("S08_autocorrelation.png",lambda ax:ax.stem(lags,ac)),
    ]
    for name,draw in details:
        fig,ax=plt.subplots(figsize=(10,4)); draw(ax); _save(fig,supplement/name)
    fig,ax=plt.subplots(figsize=(12,5))
    sns.heatmap(missing,cmap="Reds",annot=True,fmt=".0f",ax=ax)
    _save(fig,supplement/"S03_missingness_heatmap.png")
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(excerpt.index,excerpt.pm25_observed,label="Observed")
    ax.plot(excerpt.index,excerpt.pm25_clean,label="Clean"); ax.legend()
    _save(fig,supplement/"S05_long_gap.png")
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    sns.histplot(x.pm25_clean,bins=80,kde=True,ax=axs[0])
    sns.boxplot(data=x,x="instrument_name",y="pm25_clean",
                showfliers=False,ax=axs[1])
    _save(fig,supplement/"S06_distribution_instrument.png")
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    x.groupby("hour").pm25_clean.mean().plot(ax=axs[0])
    x.groupby("dow").pm25_clean.mean().plot(ax=axs[1])
    x.groupby("month").pm25_clean.mean().plot(ax=axs[2])
    _save(fig,supplement/"S07_temporal_profiles.png")
    threshold=x.loc[x.datetime.le("2022-12-31"),"pm25_clean"].quantile(.9)
    high=(x.assign(high=x.pm25_clean.ge(threshold))
          .groupby(["year","month"]).high.mean().unstack()*100)
    fig,ax=plt.subplots(figsize=(12,5)); sns.heatmap(high,cmap="YlOrRd",ax=ax)
    _save(fig,supplement/"S09_high_pollution_heatmap.png")


def result_figures(root,figdir):
    root=Path(root); figdir=Path(figdir)
    multi=root/"results/multistation_seed_metrics.csv"
    effect=root/"results/station_effectiveness.csv"
    if multi.exists():
        data=pd.read_csv(multi)
        mean=(data.groupby(["station_id","model","horizon"],as_index=False)
              .MAE.mean())
        fig,axs=plt.subplots(1,2,figsize=(15,5))
        sns.lineplot(data=mean,x="horizon",y="MAE",hue="model",
                     style="station_id",markers=True,ax=axs[0])
        if effect.exists():
            gain=pd.read_csv(effect)
            sns.barplot(data=gain,x="horizon",y="improvement_percent",
                        hue="station_id",ax=axs[1])
            axs[1].axhline(0,color="black")
        _save(fig,figdir/"03_multistation_effectiveness.png")
        fig,axs=plt.subplots(1,2,figsize=(13,4))
        summary=(data.groupby(["model","horizon"],as_index=False)
                 [["MAE","RMSE"]].mean())
        sns.lineplot(data=summary,x="horizon",y="MAE",hue="model",
                     marker="o",ax=axs[0])
        sns.lineplot(data=summary,x="horizon",y="RMSE",hue="model",
                     marker="o",ax=axs[1])
        _save(fig,figdir/"04_model_metrics.png")
        fig,ax=plt.subplots(figsize=(10,5))
        sns.lineplot(data=data,x="horizon",y="MAE",hue="model",
                     units="station_id",estimator=None,alpha=.35,ax=ax)
        ax.set_title("Five-seed, multi-station variability")
        _save(fig,figdir/"07_seed_variability.png")
    else:
        metrics=root/"results/experiment_metrics.csv"
        if metrics.exists():
            data=pd.read_csv(metrics)
            fig,axs=plt.subplots(1,2,figsize=(13,4))
            sns.lineplot(data=data,x="horizon",y="MAE",hue="model",
                         marker="o",ax=axs[0])
            sns.lineplot(data=data,x="horizon",y="RMSE",hue="model",
                         marker="o",ax=axs[1])
            _save(fig,figdir/"04_model_metrics.png")

    prediction_candidates=sorted(
        (root/"artifacts/stations").glob("*/seed_0/results/predictions.csv")
    )
    predictions=(prediction_candidates[0] if prediction_candidates
                 else root/"results/predictions.csv")
    if predictions.exists():
        data=pd.read_csv(predictions,parse_dates=["origin_time"]); h=24
        tail=data.tail(500); fig,axs=plt.subplots(1,2,figsize=(15,5))
        axs[0].plot(tail.origin_time,tail[f"y_h{h}"],label="Observed")
        for model in ("timemixer","hybrid_static","hybrid"):
            column=f"{model}_h{h}"
            if column in tail: axs[0].plot(tail.origin_time,tail[column],label=model)
        axs[0].legend()
        residual=pd.DataFrame({
            "TimeMixer":data[f"y_h{h}"]-data[f"timemixer_h{h}"],
            "Hybrid":data[f"y_h{h}"]-data[f"hybrid_h{h}"],
        }).melt(var_name="model",value_name="error")
        sns.histplot(data=residual,x="error",hue="model",
                     stat="density",common_norm=False,element="step",ax=axs[1])
        _save(fig,figdir/"05_forecast_diagnostics.png")

    ablation=root/"results/multistation_ablation.csv"
    if not ablation.exists():
        ablation=root/"results/ablation_metrics.csv"
    if ablation.exists():
        data=pd.read_csv(ablation); fig,ax=plt.subplots(figsize=(10,5))
        sns.lineplot(data=data,x="horizon",y="MAE",
                     hue="variant",marker="o",ax=ax)
        ax.set_title("Residual ablation")
        _save(fig,figdir/"06_residual_ablation.png")

    boot=root/"results/multistation_bootstrap.csv"
    if not boot.exists():
        boot=root/"results/bootstrap_comparison.csv"
    if boot.exists():
        data=pd.read_csv(boot)
        pooled=(data.groupby("horizon",as_index=False)
                [["mean_diff","ci_low","ci_high"]].mean()
                .sort_values("horizon"))
        fig,ax=plt.subplots()
        ax.errorbar(pooled.mean_diff,pooled.horizon,
                    xerr=[pooled.mean_diff-pooled.ci_low,
                          pooled.ci_high-pooled.mean_diff],fmt="o")
        ax.axvline(0,color="black",ls="--")
        _save(fig,figdir/"08_bootstrap_forest.png")


def make_all(root):
    root=Path(root); figdir=root/"figures"; figdir.mkdir(exist_ok=True)
    data=root/"data/processed/pm25_model_ready.csv"
    if not data.exists():
        from .pipeline import prepare
        prepare(root/"data/raw/stations/London_Marylebone_PM25.csv",root)
    data_quality_figures(pd.read_csv(data),figdir)
    result_figures(root,figdir)
    manifest={
        "main":sorted(p.name for p in figdir.glob("*.png")
                      if p.name[:2] in {f"{i:02d}" for i in range(1,9)}),
        "supplementary":sorted(
            p.name for p in (figdir/"supplementary").glob("*.png")
        ),
        "provenance":"All PNG charts are generated by src/pm25forecast/viz.py",
    }
    (figdir/"figure_manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf-8"
    )
    return manifest
