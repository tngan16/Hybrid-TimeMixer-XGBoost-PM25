"""Reproducible orchestration from immutable raw CSV to research artefacts.

The confirmatory test period is never used for feature fitting, scaling, early
stopping, residual-model training, threshold selection, or hyperparameters.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .config import Config
from .preprocessing import preprocess
from .splits import research_partitions, split_report
from .windows import make_windows
from .scaling import ArrayStandardizer
from .metrics import (point_metrics, high_pollution_metrics,
                      block_bootstrap_per_horizon, diebold_mariano_abs)
from .baselines import persistence, seasonal_naive, fit_xgb_direct, predict_xgb_direct
from .lstm import LSTMForecaster
from .dlinear import DLinearForecaster
from .timemixer import TimeMixer
from .train import fit_model, predict_with_embeddings, seed_all
from .hybrid import HybridResidual,rolling_bias_calibration

TABULAR=[f"pm25_lag_{v}" for v in (1,2,3,6,12,24,48,72,168)] + [
    f"pm25_{stat}_{w}" for w in (3,6,12,24,48,168)
    for stat in ("mean","std","min","max","median")
] + ["pm25_diff_1","pm25_diff_24","pm25_ratio_24","instrument_type",
     "outlier_iqr_high","outlier_iqr_extreme","rolling_spike_outlier",
     "temporal_jump_flag",
     "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend"]
CONTEXT=["instrument_type","hour_sin","hour_cos","dow_sin","dow_cos",
         "doy_sin","doy_cos","is_weekend","pm25_missing_mask",
         "gap_length_hours","was_interpolated","outlier_iqr_high",
         "outlier_iqr_extreme","rolling_spike_outlier","temporal_jump_flag"]
RESIDUAL_CONTEXT=list(dict.fromkeys(TABULAR+CONTEXT))

def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def prepare(raw, root, cfg=Config()):
    """Run deterministic cleaning/features and record split coverage."""
    root=Path(root); processed=root/"data/processed"; results=root/"results"
    results.mkdir(parents=True,exist_ok=True)
    data,audit,gaps,coverage=preprocess(raw,processed,cfg.short_gap_limit)
    partitions=research_partitions(data,cfg)
    _json(results/"split_report.json",split_report(partitions))
    for source,name in [
        (processed/"cleaning_audit.json","cleaning_audit.json"),
        (processed/"annual_coverage.csv","annual_coverage.csv"),
        (processed/"gap_inventory.csv","gap_summary.csv")]:
        (results/name).write_bytes(source.read_bytes())
    return data,audit

def _datasets(data,cfg):
    """Create train, neural-validation, calibration, and locked-test windows."""
    hour=pd.Timedelta(hours=1)
    train_end=pd.Timestamp(cfg.train_end)
    train_origin_start=None
    if cfg.train_start is not None:
        # Prevent a common-history experiment from borrowing sequence values
        # before the declared common start.
        train_origin_start=(
            pd.Timestamp(cfg.train_start)
            + pd.Timedelta(hours=cfg.input_length-1)
        )
    cal_start=pd.Timestamp(cfg.calibration_start)
    val_end=pd.Timestamp(cfg.val_end)
    tr=make_windows(
        data,cfg,origin_start=train_origin_start,
        origin_end=train_end,target_end=train_end
    )
    va=make_windows(data,cfg,tr.feature_names,origin_start=train_end+hour,
                    origin_end=cal_start-hour,target_end=cal_start-hour)
    ca=make_windows(data,cfg,tr.feature_names,origin_start=cal_start,
                    origin_end=val_end,target_end=val_end)
    te=make_windows(data,cfg,tr.feature_names,origin_start=val_end+hour,
                    target_end=data.datetime.max())
    return tr,va,ca,te

def _scale(tr,va,ca,te):
    x_scaler=ArrayStandardizer().fit(tr.X,(0,1))
    y_scaler=ArrayStandardizer().fit(tr.y,(0,))
    return (x_scaler.transform(tr.X),x_scaler.transform(va.X),
            x_scaler.transform(ca.X),x_scaler.transform(te.X),y_scaler,x_scaler)

def _metrics(predictions,y,horizons):
    return pd.concat([point_metrics(y,p,horizons,name)
                      for name,p in predictions.items()],ignore_index=True)

def _prediction_frame(times,target_times,y,predictions,horizons):
    out=pd.DataFrame({"origin_time":times.astype(str).reset_index(drop=True)})
    for j,h in enumerate(horizons):
        out[f"target_time_h{h}"]=target_times.iloc[:,j].astype(str).reset_index(drop=True)
        out[f"y_h{h}"]=y[:,j]
        for name,pred in predictions.items(): out[f"{name}_h{h}"]=pred[:,j]
    return out

def _eligible(predictions):
    return np.column_stack([np.isfinite(v).all(axis=1)
                            for v in predictions.values()]).all(axis=1)

def _numeric_context(frame,columns):
    """Convert nullable/extension columns to a finite float matrix."""
    return (
        frame[columns]
        .apply(pd.to_numeric,errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

def _training_rows(data,cfg):
    """Return the exact row interval permitted for training-only estimators."""
    mask=data.datetime.le(pd.Timestamp(cfg.train_end))
    if cfg.train_start is not None:
        mask &= data.datetime.ge(pd.Timestamp(cfg.train_start))
    return data.loc[mask]

def run_baselines(raw,root,cfg=Config()):
    """Evaluate persistence, seasonal-naive, and direct XGBoost baselines."""
    root=Path(root); data,_=prepare(raw,root,cfg); tr,va,ca,te=_datasets(data,cfg)
    train=_training_rows(data,cfg)
    xgb=fit_xgb_direct(train,TABULAR,cfg.horizons,cfg)
    predictions={
        "persistence":persistence(te.context,cfg.horizons),
        "seasonal_naive":seasonal_naive(data,te.context,cfg.horizons),
        "xgboost":predict_xgb_direct(xgb,te.context,TABULAR),
    }
    common=_eligible(predictions); y=te.y[common]
    predictions={k:v[common] for k,v in predictions.items()}
    metrics=_metrics(predictions,y,cfg.horizons)
    metrics.to_csv(root/"results/baseline_metrics.csv",index=False)
    _prediction_frame(te.origin_time.loc[common],te.target_time.loc[common],y,
                      predictions,cfg.horizons).to_csv(
                          root/"results/baseline_predictions.csv",index=False)
    return metrics

def run_experiment(raw,root,cfg=Config()):
    """Train all confirmatory models and save every reported statistic."""
    root=Path(root); seed_all(cfg.seed); data,audit=prepare(raw,root,cfg)
    tr,va,ca,te=_datasets(data,cfg)
    Xtr,Xv,Xc,Xt,ys,xs=_scale(tr,va,ca,te)
    ytr=ys.transform(tr.y); yv=ys.transform(va.y)

    tm=TimeMixer(Xtr.shape[2],len(cfg.horizons),cfg.input_length,
                 cfg.hidden_size,cfg.scales,cfg.dropout,cfg.mixer_blocks,
                 cfg.decomposition_kernel)
    tm,tm_loss,tm_history=fit_model(tm,Xtr,ytr,Xv,yv,cfg)
    tm_cal,emb_cal=_inverse_prediction(tm,Xc,ys)
    tm_test,emb_test=_inverse_prediction(tm,Xt,ys)

    lstm=LSTMForecaster(Xtr.shape[2],len(cfg.horizons),cfg.lstm_hidden_size,
                        cfg.lstm_layers,cfg.dropout)
    lstm,lstm_loss,lstm_history=fit_model(lstm,Xtr,ytr,Xv,yv,cfg)
    lstm_test,_=_inverse_prediction(lstm,Xt,ys)

    dlinear=DLinearForecaster(
        cfg.input_length,len(cfg.horizons),cfg.decomposition_kernel
    )
    dlinear,dlinear_loss,dlinear_history=fit_model(
        dlinear,Xtr,ytr,Xv,yv,cfg
    )
    dlinear_test,_=_inverse_prediction(dlinear,Xt,ys)

    ctx_cal=_numeric_context(ca.context,RESIDUAL_CONTEXT)
    ctx_test=_numeric_context(te.context,RESIDUAL_CONTEXT)
    gate_size=max(1,int(len(ca.y)*cfg.residual_gate_fraction))
    fit_end=len(ca.y)-gate_size
    if fit_end<100:
        raise ValueError("Residual-fit calibration interval is too small")
    hybrid_names=([f"base_h{h}" for h in cfg.horizons] +
                  [f"embedding_{i}" for i in range(emb_cal.shape[1])] +
                  RESIDUAL_CONTEXT)
    hybrid=HybridResidual(len(cfg.horizons),cfg).fit(
        tm_cal[:fit_end],emb_cal[:fit_end],ctx_cal[:fit_end],
        ca.y[:fit_end],hybrid_names
    ).calibrate_gate(
        tm_cal[fit_end:],emb_cal[fit_end:],ctx_cal[fit_end:],ca.y[fit_end:]
    )
    hybrid_cal=hybrid.predict(
        tm_cal[fit_end:],emb_cal[fit_end:],ctx_cal[fit_end:]
    )
    hybrid_static_test=hybrid.predict(tm_test,emb_test,ctx_test)
    hybrid_ungated_test=hybrid.predict_static(tm_test,emb_test,ctx_test)
    hybrid_test,rolling_diagnostics=rolling_bias_calibration(
        hybrid_cal,ca.y[fit_end:],hybrid_static_test,te.y,
        te.origin_time,te.target_time,cfg,
    )

    train=_training_rows(data,cfg)
    xgb_models=fit_xgb_direct(train,TABULAR,cfg.horizons,cfg)
    predictions={
        "persistence":persistence(te.context,cfg.horizons),
        "seasonal_naive":seasonal_naive(data,te.context,cfg.horizons),
        "xgboost":predict_xgb_direct(xgb_models,te.context,TABULAR),
        "dlinear":dlinear_test,
        "lstm":lstm_test,
        "timemixer":tm_test,
        "hybrid_static":hybrid_static_test,
        "hybrid":hybrid_test,
    }
    common=_eligible(predictions); y=te.y[common]
    predictions={k:v[common] for k,v in predictions.items()}
    times=te.origin_time.loc[common].reset_index(drop=True)
    target_times=te.target_time.loc[common].reset_index(drop=True)

    metrics=_metrics(predictions,y,cfg.horizons)
    metrics.to_csv(root/"results/experiment_metrics.csv",index=False)
    threshold=float(train.pm25_clean.quantile(cfg.high_pollution_quantile))
    pd.concat([high_pollution_metrics(y,p,cfg.horizons,threshold).assign(model=k)
               for k,p in predictions.items()],ignore_index=True).to_csv(
                   root/"results/high_pollution_metrics.csv",index=False)
    _prediction_frame(times,target_times,y,predictions,cfg.horizons).to_csv(
        root/"results/predictions.csv",index=False)

    bootstrap=block_bootstrap_per_horizon(
        y,predictions["hybrid"],predictions["timemixer"],cfg.horizons,
        cfg.bootstrap_block_hours,cfg.bootstrap_repetitions,cfg.seed)
    bootstrap.to_csv(root/"results/bootstrap_comparison.csv",index=False)
    dm=diebold_mariano_abs(y,predictions["hybrid"],predictions["timemixer"],
                           cfg.horizons,cfg.dm_max_lag)
    dm.to_csv(root/"results/diebold_mariano.csv",index=False)
    hybrid.shap_summary(tm_test[common],emb_test[common],ctx_test[common],
                        cfg.horizons).to_csv(root/"results/shap_summary.csv",index=False)
    pd.DataFrame({
        "horizon":cfg.horizons,
        "correction_gate_alpha":hybrid.gate_,
    }).to_csv(root/"results/residual_gate.csv",index=False)
    rolling_diagnostics.to_csv(
        root/"results/rolling_calibration_diagnostics.csv",index=False
    )

    variants={
        "robust_rolling_hybrid":predictions["hybrid"],
        "gated_static_hybrid":predictions["hybrid_static"],
        "ungated_static_hybrid":hybrid_ungated_test[common],
        "timemixer_base":predictions["timemixer"],
    }
    empty_cal=np.empty((len(ca.y),0)); empty_test=np.empty((len(te.y),0))
    specifications=[
        ("no_embedding",empty_cal,empty_test,ctx_cal,ctx_test),
        ("no_context",emb_cal,emb_test,empty_cal,empty_test),
        ("base_only",empty_cal,empty_test,empty_cal,empty_test),
    ]
    for name,ec,et,cc,ct in specifications:
        model=HybridResidual(len(cfg.horizons),cfg).fit(
            tm_cal[:fit_end],ec[:fit_end],cc[:fit_end],ca.y[:fit_end]
        ).calibrate_gate(
            tm_cal[fit_end:],ec[fit_end:],cc[fit_end:],ca.y[fit_end:]
        )
        variants[name]=model.predict(tm_test,et,ct)[common]
    pd.concat([point_metrics(y,p,cfg.horizons).assign(variant=k)
               for k,p in variants.items()],ignore_index=True).to_csv(
                   root/"results/ablation_metrics.csv",index=False)
    pd.DataFrame(tm_history).assign(model="timemixer").to_csv(
        root/"results/timemixer_history.csv",index=False)
    pd.DataFrame(lstm_history).assign(model="lstm").to_csv(
        root/"results/lstm_history.csv",index=False)
    pd.DataFrame(dlinear_history).assign(model="dlinear").to_csv(
        root/"results/dlinear_history.csv",index=False)
    np.savez(root/"results/scalers.npz",x_mean=xs.mean,x_std=xs.std,
             y_mean=ys.mean,y_std=ys.std)
    torch.save(tm.state_dict(),root/"results/timemixer.pt")
    torch.save(lstm.state_dict(),root/"results/lstm.pt")
    torch.save(dlinear.state_dict(),root/"results/dlinear.pt")

    sets={"train":tr,"neural_validation":va,"hybrid_calibration":ca,"test":te}
    run={**cfg.to_dict(),"sequence_features":tr.feature_names,
         "tabular_features":TABULAR,"context_features":CONTEXT,
         "residual_context_features":RESIDUAL_CONTEXT,
         "high_pollution_threshold":threshold,
         "timemixer_best_val_loss":tm_loss,"lstm_best_val_loss":lstm_loss,
         "dlinear_best_val_loss":dlinear_loss,
         "residual_design":{
             "de_bias":"calibration median per horizon",
             "robust_loss":cfg.residual_objective,
             "clip_quantile":cfg.residual_clip_quantile,
             "fit_windows":fit_end,
             "gate_windows":gate_size,
             "gate_alpha_by_horizon":hybrid.gate_.tolist(),
             "gate_selection":"held-out calibration MAE; alpha=0 fallback",
             "rolling_window_hours":cfg.rolling_calibration_hours,
             "maturity_rule":"target_time <= forecast origin",
         },
         "stacking":"chronological calibration holdout",
         "windows":{k:len(v.X) for k,v in sets.items()},
         "rejected_windows":{k:{"missing_input":v.rejected_missing_input,
                                  "missing_target":v.rejected_missing_target}
                             for k,v in sets.items()}}
    _json(root/"results/run_config.json",run)
    return metrics,{"bootstrap":bootstrap.to_dict("records"),
                    "diebold_mariano":dm.to_dict("records")}

def _inverse_prediction(model,X,scaler):
    pred,embedding=predict_with_embeddings(model,X)
    return scaler.inverse(pred),embedding
