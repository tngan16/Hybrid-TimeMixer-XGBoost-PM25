import numpy as np
import pandas as pd
from scipy import stats

def point_metrics(y_true,y_pred,horizons,model=None):
    rows=[]
    for j,h in enumerate(horizons):
        a=np.asarray(y_true[:,j],float); b=np.asarray(y_pred[:,j],float); ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]; err=a-b
        row={"horizon":int(h),"n":int(len(a)),"MAE":float(np.mean(np.abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),"sMAPE":float(np.mean(200*np.abs(err)/(np.abs(a)+np.abs(b)+1e-6))),"R2":float(1-np.sum(err**2)/(np.sum((a-a.mean())**2)+1e-9)),"Bias":float(np.mean(b-a))}
        if model is not None: row["model"]=model
        rows.append(row)
    return pd.DataFrame(rows)

def high_pollution_metrics(y_true,y_pred,horizons,threshold):
    rows=[]
    for j,h in enumerate(horizons):
        a,b=y_true[:,j],y_pred[:,j]; mask=a>=threshold
        rows.append({"horizon":int(h),"threshold":float(threshold),"n_high":int(mask.sum()),"high_MAE":float(np.mean(np.abs(a[mask]-b[mask]))) if mask.any() else np.nan,"event_recall":float(np.mean(b[mask]>=threshold)) if mask.any() else np.nan,"false_alarm_rate":float(np.mean(b[~mask]>=threshold)) if (~mask).any() else np.nan})
    return pd.DataFrame(rows)

def block_bootstrap_per_horizon(y,p_new,p_ref,horizons,block=24,n_boot=1000,seed=0):
    rng=np.random.default_rng(seed); rows=[]
    for j,h in enumerate(horizons):
        d=np.abs(y[:,j]-p_new[:,j])-np.abs(y[:,j]-p_ref[:,j]); n=len(d); samples=[]; blocks=max(1,int(np.ceil(n/block)))
        for _ in range(n_boot):
            starts=rng.integers(0,max(1,n-block+1),size=blocks); idx=np.concatenate([np.arange(s,min(s+block,n)) for s in starts])[:n]; samples.append(d[idx].mean())
        rows.append({"horizon":int(h),"mean_diff":float(d.mean()),"ci_low":float(np.quantile(samples,.025)),"ci_high":float(np.quantile(samples,.975))})
    return pd.DataFrame(rows)

def diebold_mariano_abs(y,p_new,p_ref,horizons,max_lag=24):
    rows=[]
    for j,h in enumerate(horizons):
        d=np.abs(y[:,j]-p_new[:,j])-np.abs(y[:,j]-p_ref[:,j]); d=d[np.isfinite(d)]; n=len(d); mean=d.mean(); gamma0=np.mean((d-mean)**2); variance=gamma0
        lag=min(max_lag,n-1)
        for k in range(1,lag+1):
            gamma=np.mean((d[k:]-mean)*(d[:-k]-mean)); variance+=2*(1-k/(lag+1))*gamma
        statistic=mean/np.sqrt(max(variance/n,1e-12)); p=2*stats.norm.sf(abs(statistic)); rows.append({"horizon":int(h),"dm_stat":float(statistic),"p_value":float(p)})
    frame=pd.DataFrame(rows); order=np.argsort(frame.p_value.to_numpy()); adjusted=np.empty(len(frame)); running=0.0
    for rank,idx in enumerate(order): running=max(running,min(1.0,(len(frame)-rank)*frame.p_value.iloc[idx])); adjusted[idx]=running
    frame["holm_p_value"]=adjusted; return frame
