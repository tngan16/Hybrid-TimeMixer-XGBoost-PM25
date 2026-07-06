"""Robust residual correction and leakage-safe rolling calibration."""
from collections import deque

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBRegressor


class HybridResidual:
    """Median de-bias + pseudo-Huber XGBoost + bounded correction."""
    def __init__(self,n_horizons,cfg):
        self.cfg=cfg
        self.models=[
            XGBRegressor(
                n_estimators=cfg.xgb_estimators,
                max_depth=max(3,cfg.xgb_max_depth-1),
                learning_rate=cfg.xgb_learning_rate,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample,
                min_child_weight=cfg.xgb_min_child_weight,
                reg_alpha=.01,
                reg_lambda=1.0,
                objective=cfg.residual_objective,
                random_state=cfg.seed+j,
                n_jobs=-1,
            ) for j in range(n_horizons)
        ]
        self.feature_names=None
        self.bias_=np.zeros(n_horizons)
        self.bounds_=np.column_stack([
            np.full(n_horizons,-np.inf),np.full(n_horizons,np.inf)
        ])
        self.gate_=np.ones(n_horizons,float)

    def features(self,base,emb,context):
        return np.column_stack([base,emb,context])

    def fit(self,base,emb,context,y,feature_names=None):
        z=self.features(base,emb,context)
        self.feature_names=feature_names or [
            f"feature_{i}" for i in range(z.shape[1])
        ]
        residual=y-base
        self.bias_=np.nanmedian(residual,axis=0)
        centred=residual-self.bias_
        tail=1-self.cfg.residual_clip_quantile
        self.bounds_=np.column_stack([
            np.nanquantile(centred,tail,axis=0),
            np.nanquantile(centred,1-tail,axis=0),
        ])
        for j,model in enumerate(self.models):
            model.fit(z,centred[:,j])
        return self

    def correction(self,base,emb,context):
        z=self.features(base,emb,context)
        correction=np.column_stack([model.predict(z) for model in self.models])
        correction=np.clip(correction,self.bounds_[:,0],self.bounds_[:,1])
        return self.bias_+correction

    def predict_static(self,base,emb,context):
        """Return the un-gated robust residual forecast for ablation."""
        return base+self.correction(base,emb,context)

    def calibrate_gate(self,base,emb,context,y):
        """Choose a horizon-specific correction scale on held-out calibration.

        Alpha=0 is an explicit fallback to the TimeMixer base. The locked test
        is never inspected while selecting the gate.
        """
        correction=self.correction(base,emb,context)
        grid=np.asarray(self.cfg.residual_gate_grid,float)
        for j in range(base.shape[1]):
            losses=np.asarray([
                np.nanmean(np.abs(y[:,j]-(base[:,j]+alpha*correction[:,j])))
                for alpha in grid
            ])
            self.gate_[j]=grid[int(np.nanargmin(losses))]
        return self

    def predict(self,base,emb,context):
        return base+self.gate_*self.correction(base,emb,context)

    def shap_summary(self,base,emb,context,horizons):
        z=self.features(base,emb,context)
        rows=[]
        for j,(h,model) in enumerate(zip(horizons,self.models)):
            contribution=model.get_booster().predict(
                DMatrix(z),pred_contribs=True
            )[:,:-1]
            importance=self.gate_[j]*np.mean(np.abs(contribution),axis=0)
            rows.extend({
                "horizon":int(h),
                "feature":name,
                "mean_abs_shap":float(value),
            } for name,value in zip(self.feature_names,importance))
        return pd.DataFrame(rows)


def rolling_bias_calibration(
    calibration_prediction,calibration_y,test_prediction,test_y,
    test_origin_time,test_target_time,cfg,
):
    """Use only errors whose target time is observable at forecast origin."""
    prediction=np.asarray(test_prediction,float).copy()
    y=np.asarray(test_y,float)
    initial=np.asarray(calibration_y)-np.asarray(calibration_prediction)
    origins=pd.to_datetime(pd.Series(test_origin_time),utc=True).reset_index(drop=True)
    targets=pd.DataFrame(test_target_time).apply(pd.to_datetime,utc=True)
    window=pd.Timedelta(hours=cfg.rolling_calibration_hours)
    diagnostics=[]
    for horizon_index in range(prediction.shape[1]):
        seed_errors=initial[:,horizon_index]
        seed_errors=seed_errors[np.isfinite(seed_errors)]
        clip=(np.nanquantile(np.abs(seed_errors),cfg.residual_clip_quantile)
              if len(seed_errors) else 0.0)
        # Only the active 30-day window is retained. The previous
        # implementation rescanned every matured error for every origin,
        # producing quadratic runtime on multi-year test sets.
        recent=deque()
        target_column=targets.iloc[:,horizon_index]
        pointer=0
        for row,origin in enumerate(origins):
            while pointer<row and target_column.iat[pointer]<=origin:
                error=y[pointer,horizon_index]-prediction[pointer,horizon_index]
                if np.isfinite(error):
                    recent.append((target_column.iat[pointer],float(error)))
                pointer+=1
            oldest_allowed=origin-window
            while recent and recent[0][0]<oldest_allowed:
                recent.popleft()
            recent_errors=np.fromiter(
                (error for _,error in recent),dtype=float,count=len(recent)
            )
            # Calibration errors are a cold-start prior only. Once enough
            # operational errors have matured, the update is genuinely rolling.
            needed=max(
                0,cfg.rolling_calibration_min_points-len(recent_errors)
            )
            prior=seed_errors[-needed:] if needed else np.empty(0)
            pool=np.concatenate([prior,recent_errors])
            adjustment=0.0
            if len(pool)>=cfg.rolling_calibration_min_points:
                weight=len(pool)/(len(pool)+cfg.rolling_calibration_shrinkage)
                adjustment=float(np.clip(
                    weight*np.nanmedian(pool),-clip,clip
                ))
            prediction[row,horizon_index]+=adjustment
            diagnostics.append({
                "row":row,"horizon_index":horizon_index,
                "origin_time":origin,"available_errors":len(pool),
                "rolling_adjustment":adjustment,
            })
    return prediction,pd.DataFrame(diagnostics)
