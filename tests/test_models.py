import torch
import numpy as np
import pandas as pd
from dataclasses import replace
from pm25forecast.config import Config
from pm25forecast.dlinear import DLinearForecaster
from pm25forecast.hybrid import HybridResidual,rolling_bias_calibration
from pm25forecast.lstm import LSTMForecaster
from pm25forecast.timemixer import TimeMixer

def test_lstm_output_and_embedding_shapes():
    model=LSTMForecaster(9,5,hidden=16,layers=2)
    y,e=model(torch.randn(4,48,9),return_embedding=True)
    assert y.shape==(4,5) and e.shape==(4,16)

def test_timemixer_output_and_embedding_shapes():
    model=TimeMixer(9,5,input_length=48,hidden=16,scales=(1,2,4),blocks=1,decomposition_kernel=7)
    y,e=model(torch.randn(4,48,9),return_embedding=True)
    assert y.shape==(4,5) and e.shape==(4,48)
    assert torch.isfinite(y).all()

def test_dlinear_output_shape():
    model=DLinearForecaster(48,5,7)
    y,e=model(torch.randn(4,48,9),return_embedding=True)
    assert y.shape==(4,5) and e.shape==(4,2)

def test_robust_residual_is_finite():
    rng=np.random.default_rng(7)
    base=rng.normal(10,2,(60,2)); y=base+rng.normal(0,.5,(60,2))
    emb=rng.normal(size=(60,3)); context=rng.normal(size=(60,2))
    cfg=replace(Config(),horizons=(1,6),xgb_estimators=10)
    prediction=HybridResidual(2,cfg).fit(base,emb,context,y).predict(
        base,emb,context
    )
    assert prediction.shape==y.shape and np.isfinite(prediction).all()

def test_residual_gate_can_fall_back_to_base():
    rng=np.random.default_rng(11)
    base=np.full((80,1),10.0)
    y_fit=base[:40]+2.0
    embedding=rng.normal(size=(80,2))
    context=rng.normal(size=(80,2))
    cfg=replace(
        Config(),horizons=(1,),xgb_estimators=10,
        residual_gate_grid=(0.0,0.5,1.0),
    )
    model=HybridResidual(1,cfg).fit(
        base[:40],embedding[:40],context[:40],y_fit
    )
    # On held-out calibration, the uncorrected base is exact, so alpha=0.
    model.calibrate_gate(
        base[40:],embedding[40:],context[40:],base[40:]
    )
    assert model.gate_[0]==0.0
    assert np.allclose(
        model.predict(base[40:],embedding[40:],context[40:]),base[40:]
    )

def test_rolling_calibration_does_not_read_future_target():
    cfg=replace(
        Config(),horizons=(1,),rolling_calibration_hours=24,
        rolling_calibration_min_points=1,rolling_calibration_shrinkage=0,
    )
    cal_pred=np.zeros((4,1)); cal_y=np.ones((4,1))
    test_pred=np.zeros((5,1)); test_y=np.arange(5,dtype=float).reshape(-1,1)
    origins=pd.date_range("2025-01-01",periods=5,freq="h",tz="UTC")
    targets=pd.DataFrame({0:origins+pd.Timedelta(hours=1)})
    first,_=rolling_bias_calibration(
        cal_pred,cal_y,test_pred,test_y,origins,targets,cfg
    )
    changed=test_y.copy(); changed[-1,0]=9999
    second,_=rolling_bias_calibration(
        cal_pred,cal_y,test_pred,changed,origins,targets,cfg
    )
    assert np.allclose(first[:4],second[:4])
