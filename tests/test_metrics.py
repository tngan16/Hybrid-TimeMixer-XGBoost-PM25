import numpy as np
from pm25forecast.metrics import point_metrics, high_pollution_metrics, diebold_mariano_abs

def test_perfect_forecast_metrics():
    y=np.array([[1.,2.],[3.,4.],[5.,6.]])
    m=point_metrics(y,y,(1,6),"perfect")
    assert (m.MAE==0).all() and (m.RMSE==0).all()
    assert np.allclose(m.R2,1)

def test_high_pollution_event_scores():
    y=np.array([[1.],[10.],[12.],[2.]])
    p=np.array([[2.],[11.],[5.],[9.]])
    m=high_pollution_metrics(y,p,(1,),9).iloc[0]
    assert m.n_high==2 and m.event_recall==0.5 and m.false_alarm_rate==0.5

def test_dm_output_has_adjusted_probabilities():
    rng=np.random.default_rng(4); y=rng.normal(size=(200,2))
    a=y+rng.normal(0,.2,size=y.shape); b=y+rng.normal(0,.5,size=y.shape)
    out=diebold_mariano_abs(y,a,b,(1,6),max_lag=6)
    assert set(["dm_stat","p_value","holm_p_value"]).issubset(out)
    assert out.holm_p_value.between(0,1).all()
