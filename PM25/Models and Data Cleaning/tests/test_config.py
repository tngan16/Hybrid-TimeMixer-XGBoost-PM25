import pytest
from dataclasses import replace
from pm25forecast.config import Config

def test_default_config_is_valid_and_serialisable():
    cfg=Config(); assert cfg.to_dict()["horizons"]==(1,6,12,24,48)
    assert cfg.with_seed(7).seed==7

def test_invalid_horizons_are_rejected():
    with pytest.raises(ValueError): replace(Config(),horizons=(6,1))

def test_common_history_start_is_serialised():
    cfg=replace(Config(),train_start="2021-01-01 00:00:00+00:00")
    assert cfg.to_dict()["train_start"].startswith("2021-01-01")

def test_training_start_must_precede_training_end():
    with pytest.raises(ValueError):
        replace(Config(),train_start="2023-01-01 00:00:00+00:00")

def test_invalid_residual_gate_is_rejected():
    with pytest.raises(ValueError):
        replace(Config(),residual_gate_grid=(-0.1,0.5,1.0))
