"""Central, serialisable experiment configuration."""
from dataclasses import asdict, dataclass, replace

@dataclass(frozen=True)
class Config:
    """Every choice that can affect a reported result.

    The validation era is divided into neural-validation and later hybrid-
    calibration intervals, avoiding reuse of early-stopping timestamps.
    """
    input_length: int = 168
    horizons: tuple[int, ...] = (1, 6, 12, 24, 48)
    train_start: str | None = None
    train_end: str = "2022-12-31 23:00:00+00:00"
    calibration_start: str = "2024-01-01 00:00:00+00:00"
    val_end: str = "2024-06-30 23:00:00+00:00"
    short_gap_limit: int = 3
    hidden_size: int = 64
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    scales: tuple[int, ...] = (1, 2, 4, 8)
    mixer_blocks: int = 2
    decomposition_kernel: int = 25
    dropout: float = 0.10
    batch_size: int = 128
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 42
    xgb_estimators: int = 600
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.04
    xgb_subsample: float = 0.80
    xgb_colsample: float = 0.80
    xgb_min_child_weight: float = 3.0
    residual_objective: str = "reg:pseudohubererror"
    residual_clip_quantile: float = 0.995
    residual_gate_fraction: float = 0.30
    residual_gate_grid: tuple[float, ...] = (
        0.0, 0.10, 0.25, 0.50, 0.75, 1.0
    )
    rolling_calibration_hours: int = 720
    rolling_calibration_min_points: int = 168
    rolling_calibration_shrinkage: float = 168.0
    high_pollution_quantile: float = 0.90
    bootstrap_repetitions: int = 1000
    bootstrap_block_hours: int = 24
    dm_max_lag: int = 24

    def __post_init__(self):
        if self.input_length < max(self.scales):
            raise ValueError("input_length must be at least the largest scale")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if self.short_gap_limit < 0:
            raise ValueError("short_gap_limit cannot be negative")
        if not 0 < self.high_pollution_quantile < 1:
            raise ValueError("high_pollution_quantile must be between zero and one")
        if not 0.5 < self.residual_clip_quantile < 1:
            raise ValueError("residual_clip_quantile must be between 0.5 and one")
        if not 0 < self.residual_gate_fraction < 0.5:
            raise ValueError("residual_gate_fraction must be between 0 and 0.5")
        if (
            not self.residual_gate_grid
            or min(self.residual_gate_grid) < 0
            or max(self.residual_gate_grid) > 1
        ):
            raise ValueError("residual_gate_grid values must be in [0, 1]")
        if self.rolling_calibration_hours <= 0:
            raise ValueError("rolling_calibration_hours must be positive")
        if self.rolling_calibration_min_points < 1:
            raise ValueError("rolling_calibration_min_points must be positive")
        if self.train_start is not None and not (
            self.train_start < self.train_end
        ):
            raise ValueError("train_start must be earlier than train_end")
        if not (self.train_end < self.calibration_start < self.val_end):
            raise ValueError("split timestamps must be chronological")

    def to_dict(self):
        return asdict(self)

    def with_seed(self, seed):
        return replace(self, seed=seed)
