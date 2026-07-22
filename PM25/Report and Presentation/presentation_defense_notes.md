# Presentation defense notes

This project should be presented as a leakage-safe empirical study, not as a
state-of-the-art claim. The strongest message is:

> Simple baselines remain highly competitive for PM2.5-only forecasting. The
> gated residual layer improves the compact TimeMixer base and prevents harmful
> over-correction, but it does not beat the strongest non-hybrid comparator in
> this implementation.

## How to explain negative PM2.5 source values

PM2.5 concentration cannot be physically negative. Negative values in the raw
monitoring files are treated as invalid source readings, usually caused by
instrument noise, calibration behaviour, or recording artifacts.

The pipeline does not use negative PM2.5 as real pollution concentration:

- raw values are preserved for auditability;
- negative values are masked from the modelling target;
- long invalid gaps are not interpolated;
- windows crossing unresolved missingness are excluded.

Suggested speaking line:

> The negative PM2.5 readings are not model outputs and not real air pollution
> levels. They are data-quality artifacts. We keep them in the audit trail, but
> we do not train or evaluate the model as if negative pollution were real.

## How to explain the negative model result

Avoid saying “mean improvement = -23.33%” in the presentation. It sounds like
the method is simply broken. Use this wording instead:

> Against the strongest matched comparator, the hybrid has a 23.33% average MAE
> gap and wins 0 out of 25 station-horizon comparisons. However, it improves the
> compact TimeMixer base and greatly reduces ungated residual over-correction.

This separates two questions:

1. Does gating improve the TimeMixer base? Yes.
2. Does the full hybrid beat the strongest baseline? No.

## Main defendable contribution

The defendable contribution is not “hybrid wins”. The defendable contribution is:

- a reproducible PM2.5-only forecasting pipeline;
- five-station, five-horizon, five-seed evaluation;
- leakage-safe chronological split with separate residual calibration;
- evidence that DLinear/LSTM/XGBoost are strong PM2.5-only baselines;
- evidence that ungated residual correction can over-correct;
- evidence that a held-out gate can abstain and reduce residual damage.

## If asked why TimeMixer is weak

Suggested answer:

> The TimeMixer implementation in this student project is a compact research
> adaptation, not a full benchmark reproduction. It underperforms persistence in
> pooled MAE, which limits the hybrid. This is why the paper treats the result as
> a residual-calibration lesson rather than a superiority claim. A stronger base
> TimeMixer is the first next experiment.

## If asked which model is best

Use careful wording:

> XGBoost is strongest at the 1-hour horizon. LSTM and DLinear are practically
> tied overall, with only 0.011 MAE difference in pooled MAE. Therefore, the
> safest conclusion is that simple/recurrent baselines are stronger than the
> current hybrid under PM2.5-only inputs.

