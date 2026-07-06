# Project status

## Confirmatory run completed

- Five UK stations: MY1, BIR_A4540, HP1, CHB, and KC1.
- Five random seeds (0--4), with a 30-epoch maximum: 25 station-seed runs.
- Direct horizons: 1, 6, 12, 24, and 48 hours.
- Residual safeguards: calibration-median de-biasing, pseudo-Huber XGBoost,
  calibration-only gating and clipping, and leakage-safe rolling calibration.
- Eight code-generated main figures and nine supplementary EDA panels.
- Editable Draw.io framework source and exported PDF are included.
- Machine-readable metrics, ablations, gate diagnostics, bootstrap results, and
  LaTeX result fragments are included under `results/` and `paper/`.

## Confirmatory finding

The gated hybrid improves the TimeMixer base in pooled MAE at every horizon
(42.58%, 3.86%, 1.72%, 1.20%, and 0.84%), showing that the revised residual
layer prevents most harmful long-horizon corrections. It does not outperform
the strongest matched comparator: 0 of 25 station--horizon comparisons are
wins, and the mean relative improvement against that comparator is -23.33%.
Overall pooled MAE is 3.708 for the hybrid, 4.078 for TimeMixer, and 3.271 for
LSTM. The repository therefore makes no SOTA or best-model claim.

## Before submission

1. Replace author, email, affiliation, and acknowledgement placeholders.
2. Compile `paper/main.tex` after every manuscript change.
3. Check that the compiled PDF states five stations, five seeds, 30 epochs,
   and the 0/25 strongest-comparator result.
4. If reported, run the common-history sensitivity analysis separately; do
   not imply it was completed unless its output files exist.
5. Confirm UK-Air attribution and dataset redistribution conditions.
