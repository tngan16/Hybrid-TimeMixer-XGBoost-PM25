# SOTA claim policy

The completed confirmatory run does not establish SOTA. The gated hybrid
improves TimeMixer at every pooled horizon, but wins 0/25 station--horizon
comparisons against the strongest matched non-hybrid comparator. SOTA is
specific to dataset, split, horizon, and metric.

Required matched comparators are persistence, seasonal-naive, direct XGBoost,
DLinear, LSTM, and TimeMixer. For a publication-level global SOTA claim, also
run the official PatchTST implementation on the same forecast origins,
horizons, and five seeds.

Permitted conclusions for the present evidence are:

- the residual safeguards improve the TimeMixer base at all pooled horizons;
- calibration gating reduces harmful residual over-correction;
- the hybrid does not beat the strongest comparator in this benchmark;
- LSTM has the lowest overall pooled MAE among the evaluated models.

Do not copy scores from unrelated benchmark datasets and call them a matched
comparison.
