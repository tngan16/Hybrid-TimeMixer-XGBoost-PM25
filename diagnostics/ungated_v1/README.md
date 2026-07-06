# Ungated residual v1 diagnostic

This directory preserves the completed five-station, five-seed, 30-epoch
experiment that motivated calibration-gated residual v2.

The rolling v1 hybrid won 0 of 25 station--horizon comparisons and had mean
relative MAE improvement of -65.23% against the strongest matched non-hybrid
comparator. Static residual correction was worse than rolling correction,
showing that rolling calibration mitigated—but did not solve—over-correction.

These files are retained for auditability and must not be reported as results
of gated residual v2. Final manuscript outputs are generated only from root
`results/` files carrying `method_version=gated_residual_v2`.
