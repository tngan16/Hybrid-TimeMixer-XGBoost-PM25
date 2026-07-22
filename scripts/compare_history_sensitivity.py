"""Compare station-available and common-2021 training-history experiments.

Research-question link:
    RQ3: Are conclusions stable when every station uses the same 2021-2025
    history instead of each station's full available history?

The analysis uses pandas SQL-style GROUP BY and JOIN operations. It exports
both the complete model comparison and the hybrid-only sensitivity table.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    full_path = RESULTS / "multistation_seed_metrics.csv"
    common_path = RESULTS / "common_2021_multistation_seed_metrics.csv"
    missing = [str(p) for p in (full_path, common_path) if not p.exists()]
    if missing:
        raise SystemExit("Missing completed experiment files: " + ", ".join(missing))

    full = pd.read_csv(full_path)
    common = pd.read_csv(common_path)
    keys = ["station_id", "model", "horizon"]
    metrics = ["MAE", "RMSE", "sMAPE", "R2", "Bias"]
    available = [m for m in metrics if m in full.columns and m in common.columns]

    # RQ3 step 1: GROUP BY station, model, and horizon under each history setup.
    full_mean = full.groupby(keys, as_index=False)[available].mean()
    common_mean = common.groupby(keys, as_index=False)[available].mean()

    # RQ3 step 2: JOIN available-history results with common-2021 results.
    comparison = full_mean.merge(
        common_mean, on=keys, suffixes=("_available_history", "_common_2021")
    )

    # RQ3 step 3: compute sensitivity deltas. Small deltas mean the conclusion
    # does not depend strongly on unequal training-history length.
    for metric in available:
        comparison[f"{metric}_delta_common_minus_available"] = (
            comparison[f"{metric}_common_2021"]
            - comparison[f"{metric}_available_history"]
        )
    comparison.to_csv(
        RESULTS / "history_sensitivity_comparison.csv", index=False
    )

    # RQ3 final answer for the proposed model: hybrid-only sensitivity table.
    hybrid = comparison.loc[comparison.model.eq("hybrid")].copy()
    hybrid.to_csv(
        RESULTS / "history_sensitivity_hybrid.csv", index=False
    )
    print(hybrid.to_string(index=False))


if __name__ == "__main__":
    main()
