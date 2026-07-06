"""Compare station-available and common-2021 training-history experiments."""
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
    full_mean = full.groupby(keys, as_index=False)[available].mean()
    common_mean = common.groupby(keys, as_index=False)[available].mean()
    comparison = full_mean.merge(
        common_mean, on=keys, suffixes=("_available_history", "_common_2021")
    )
    for metric in available:
        comparison[f"{metric}_delta_common_minus_available"] = (
            comparison[f"{metric}_common_2021"]
            - comparison[f"{metric}_available_history"]
        )
    comparison.to_csv(
        RESULTS / "history_sensitivity_comparison.csv", index=False
    )

    hybrid = comparison.loc[comparison.model.eq("hybrid")].copy()
    hybrid.to_csv(
        RESULTS / "history_sensitivity_hybrid.csv", index=False
    )
    print(hybrid.to_string(index=False))


if __name__ == "__main__":
    main()
