"""Export tables and narrative only from a complete gated-v2 experiment."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def _latex_name(name):
    return str(name).replace("_", r"\_")


def main():
    metrics_path = RESULTS / "multistation_seed_metrics.csv"
    effect_path = RESULTS / "station_effectiveness.csv"
    required = [metrics_path, effect_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing confirmatory files: " + ", ".join(missing))

    metrics = pd.read_csv(metrics_path)
    effects = pd.read_csv(effect_path)
    if set(metrics["seed"].unique()) != {0, 1, 2, 3, 4}:
        raise SystemExit("Results are incomplete: seeds must be exactly 0--4.")
    if set(metrics["epochs"].unique()) != {30}:
        raise SystemExit("Results are not the pre-specified 30-epoch runs.")
    if (
        "method_version" not in metrics
        or set(metrics["method_version"].unique()) != {"gated_residual_v2"}
    ):
        raise SystemExit("Results are not calibration-gated residual v2.")
    if metrics[["station_id", "seed"]].drop_duplicates().shape[0] != 25:
        raise SystemExit("Expected exactly 25 station--seed runs.")

    summary = (
        metrics.groupby(["model", "horizon"])["MAE"]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall = metrics.groupby("model")["MAE"].mean().sort_values()
    best_model = str(overall.index[0])
    best_mae = float(overall.iloc[0])
    hybrid_mae = float(overall["hybrid"])
    timemixer_mae = float(overall["timemixer"])
    wins = int(effects["hybrid_wins"].sum())
    total = int(len(effects))
    mean_gain = float(effects["improvement_percent"].mean())
    end = r" \\"

    lines = [
        "% Auto-generated from complete gated-v2 multi-station results.",
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\caption{Locked-test MAE pooled over station--seed runs. Parentheses "
        r"show dispersion across both stations and seeds; station-specific "
        r"comparisons are retained in the machine-readable results.}",
        r"\label{tab:multistation-mae}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Model & 1 h & 6 h & 12 h & 24 h & 48 h" + end,
        r"\midrule",
    ]
    horizons = [1, 6, 12, 24, 48]
    for model in summary["model"].drop_duplicates():
        part = summary[summary.model.eq(model)].set_index("horizon")
        values = [
            f"{part.loc[h, 'mean']:.3f} ({part.loc[h, 'std']:.3f})"
            if h in part.index else "--"
            for h in horizons
        ]
        lines.append(_latex_name(model) + " & " + " & ".join(values) + end)
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
        f"The gated rolling hybrid won {wins} of {total} matched "
        f"station--horizon comparisons. Its mean relative MAE improvement "
        f"against the strongest non-hybrid comparator was {mean_gain:.2f}\\%.",
    ]
    (PAPER / "generated_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    supports = wins > total / 2 and mean_gain > 0
    verdict = (
        "The result supports effectiveness within the evaluated model set."
        if supports else
        "The result does not support a superiority or state-of-the-art claim."
    )
    abstract = (
        f"The confirmatory experiment completed five stations, five seeds, "
        f"and a 30-epoch maximum. The gated rolling hybrid won {wins} of "
        f"{total} station--horizon comparisons, with mean relative MAE "
        f"improvement of {mean_gain:.2f}\\% against the strongest matched "
        f"non-hybrid comparator. The lowest pooled MAE was {best_mae:.3f}, "
        f"obtained by {_latex_name(best_model)}. {verdict}"
    )
    (PAPER / "generated_abstract.tex").write_text(
        abstract + "\n", encoding="utf-8"
    )

    gate_path = RESULTS / "multistation_residual_gate.csv"
    gate_sentence = ""
    if gate_path.exists():
        gate = pd.read_csv(gate_path)
        zero_rate = 100 * gate.correction_gate_alpha.eq(0).mean()
        mean_alpha = gate.correction_gate_alpha.mean()
        gate_sentence = (
            f"The held-out gate selected zero correction in {zero_rate:.1f}\\% "
            f"of station--seed--horizon fits and had mean alpha "
            f"{mean_alpha:.2f}, quantifying how often fallback protection was "
            f"needed. "
        )
    discussion = (
        f"Across all station--seed--horizon rows, the hybrid pooled MAE was "
        f"{hybrid_mae:.3f}, compared with {timemixer_mae:.3f} for TimeMixer "
        f"and {best_mae:.3f} for {_latex_name(best_model)}. "
        + gate_sentence
        + "The correction gate was selected without locked-test access, so "
        "any remaining performance gap reflects generalization rather than "
        "post-hoc test tuning. Station-level heterogeneity, event behavior, "
        "paired uncertainty, and the ablations must be considered together."
    )
    (PAPER / "generated_discussion.tex").write_text(
        discussion + "\n", encoding="utf-8"
    )
    conclusion = (
        f"The gated hybrid won {wins} of {total} matched comparisons and "
        f"achieved mean relative improvement of {mean_gain:.2f}\\%. "
        f"The best pooled model was {_latex_name(best_model)} "
        f"(MAE {best_mae:.3f}). {verdict} Claims are therefore restricted to "
        "the completed matched protocol and do not imply global SOTA."
    )
    (PAPER / "generated_conclusion.tex").write_text(
        conclusion + "\n", encoding="utf-8"
    )
    print(PAPER / "generated_results.tex")


if __name__ == "__main__":
    main()
