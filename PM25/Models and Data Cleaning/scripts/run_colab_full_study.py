"""Run the complete Colab-friendly study protocol.

This wrapper avoids Windows PowerShell scripts on Colab and executes:
1) environment validation,
2) tests,
3) station preprocessing,
4) primary multi-station experiment,
5) common-2021 history-sensitivity experiment,
6) sensitivity comparison,
7) figure generation,
8) LaTeX result export,
9) framework PDF export.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(step: str, command: list[str]) -> None:
    print(f"\n[RUN] {step}")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest when the Colab session is time-constrained.",
    )
    args = parser.parse_args()

    py = sys.executable
    seed_args = [str(seed) for seed in args.seeds]
    epoch_args = ["--epochs", str(args.epochs)]

    run("Environment validation", [py, "scripts/validate_environment.py"])
    if not args.skip_tests:
        run("Test suite", [py, "-m", "pytest"])

    run("Station preprocessing", [py, "scripts/preprocess_stations.py"])

    run(
        "Primary multi-station experiment",
        [py, "scripts/run_multistation.py", *epoch_args, "--seeds", *seed_args],
    )

    run(
        "Common-2021 history-sensitivity experiment",
        [
            py,
            "scripts/run_multistation.py",
            *epoch_args,
            "--seeds",
            *seed_args,
            "--train-start",
            "2021-01-01 00:00:00+00:00",
            "--output-prefix",
            "common_2021_",
        ],
    )

    run("History sensitivity comparison", [py, "scripts/compare_history_sensitivity.py"])
    run("Figure generation", [py, "scripts/make_figures.py"])
    run("LaTeX result export", [py, "scripts/export_latex_results.py"])
    run("Framework PDF export", [py, "scripts/export_framework_pdf.py"])

    print("\n[DONE] Full study completed.")
    print("Review results/, figures/, diagnostics/, and paper/generated_results.tex.")


if __name__ == "__main__":
    main()
