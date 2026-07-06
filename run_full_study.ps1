$ErrorActionPreference = "Stop"
function Assert-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}
python -m pip install -e ".[dev,app]"
Assert-Success "Dependency installation"
python scripts/validate_environment.py
Assert-Success "Environment validation"
python -m pytest
Assert-Success "Test suite"
if (-not (Test-Path -LiteralPath "data/stations.csv")) {
    throw "Create data/stations.csv from data/stations.example.csv after downloading the additional station datasets."
}
python scripts/run_multistation.py --epochs 30 --seeds 0 1 2 3 4
Assert-Success "Multi-station five-seed experiment"
python scripts/make_figures.py
Assert-Success "Figure generation"
python scripts/export_framework_pdf.py
Assert-Success "Framework PDF generation"
python scripts/export_latex_results.py
Assert-Success "LaTeX result export"
Write-Host "Completed: all configured stations x 5 seeds x 30 epochs."
