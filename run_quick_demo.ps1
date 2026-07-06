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
python scripts/run_full_study.py --epochs 3 --seed 42
Assert-Success "Quick experiment"
Write-Host "Quick study completed successfully. Open figures/ and results/."
