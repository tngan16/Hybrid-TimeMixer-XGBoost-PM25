$ErrorActionPreference = "Stop"

function Assert-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath "data/stations.csv")) {
    throw "Missing data/stations.csv"
}

python scripts/run_multistation.py `
    --epochs 30 `
    --seeds 0 1 2 3 4 `
    --train-start "2021-01-01 00:00:00+00:00" `
    --output-prefix "common_2021_"
Assert-Success "Common-history sensitivity experiment"

python scripts/compare_history_sensitivity.py
Assert-Success "History sensitivity comparison"

Write-Host "Completed common-history sensitivity: 2021-01-01 through 2022-12-31 training."
