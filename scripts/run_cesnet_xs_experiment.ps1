param(
    [int]$TrainRows = 100000,
    [int]$CalibrationRows = 20000,
    [int]$KnownTestRows = 20000,
    [int]$UnknownTestRows = 20000,
    [int]$Epochs = 15
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Scripts = Join-Path $ProjectRoot ".venv\Scripts"
$Prepared = Join-Path $ProjectRoot "data\processed\datazoo_xs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project environment not found. Create .venv and install the project first."
}

& (Join-Path $Scripts "driftmamba-export-datazoo.exe") `
    --data-root (Join-Path $ProjectRoot "data\raw\cesnet-quic22") `
    --output-directory $Prepared `
    --train-rows $TrainRows `
    --calibration-rows $CalibrationRows `
    --known-test-rows $KnownTestRows `
    --unknown-test-rows $UnknownTestRows

& (Join-Path $Scripts "driftmamba-train-baselines.exe") `
    --data-directory $Prepared `
    --output-directory (Join-Path $ProjectRoot "reports\baselines") `
    --models-directory (Join-Path $ProjectRoot "models\baselines")

& (Join-Path $Scripts "driftmamba-train-deep.exe") `
    --data-directory $Prepared `
    --models-directory (Join-Path $ProjectRoot "models\deep") `
    --reports-directory (Join-Path $ProjectRoot "reports\deep") `
    --epochs $Epochs

& $Python -m pytest -q

Write-Host "Experiment complete. Review reports and run the notebooks in the notebooks directory."
