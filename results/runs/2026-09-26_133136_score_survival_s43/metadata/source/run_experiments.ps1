param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Series,

    [Parameter(Mandatory = $true)]
    [int[]]$Seeds,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$Steps,

    [string]$Config = 'configs\ppo_200k.json',
    [ValidatePattern('^\d{4}-\d{2}-\d{2}_\d{6}$')]
    [string]$RunStamp = (Get-Date -Format 'yyyy-MM-dd_HHmmss'),
    [switch]$Preview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$configPath = Join-Path $PSScriptRoot $Config
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python not found: $python"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Configuration not found: $configPath"
}
if ($Seeds.Count -ne (@($Seeds | Sort-Object -Unique)).Count) {
    throw 'Each seed must be unique within a series.'
}

$planned = @(
    foreach ($seed in $Seeds) {
        $runName = "${RunStamp}_${Series}_s${seed}"
        $runDir = Join-Path $PSScriptRoot "results\runs\$runName"
        if (Test-Path -LiteralPath $runDir) {
            throw "Run directory already exists: $runDir. Choose another series name."
        }
        [pscustomobject]@{ Seed = $seed; Name = $runName; Directory = $runDir }
    }
)

Write-Host "Series: $Series | $($planned.Count) run(s) | $Steps requested transitions per run"
Write-Host "Configuration: $configPath"
foreach ($run in $planned) {
    Write-Host "  $($run.Name) -> $($run.Directory)"
}
if ($Preview) {
    Write-Host 'Preview only; no training started.'
    return
}

Push-Location $PSScriptRoot
try {
    foreach ($run in $planned) {
        Write-Host "`n=== Starting $($run.Name) ===" -ForegroundColor Cyan
        & $python src/train.py --config $configPath --run-dir $run.Directory --seed $run.Seed --steps $Steps
        $trainExitCode = $LASTEXITCODE
        if ($trainExitCode -ne 0) {
            & $python src/experiment_catalog.py
            throw "Training failed for $($run.Name) (exit code $trainExitCode)."
        }
        Write-Host "=== Finished $($run.Name): $($run.Directory)\FICHE.md ===" -ForegroundColor Green
    }
} finally {
    Pop-Location
}
