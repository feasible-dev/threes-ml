param(
    [ValidateSet(42, 43)]
    [int[]]$Seeds = @(42, 43),
    [ValidateRange(1, 2147483647)]
    [int]$Steps = 200000,
    [ValidateRange(1, 10000)]
    [int]$HoldoutGames = 300,
    [ValidateRange(0, 2147483647)]
    [int]$HoldoutSeed = 300000,
    [switch]$Preview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$conditions = @(
    [pscustomobject]@{ Name = 'baseline'; Config = 'configs\ppo_200k.json' },
    [pscustomobject]@{ Name = 'stronger_updates'; Config = 'configs\ppo_200k_stronger_updates.json' },
    [pscustomobject]@{ Name = 'score_survival'; Config = 'configs\ppo_200k_score_survival.json' }
)
$batchId = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$launcher = Join-Path $PSScriptRoot 'run_experiments.ps1'
$runsRoot = Join-Path $PSScriptRoot 'results\runs'
$comparisonDir = Join-Path $PSScriptRoot "results\comparisons\${batchId}_ppo_reward_updates"

if ($Seeds.Count -ne (@($Seeds | Sort-Object -Unique)).Count) {
    throw 'Each seed must be unique.'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python not found: $python"
}

Write-Host "Batch ${batchId}: $($conditions.Count) conditions x $($Seeds.Count) seeds, $Steps transitions per run"
Write-Host 'Runs and evaluation progress will print in this terminal.'
Write-Host 'Training uses CPU; conditions run sequentially.'

Push-Location $PSScriptRoot
try {
    $models = @()
    foreach ($condition in $conditions) {
        $series = $condition.Name
        Write-Host "`n=== $($condition.Name): $($condition.Config) ===" -ForegroundColor Cyan
        & $launcher -Series $series -Seeds $Seeds -Steps $Steps -Config $condition.Config -RunStamp $batchId -Preview:$Preview
        if (-not $?) {
            throw "Condition failed: $($condition.Name)"
        }
        if ($Preview) {
            continue
        }
        foreach ($seed in $Seeds) {
            $matches = @(Get-ChildItem -LiteralPath $runsRoot -Directory |
                Where-Object { $_.Name -eq "${batchId}_${series}_s${seed}" })
            if ($matches.Count -ne 1) {
                throw "Expected one run for $series seed $seed; found $($matches.Count)."
            }
            $model = Join-Path $matches[0].FullName 'models\model.zip'
            if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
                throw "Model not found: $model"
            }
            $models += $model
        }
    }

    if ($Preview) {
        Write-Host "`nPreview complete. No training or evaluation started."
        return
    }

    $csv = Join-Path $comparisonDir 'holdout_models.csv'
    $references = Join-Path $comparisonDir 'holdout_heuristics.csv'
    $report = Join-Path $comparisonDir 'comparison.md'
    Write-Host "`n=== Held-out evaluation: $HoldoutGames games per agent ===" -ForegroundColor Cyan
    & $python src/compare_agents.py --games $HoldoutGames --seed $HoldoutSeed --csv $references
    if ($LASTEXITCODE -ne 0) {
        throw "Heuristic evaluation failed with code $LASTEXITCODE"
    }
    & $python src/evaluate_models.py @models --games $HoldoutGames --seed $HoldoutSeed --csv $csv --reference-csv $references --report $report
    if ($LASTEXITCODE -ne 0) {
        throw "Held-out evaluation failed with code $LASTEXITCODE"
    }
    Write-Host "`nComparison: $report" -ForegroundColor Green
} finally {
    Pop-Location
}
