# Historical launcher; see scripts/README.md. Uses current source unless run from an archived snapshot.
param(
    [int[]]$Seeds = @(52, 53),
    [ValidateRange(1, 2147483647)]
    [int]$Steps = 300000,
    [ValidateRange(1, 10000)]
    [int]$ComparisonGames = 300,
    [ValidateRange(0, 2147483647)]
    [int]$ComparisonSeed = 500000,
    [switch]$Preview
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Seeds.Count -eq 0 -or $Seeds.Count -ne @($Seeds | Sort-Object -Unique).Count) {
    throw 'Provide at least one seed, with no duplicates.'
}

$batchId = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$launcher = Join-Path $PSScriptRoot 'run_experiments.ps1'
$comparisonDir = Join-Path $ProjectRoot "results\comparisons\${batchId}_article_merge"
$conditions = @(
    [pscustomobject]@{ Name = 'article_merge_64'; Config = 'configs\article_merge_64.json' },
    [pscustomobject]@{ Name = 'article_merge_256'; Config = 'configs\article_merge_256.json' }
)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python not found: $python"
}

Write-Host "Batch $batchId | $($conditions.Count) conditions x $($Seeds.Count) seeds | $Steps requested transitions per run"
Write-Host 'Training and evaluations print in this terminal. Runs use CPU and execute sequentially.'

Push-Location $ProjectRoot
try {
    $models = @()
    foreach ($condition in $conditions) {
        Write-Host "`n=== $($condition.Name) ===" -ForegroundColor Cyan
        & $launcher -Series $condition.Name -Config $condition.Config -Seeds $Seeds -Steps $Steps -RunStamp $batchId -Preview:$Preview
        if (-not $?) {
            throw "Condition failed: $($condition.Name)"
        }
        if ($Preview) { continue }

        foreach ($seed in $Seeds) {
            $runName = "${batchId}_$($condition.Name)_s${seed}"
            $model = Join-Path $ProjectRoot "results\runs\$runName\models\model.zip"
            if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
                throw "Model not found: $model"
            }
            $models += $model
        }
    }

    if ($Preview) {
        Write-Host "`nPreview complete; no training started."
        return
    }

    $heuristics = Join-Path $comparisonDir 'heuristics.csv'
    $scores = Join-Path $comparisonDir 'models.csv'
    $report = Join-Path $comparisonDir 'comparison.md'
    Write-Host "`n=== Comparison: $ComparisonGames games per agent ===" -ForegroundColor Cyan
    & $python src/compare_agents.py --games $ComparisonGames --seed $ComparisonSeed --csv $heuristics
    if ($LASTEXITCODE -ne 0) { throw "Heuristic evaluation failed: $LASTEXITCODE" }
    & $python src/evaluate_models.py @models --games $ComparisonGames --seed $ComparisonSeed --csv $scores --reference-csv $heuristics --report $report
    if ($LASTEXITCODE -ne 0) { throw "Model evaluation failed: $LASTEXITCODE" }
    & $python src/attach_comparison.py $comparisonDir --models-csv models.csv --heuristics-csv heuristics.csv
    if ($LASTEXITCODE -ne 0) { throw "Could not attach comparison: $LASTEXITCODE" }
    Write-Host "`nComparison: $report" -ForegroundColor Green
} finally {
    Pop-Location
}
