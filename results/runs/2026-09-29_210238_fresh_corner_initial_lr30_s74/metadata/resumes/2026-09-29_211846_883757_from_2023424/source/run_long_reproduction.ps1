param(
    [string[]]$Runs = @(
        'results\runs\2026-09-26_162446_recurrent_v2_curriculum_decay_s72',
        'results\runs\2026-09-26_162446_recurrent_v2_curriculum_s72'
    ),
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 90000000,
    [ValidateRange(1, 10000)][int]$EvaluationGames = 1000,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $Runs.Count -or (@($Runs | Select-Object -Unique).Count -ne $Runs.Count)) {
        throw 'Choose at least one run, without duplicates.'
    }
    $python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
    if (-not (Test-Path -LiteralPath 'external_models\kiok\onnx_model.onnx')) {
        throw 'Published reference missing; see external_models/kiok/README.md.'
    }
    Write-Host "Long continuation | $AdditionalSteps ADDITIONAL transitions per run | $($Runs.Count) runs"
    Write-Host 'Latest complete checkpoints; sequential GPU training; progress stays visible in this terminal.'
    Write-Host 'Default budget: approximately 100M total per run, about 10-12 hours combined at the measured throughput.'
    Write-Host 'Rerunning this command adds the requested budget AGAIN. For interruptions, resume the unfinished run separately.'

    # Inspect all checkpoints first. The continuation launcher enforces source,
    # dependency and active-process checks again before executing each job.
    foreach ($run in $Runs) {
        & .\resume_reproduction.ps1 -Run $run -AdditionalSteps $AdditionalSteps -Preview
    }
    $stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss_fff'
    $comparison = "results\comparisons\${stamp}_long_recurrent_v2"
    if (-not $Preview) {
        foreach ($run in $Runs) {
            & .\resume_reproduction.ps1 -Run $run -AdditionalSteps $AdditionalSteps -Note 'Long-budget comparison of curriculum with constant versus halved learning rate; selected from the 2026-09-26 v2 validation.'
        }
        $models = @($Runs | ForEach-Object { Join-Path $_ 'models\best.zip' })
        & $python -u src/evaluate_reproduction.py --models @models --published --games $EvaluationGames --seed 810000 --output $comparison
        if ($LASTEXITCODE -ne 0) { throw "Post-training comparison failed: $LASTEXITCODE" }
    }
    Write-Host "Comparison: $EvaluationGames natural-start games per policy, seeds beginning at 810000."
    Write-Host "Report: $comparison\comparison.md"
    if ($Preview) { Write-Host 'Preview only: no training started and no run files changed.' }
} finally { Pop-Location }
