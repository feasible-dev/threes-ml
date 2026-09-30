param(
    [ValidateSet('control', 'curriculum', 'decay', 'curriculum_decay')]
    [string[]]$Variants = @('control', 'curriculum', 'curriculum_decay'),
    [ValidateRange(0, 2147483647)][int[]]$Seeds = @(72),
    [ValidateRange(1, 2000000000)][int]$Steps = 10000000,
    [ValidateRange(1, 100000)][int]$EvaluationGames = 300,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
    $stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
    $models = @()
    if (($Variants | Select-Object -Unique).Count -ne $Variants.Count -or
        ($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
        throw 'Variants and seeds must not contain duplicates.'
    }
    Write-Host "Audited engine v2 | $Steps transitions per run | $($Variants.Count * $Seeds.Count) runs"
    Write-Host 'Random initial weights; RTX 3060; sequential learners, 128 games batched inside each learner.'
    Write-Host 'Natural-start validation every ~500k transitions. Final-test seeds remain unused.'
    Write-Host 'Control -> curriculum isolates scaffolding; curriculum -> curriculum_decay isolates LR halving at 5M.'
    if (-not $Preview) {
        if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
        if (-not (Test-Path -LiteralPath 'external_models\kiok\onnx_model.onnx')) {
            throw 'Published reference missing; see external_models/kiok/README.md.'
        }
    }
    foreach ($variant in $Variants) {
        foreach ($seed in $Seeds) {
            $run = "results\runs\${stamp}_recurrent_v2_${variant}_s${seed}"
            $config = "configs\reproduction_v2_${variant}.json"
            Write-Host "`n$run | $config | seed $seed"
            if (-not $Preview) {
                & $python -u src/train_reproduction.py --config $config --seed $seed --steps $Steps --run-dir $run
                if ($LASTEXITCODE -ne 0) { throw "Training stopped ($LASTEXITCODE). Resume this v2 run from its last complete checkpoint." }
            }
            $models += "$run\models\best.zip"
        }
    }
    $comparison = "results\comparisons\${stamp}_recurrent_v2"
    Write-Host "`nComparison: $EvaluationGames natural games per agent, including the published reference; seeds start at 800000."
    Write-Host "Report: $comparison\comparison.md"
    if ($Preview) {
        Write-Host 'Preview only. Remove -Preview to run visibly in this terminal.'
    } else {
        & $python -u src/evaluate_reproduction.py --models @models --published --games $EvaluationGames --seed 800000 --output $comparison
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $LASTEXITCODE" }
    }
} finally { Pop-Location }
