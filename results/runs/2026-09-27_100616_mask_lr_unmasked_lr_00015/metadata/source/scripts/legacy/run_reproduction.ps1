# Historical launcher; see scripts/README.md. Uses current source unless run from an archived snapshot.
param(
    [ValidateSet('base', 'shaped')][string[]]$Variants = @('base', 'shaped'),
    [ValidateRange(0, 2147483647)][int[]]$Seeds = @(71),
    [ValidateRange(1, 2000000000)][int]$Steps = 2000000,
    [ValidateRange(1, 100000)][int]$EvaluationGames = 300,
    [switch]$Preview
)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ErrorActionPreference = 'Stop'
Push-Location $ProjectRoot
try {
    $python = Join-Path $ProjectRoot '.venv-repro\Scripts\python.exe'
    $stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
    $models = @()
    if (-not $Preview) {
        if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
        if (-not (Test-Path -LiteralPath 'external_models\kiok\onnx_model.onnx')) {
            throw 'The published ONNX reference is missing; see external_models/kiok/README.md.'
        }
    }
    Write-Host "Published-architecture recurrent PPO | $Steps transitions per run"
    Write-Host 'New runs use random weights. Runs execute sequentially; 128 games run together within each run.'
    foreach ($variant in $Variants) {
        foreach ($seed in $Seeds) {
            $run = "results\runs\${stamp}_recurrent_${variant}_s${seed}"
            $config = "configs\reproduction_${variant}.json"
            Write-Host "`n$run | $config | training seed $seed"
            if (-not $Preview) {
                & $python -u src/train_reproduction.py --config $config --seed $seed --steps $Steps --run-dir $run
                if ($LASTEXITCODE -ne 0) { throw "Training stopped ($LASTEXITCODE). Resume from the run's last complete checkpoint." }
            }
            $models += "$run\models\best.zip"
        }
    }
    $comparison = "results\comparisons\${stamp}_recurrent"
    Write-Host "`nComparison: $EvaluationGames games per model and published reference, validation seeds starting at 800000"
    Write-Host "Reports: $comparison"
    if ($Preview) {
        Write-Host 'Preview only. Remove -Preview to launch in this terminal.'
    } else {
        & $python -u src/evaluate_reproduction.py --models @models --published --games $EvaluationGames --seed 800000 --output $comparison
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $LASTEXITCODE" }
    }
} finally { Pop-Location }
