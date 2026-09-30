param(
    [string]$Model = 'external_models\kiok\onnx_model.onnx',
    [ValidateRange(0, 2147483647)][int]$Seed = 730000,
    [ValidateRange(0.25, 60)][double]$Speed = 3,
    [switch]$Autoplay,
    [switch]$Features,
    [switch]$OurBest
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($OurBest) {
        if ($PSBoundParameters.ContainsKey('Model')) { throw 'Choose either -OurBest or -Model.' }
        $Model = 'results\runs\2026-09-28_230211_edge_corner_lr25_500m_masked_lr_000025_edge_corner\models\checkpoints\step_000536756224\best.zip'
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model file is missing: $Model" }
    # The existing CPU environment includes Pygame; keep the training GPU free.
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'The CPU .venv environment is missing.' }
    $arguments = @('-u', 'src/inspect_policy.py', '--model', $Model, '--seed', "$Seed", '--speed', $Speed.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($Autoplay) { $arguments += '--autoplay' }
    if ($Features) { $arguments += '--features' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Policy inspector stopped ($LASTEXITCODE)." }
} finally { Pop-Location }
