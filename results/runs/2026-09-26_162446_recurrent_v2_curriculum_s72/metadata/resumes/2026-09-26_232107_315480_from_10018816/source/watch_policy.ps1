param(
    [string]$Model = 'external_models\kiok\onnx_model.onnx',
    [ValidateRange(0, 2147483647)][int]$Seed = 730000,
    [ValidateRange(0.25, 60)][double]$Speed = 3,
    [switch]$Autoplay,
    [switch]$Features
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # The existing CPU environment includes Pygame; keep the training GPU free.
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'The CPU .venv environment is missing.' }
    $arguments = @('-u', 'src/inspect_policy.py', '--model', $Model, '--seed', "$Seed", '--speed', $Speed.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($Autoplay) { $arguments += '--autoplay' }
    if ($Features) { $arguments += '--features' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Policy inspector stopped ($LASTEXITCODE)." }
} finally { Pop-Location }
