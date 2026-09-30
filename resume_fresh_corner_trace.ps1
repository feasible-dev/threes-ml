param(
    [string]$Run = 'results\runs\2026-09-29_210238_fresh_corner_initial_lr30_s74',
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/resume_fresh_corner_trace.py', '--run', $Run)
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Fresh corner continuation stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
