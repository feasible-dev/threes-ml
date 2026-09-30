param(
    [ValidateRange(1, 2000000000)][int]$Steps = 100000000,
    [ValidateRange(0, 2147483647)][int]$Seed = 74,
    [string]$ResumeRun,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/fresh_corner_initial.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        $arguments += @('--steps', "$Steps", '--seed', "$Seed")
    }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Fresh initial-LR corner training stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
