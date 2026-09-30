param(
    [ValidateRange(1, 2000000000)][int]$Steps = 18000000,
    [ValidateRange(0, 2147483647)][int]$Seed = 73,
    [string]$ResumeRun,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/fresh_corner_strong.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        $arguments += @('--steps', "$Steps", '--seed', "$Seed")
    }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Fresh corner training stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
