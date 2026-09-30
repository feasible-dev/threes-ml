param(
    [string]$Run,
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 10000000,
    [string]$Note = '',
    [switch]$List,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/resume_reproduction.py', '--additional-steps', "$AdditionalSteps")
    if ($List -or -not $Run) {
        $arguments += '--list'
    } else {
        $arguments += @('--run', $Run)
        if ($Note) { $arguments += @('--note', $Note) }
        if ($Preview) { $arguments += '--preview' }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Continuation stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
