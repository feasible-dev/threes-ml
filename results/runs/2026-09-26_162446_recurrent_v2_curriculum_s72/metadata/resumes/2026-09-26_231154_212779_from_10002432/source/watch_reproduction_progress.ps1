param(
    [Parameter(Mandatory=$true)][string]$Run,
    [switch]$Follow
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'The project Python environment is missing.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/reproduction_progress.py', $Run)
    if ($Follow) { $arguments += '--follow' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Progress display stopped ($LASTEXITCODE)." }
} finally { Pop-Location }
