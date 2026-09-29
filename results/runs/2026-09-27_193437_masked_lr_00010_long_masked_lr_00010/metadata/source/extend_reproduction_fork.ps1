param(
    [Parameter(Mandatory = $true)][string]$ParentFork,
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 20000000,
    [string]$Group,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/extend_reproduction_fork.py', '--parent-fork', $ParentFork,
                   '--additional-steps', "$AdditionalSteps")
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Fork extension stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
