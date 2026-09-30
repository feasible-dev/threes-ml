param(
    [string]$ParentFork,
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 5000000,
    [string]$Group,
    [string]$ResumeRun,
    [string]$CompareGroup,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/run_masked_lr_probe.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } elseif ($CompareGroup) {
        $arguments += @('--compare-group', $CompareGroup)
    } else {
        if (-not $ParentFork) { throw 'ParentFork is required for a new LR probe.' }
        $arguments += @('--parent-fork', $ParentFork, '--additional-steps', "$AdditionalSteps")
    }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "LR probe launcher stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
