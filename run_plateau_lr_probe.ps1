param(
    [string]$ParentFork = 'results\runs\2026-09-27_193437_masked_lr_00010_long_masked_lr_00010',
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 10000000,
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
    $arguments = @('-u', 'src/run_plateau_lr_probe.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } elseif ($CompareGroup) {
        $arguments += @('--compare-group', $CompareGroup)
    } else {
        $arguments += @('--parent-fork', $ParentFork, '--additional-steps', "$AdditionalSteps")
    }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Plateau LR probe stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
