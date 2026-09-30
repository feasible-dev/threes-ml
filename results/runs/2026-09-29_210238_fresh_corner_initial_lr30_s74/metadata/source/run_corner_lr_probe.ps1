param(
    [string]$ParentFork = 'results\runs\2026-09-28_230211_edge_corner_lr25_500m_masked_lr_000025_edge_corner',
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 1000000,
    [string]$Group,
    [string]$ResumeRun,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/corner_lr_probe.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        $arguments += @('--parent-fork', $ParentFork, '--additional-steps', "$AdditionalSteps")
    }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Corner LR probe stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
