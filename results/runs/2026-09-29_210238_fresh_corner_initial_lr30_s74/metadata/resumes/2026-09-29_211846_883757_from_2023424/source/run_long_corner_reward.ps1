param(
    [string]$ParentFork = 'results\runs\2026-09-28_223320_edge_corner_398m_masked_lr_00005_edge_corner',
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 500000000,
    [string]$Group,
    [string]$ResumeRun,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/long_corner_reward.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        $arguments += @('--parent-fork', $ParentFork, '--additional-steps', "$AdditionalSteps")
    }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Long edge/corner continuation stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
