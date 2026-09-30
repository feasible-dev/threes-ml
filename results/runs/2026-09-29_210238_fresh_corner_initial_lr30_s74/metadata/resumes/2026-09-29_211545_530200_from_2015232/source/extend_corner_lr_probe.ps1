param(
    [string]$ParentFork = 'results\runs\2026-09-29_185906_corner_lr125_1m_masked_lr_0000125_edge_corner',
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 100000000,
    [string]$Group,
    [string]$ResumeRun,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/extend_corner_lr_probe.py')
    if ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        $arguments += @('--parent-fork', $ParentFork, '--additional-steps', "$AdditionalSteps")
    }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Long corner LR extension stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
