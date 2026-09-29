param(
    [string]$ParentRun,
    [long]$ParentSteps,
    [string]$ResumeRun,
    [string]$CompareGroup,
    [ValidateRange(1, 2000000000)][int]$AdditionalSteps = 5000000,
    [ValidateSet('unmasked_lr_00015', 'masked_lr_00015', 'unmasked_lr_00005', 'masked_lr_00005')][string]$Arm,
    [switch]$All,
    [string]$Group,
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup_reproduction.ps1 first.' }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', 'src/run_reproduction_forks.py')
    if ($CompareGroup) {
        $arguments += @('--compare-group', $CompareGroup)
    } elseif ($ResumeRun) {
        $arguments += @('--resume-run', $ResumeRun)
    } else {
        if (-not $ParentRun -or $ParentSteps -lt 1) { throw 'ParentRun and ParentSteps are required for new forks.' }
        $arguments += @('--parent-run', $ParentRun, '--parent-steps', "$ParentSteps",
                        '--additional-steps', "$AdditionalSteps")
    }
    if ($Arm) { $arguments += @('--arm', $Arm) }
    if ($All) { $arguments += '--all' }
    if ($Group) { $arguments += @('--group', $Group) }
    if ($Preview) { $arguments += '--preview' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Fork launcher stopped ($LASTEXITCODE). See the explanation above." }
} finally { Pop-Location }
