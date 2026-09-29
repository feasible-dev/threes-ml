param()
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv-repro\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        $bootstrap = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $bootstrap)) { $bootstrap = 'python' }
        & $bootstrap -m venv .venv-repro
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-repro' }
    }
    & $python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
    & $python -m pip install -r requirements-reproduction.txt
    if ($LASTEXITCODE -ne 0) { throw 'Reproduction dependencies installation failed' }
    & $python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed' }
    Write-Host 'Ready. Preview experiments with .\run_reproduction.ps1 -Preview'
} finally { Pop-Location }
