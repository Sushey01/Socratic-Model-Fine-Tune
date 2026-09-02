# Windows entry: install uv + Python 3.12 if missing, then run start.py.
# Usage:  powershell -ExecutionPolicy Bypass -File .\start.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Add-PathDir([string]$Dir) {
    if ($Dir -and (Test-Path $Dir) -and ($env:Path -notlike "*$Dir*")) {
        $env:Path = "$Dir;$env:Path"
    }
}

Add-PathDir "$env:USERPROFILE\.local\bin"
Add-PathDir "$env:USERPROFILE\.cargo\bin"
Add-PathDir "$env:LOCALAPPDATA\uv"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    Add-PathDir "$env:USERPROFILE\.local\bin"
    Add-PathDir "$env:USERPROFILE\.cargo\bin"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not on PATH. Open a new PowerShell and run: uv run python start.py"
    exit 1
}

Write-Host "Ensuring Python 3.12 (uv)..."
& uv python install 3.12
& uv python pin 3.12

& uv run --python 3.12 python start.py @args
$uvExit = $LASTEXITCODE
if ($uvExit -eq 0) { exit 0 }

# Application Control (os error 4551) often blocks .venv\Scripts\python.exe
$managed = (& uv python find 3.12 2>$null | Select-Object -First 1)
if ($managed -and (Test-Path $managed)) {
    Write-Host "Retrying with uv-managed Python (bypasses blocked .venv python)..."
    $site = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
    $trainSite = Join-Path $PSScriptRoot ".venv-train\Lib\site-packages"
    if (Test-Path $trainSite) { $env:PYTHONPATH = $trainSite } elseif (Test-Path $site) { $env:PYTHONPATH = $site }
    & $managed (Join-Path $PSScriptRoot "start.py") @args
    exit $LASTEXITCODE
}

exit $uvExit
