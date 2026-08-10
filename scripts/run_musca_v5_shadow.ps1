$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

Start-Process $python -WindowStyle Hidden -WorkingDirectory $root -ArgumentList @(
    "-m", "adaptive_bot.musca_v5_shadow"
)
Write-Host "MUSCA V5 stable multi-horizon shadow: http://127.0.0.1:8080/?profile=musca-v5"
