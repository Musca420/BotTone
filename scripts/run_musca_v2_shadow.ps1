$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Start-Process $python -WindowStyle Hidden -WorkingDirectory $root -ArgumentList @(
    "-m", "adaptive_bot.musca_v2"
)
Write-Host "MUSCA V2 shadow avviato nel menu Strategy profile: http://127.0.0.1:8080"
