$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Start-Process $python -WindowStyle Hidden -WorkingDirectory $root -ArgumentList @(
    "-m", "adaptive_bot.binance_l2_collector"
)
Write-Host "Binance BTCUSDT L2 collector avviato. Stato: data/reports/binance_l2_collector.status.json"
