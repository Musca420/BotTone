param([ValidateRange(0, 8)][int]$VipLevel = 0)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$pidPath = Join-Path $root "data\reports\btc_vwap_forward_audit.pid"
$env:BITUNIX_VIP_LEVEL = "$VipLevel"

if (Test-Path $pidPath) {
    $workerPid = [int](Get-Content -Raw $pidPath)
    if (Get-Process -Id $workerPid -ErrorAction SilentlyContinue) {
        Write-Host "BTC VWAP forward audit gia attivo (PID $workerPid)."
        exit 0
    }
}

$worker = Start-Process $python -PassThru -WindowStyle Hidden -WorkingDirectory $root -ArgumentList @(
    "-m", "adaptive_bot.btc_cross_exchange_forward_audit"
)
[IO.File]::WriteAllText($pidPath, [string]$worker.Id)
Write-Host "BTC VWAP forward audit avviato per Bitunix VIP0-VIP5: http://127.0.0.1:8080/?profile=musca-v5-vip0"
Write-Host "Worker PID $($worker.Id); un secondo avvio non crea duplicati."
Write-Host "Report: data/reports/btc_cross_exchange_forward_audit.json"
