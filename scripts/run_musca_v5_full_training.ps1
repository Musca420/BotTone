$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$status = "data/reports/btc_vwap_alpha_v1.status.json"
$stdout = "data/logs/musca-v5-full-training.stdout.log"
$stderr = "data/logs/musca-v5-full-training.stderr.log"
New-Item -ItemType Directory -Force -Path "data/logs", "data/reports" | Out-Null

$worker = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "adaptive_bot.btc_vwap_alpha" `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

while (-not $worker.HasExited) {
    Clear-Host
    Write-Host "MUSCA V5 - FULL ECONOMIC ALPHA GPU TRAINING"
    Write-Host (Get-Date -Format "dd/MM/yyyy HH:mm:ss")
    if (Test-Path $status) {
        $progress = Get-Content $status -Raw | ConvertFrom-Json
        Write-Host ("FASE: {0}" -f $progress.phase)
        Write-Host ("BLOCCO: {0}" -f $progress.detail)
        Write-Host ("AVANZAMENTO: {0:N1}%" -f $progress.percent)
    } else {
        Write-Host "FASE: inizializzazione"
    }
    $process = Get-Process -Id $worker.Id -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host ("PID {0} | CPU {1:N1} s | RAM {2:N2} GB" -f `
            $process.Id, $process.CPU, ($process.WorkingSet64 / 1GB))
    }
    Start-Sleep -Seconds 5
    $worker.Refresh()
}

Clear-Host
Write-Host "MUSCA V5 - TRAINING CONCLUSO"
if (Test-Path $status) { Get-Content $status }
if ($worker.ExitCode -ne 0) {
    Write-Host "Worker fallito. Ultime righe stderr:"
    Get-Content $stderr -Tail 40
}
Write-Host "Premi Invio per chiudere."
Read-Host | Out-Null
