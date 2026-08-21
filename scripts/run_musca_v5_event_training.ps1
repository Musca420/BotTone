$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$status = "data/reports/musca_v5_event_policy.status.json"
$stdout = "data/logs/musca-v5-event-training.stdout.log"
$stderr = "data/logs/musca-v5-event-training.stderr.log"
New-Item -ItemType Directory -Force -Path "data/logs", "data/reports" | Out-Null

$worker = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "adaptive_bot.musca_v5_event_policy" `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

while (-not $worker.HasExited) {
    Clear-Host
    Write-Host "MUSCA V5 - EVENT POLICY GPU TRAINING"
    Write-Host (Get-Date -Format "dd/MM/yyyy HH:mm:ss")
    if (Test-Path $status) {
        Get-Content $status
    } else {
        Write-Host "Inizializzazione worker..."
    }
    $process = Get-Process -Id $worker.Id -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host ""
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
    Write-Host ""
    Write-Host "Worker fallito. Ultime righe stderr:"
    Get-Content $stderr -Tail 30
}
Write-Host ""
Write-Host "Premi Invio per chiudere."
Read-Host | Out-Null
