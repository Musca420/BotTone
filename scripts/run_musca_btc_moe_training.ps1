param(
    [switch]$ForceMatrix
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$status = "data/reports/musca_btc_moe.status.json"
$stdout = "data/logs/musca-btc-moe.stdout.log"
$stderr = "data/logs/musca-btc-moe.stderr.log"
New-Item -ItemType Directory -Force -Path "data/logs", "data/reports" | Out-Null

$arguments = @("-m", "adaptive_bot.musca_btc_moe")
if ($ForceMatrix) { $arguments += "--force-matrix" }
$worker = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList $arguments -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

while (-not $worker.HasExited) {
    Clear-Host
    Write-Host "MUSCA BTC - MIXTURE OF EXPERTS GPU"
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
    $gpu = & nvidia-smi `
        --query-gpu=utilization.gpu,memory.used,memory.total `
        --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        Write-Host ("GPU: {0}% | VRAM: {1}/{2} MiB" -f ($gpu -split ", "))
    }
    Start-Sleep -Seconds 5
    $worker.Refresh()
}

Clear-Host
Write-Host "MUSCA BTC - TRAINING CONCLUSO"
if (Test-Path $status) { Get-Content $status }
if ($worker.ExitCode -ne 0) {
    Write-Host "Worker fallito. Ultime righe stderr:"
    Get-Content $stderr -Tail 50
}
Write-Host "Premi Invio per chiudere."
Read-Host | Out-Null
