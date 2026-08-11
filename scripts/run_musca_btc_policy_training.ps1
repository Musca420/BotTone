param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$status = "data/reports/musca_btc_policy.status.json"
$stdout = "data/logs/musca-btc-policy.stdout.log"
$stderr = "data/logs/musca-btc-policy.stderr.log"
New-Item -ItemType Directory -Force -Path "data/logs", "data/reports" | Out-Null

$arguments = @("-m", "adaptive_bot.cli", "musca-btc-policy-train")
if (-not $Fresh) { $arguments += "--resume" }
$worker = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList $arguments -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru -WindowStyle Hidden

while (-not $worker.HasExited) {
    Clear-Host
    Write-Host "MUSCA BTC BINANCE - PARAMETERIZED MULTI-EXPERT POLICY"
    Write-Host (Get-Date -Format "dd/MM/yyyy HH:mm:ss")
    if (Test-Path $status) {
        try {
            $progress = Get-Content $status -Raw | ConvertFrom-Json
            Write-Host ("FASE: {0}" -f $progress.phase)
            Write-Host ("BLOCCO: {0}" -f $progress.detail)
            Write-Host ("AVANZAMENTO: {0:N1}%" -f $progress.percent)
            if ($progress.blocks_completed -ne $null) {
                Write-Host ("CHECKPOINT: {0}/{1}" -f `
                    $progress.blocks_completed, $progress.blocks_total)
            }
            if ($progress.rows -ne $null) {
                Write-Host ("PIANI/LABEL: {0:N0}" -f $progress.rows)
            }
            if ($progress.eta_seconds -ne $null) {
                Write-Host ("ETA: {0}" -f ([TimeSpan]::FromSeconds($progress.eta_seconds)))
            }
        } catch {
            Write-Host "Stato in aggiornamento atomico; nuovo tentativo tra 5 secondi."
        }
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
Write-Host "MUSCA BTC BINANCE - TRAINING CONCLUSO"
if (Test-Path $status) { Get-Content $status }
if ($worker.ExitCode -ne 0) {
    Write-Host "Worker fallito. Ultime righe stderr:"
    if (Test-Path $stderr) { Get-Content $stderr -Tail 80 }
}
Write-Host "Premi Invio per chiudere. Il worker e' gia' terminato."
Read-Host | Out-Null
