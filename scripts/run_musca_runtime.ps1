param(
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$statusPath = Join-Path $root "data\reports\musca_auto_moe_runtime.status.json"
$mutex = [Threading.Mutex]::new($false, "Local\BotToneMuscaAutoMoeRuntimeSupervisor")

if (-not $mutex.WaitOne(0)) {
    exit 0
}

function Write-RuntimeStatus {
    param([hashtable]$Payload)
    $Payload["updated_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $directory = Split-Path -Parent $statusPath
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporary = "$statusPath.$PID.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        ($Payload | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Get-RuntimeProcesses {
    @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -in @("python.exe", "adaptive-bot.exe", "cmd.exe") -and $_.CommandLine
    })
}

function Ensure-PythonWorker {
    param(
        [string]$Name,
        [string]$Pattern,
        [string[]]$Arguments,
        [object[]]$Processes
    )
    $existing = @($Processes | Where-Object { $_.CommandLine -match $Pattern })
    if ($existing.Count -gt 0) {
        return @{name=$Name; state="running"; pids=@($existing.ProcessId)}
    }
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru
    return @{name=$Name; state="started"; pids=@($process.Id)}
}

try {
    do {
        $workers = @()
        $errorMessage = $null
        try {
            $processes = Get-RuntimeProcesses
            $workers += Ensure-PythonWorker "binance_l2" `
                "adaptive_bot\.binance_l2_collector" `
                @("-m", "adaptive_bot.binance_l2_collector") $processes
            $workers += Ensure-PythonWorker "musca_btc_auto_moe_paper" `
                "adaptive_bot\.musca_v8_binance" `
                @(
                    "-m", "adaptive_bot.musca_v8_binance",
                    "--config", "configs\binance_btcusdt_paper.yaml"
                ) $processes

            $dashboard = Get-NetTCPConnection -LocalPort 8080 -State Listen `
                -ErrorAction SilentlyContinue
            if ($null -eq $dashboard) {
                $process = Start-Process -FilePath "cmd.exe" `
                    -ArgumentList "/c", "scripts\run_dashboard_btc.cmd" `
                    -WorkingDirectory $root -WindowStyle Hidden -PassThru
                $workers += @{name="dashboard_8080"; state="started"; pids=@($process.Id)}
            } else {
                $workers += @{
                    name="dashboard_8080"
                    state="running"
                    pids=@($dashboard.OwningProcess)
                }
            }
        } catch {
            $errorMessage = $_.Exception.Message
        }

        Write-RuntimeStatus @{
            phase = if ($errorMessage) {"degraded"} else {"running"}
            detail = if ($errorMessage) {$errorMessage} else {"Musca BTC Auto-MoE Binance paper runtime supervised"}
            supervisor_pid = $PID
            workers = $workers
            persistent_state = @(
                "data/research/musca_btc_auto_moe_paper_state.json",
                "data/reports/musca_btc_auto_moe.json",
                "data/models/musca_btc_auto_moe/research_bundle.joblib"
            )
        }
        if ($Once) {
            break
        }
        Start-Sleep -Seconds 60
    } while ($true)
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
