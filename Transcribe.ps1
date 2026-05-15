# Interactive wrapper around transcribe.py with ETA + stall watchdog.
# Usage:
#   .\Transcribe.ps1                              # prompts for source
#   .\Transcribe.ps1 "https://youtu.be/..."       # one-shot
#   .\Transcribe.ps1 "C:\path\to\video.mp4"
#   .\Transcribe.ps1 -Source "..." -Model large-v3-turbo -BatchSize 8
#   .\Transcribe.ps1 -Loop                        # keep prompting after each
#   .\Transcribe.ps1 -StallSeconds 120            # watchdog threshold (default 90)
#   .\Transcribe.ps1 -SpeedFactor 6               # ETA divisor; raise for faster GPUs
#
# If you get an execution-policy error, run once in PowerShell:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Source,

    [string]$Model = "large-v3",
    [int]$BatchSize = 2,
    [string]$Language,
    [string]$ComputeType,
    [switch]$NoFastpath,
    [switch]$Loop,

    # Watchdog: print a heartbeat after this many seconds without stdout activity.
    [int]$StallSeconds = 90,

    # ETA = audioSec / SpeedFactor. Calibrated for RTX 3050 Laptop + large-v3 + int8_float16.
    # Bump to 8-12 for RTX 3060+, 25+ for RTX 4090.
    [double]$SpeedFactor = 4.0
)

$ErrorActionPreference = "Stop"

# --- 1. Refresh PATH so winget-installed tools (ffmpeg/ffprobe) are visible
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path", "User")

# --- 1b. Force unbuffered Python stdout so [acquire]/[whisper]/[align] markers
# stay in chronological order when captured by the watchdog event handlers.
$env:PYTHONUNBUFFERED = "1"

# --- 2. Locate venv + script (relative to THIS file, so it works from any cwd)
$root       = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$pyScript   = Join-Path $root "transcribe.py"

if (-not (Test-Path $venvPython)) {
    Write-Host "venv not found at $venvPython" -ForegroundColor Red
    Write-Host "Run the setup steps from README.md first." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $pyScript)) {
    Write-Host "transcribe.py not found at $pyScript" -ForegroundColor Red
    exit 1
}

# ---------- helpers ----------

function Get-MediaDurationSeconds {
    param([string]$source)
    try {
        if ($source -match '^https?://') {
            $out = & $venvPython -m yt_dlp --no-warnings --skip-download --print duration $source 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $first = ($out -split "`n" | Where-Object { $_ -match '^\d' } | Select-Object -First 1)
                if ($first) { return [double]$first.Trim() }
            }
        } elseif (Test-Path $source) {
            $out = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $source 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                return [double]([string]$out).Trim()
            }
        }
    } catch {}
    return $null
}

function Format-Hms {
    param([double]$sec)
    if ($sec -lt 0) { $sec = 0 }
    $ts = [TimeSpan]::FromSeconds($sec)
    if ($ts.TotalHours -ge 1) { return ("{0:hh\:mm\:ss}" -f $ts) }
    return ("{0:mm\:ss}" -f $ts)
}

function Get-GpuStatus {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    try {
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if (-not $line) { return $null }
        $parts = ([string]$line).Trim() -split ',\s*'
        return [pscustomobject]@{
            Util = [int]$parts[0]
            UsedMB = [int]$parts[1]
            TotalMB = [int]$parts[2]
        }
    } catch { return $null }
}

function ConvertTo-EscapedArg([string]$a) {
    # Quote args containing whitespace or quotes; escape inner quotes.
    if ($a -match '\s' -or $a -match '"') {
        return '"' + ($a -replace '"', '\"') + '"'
    }
    return $a
}

# ---------- main ----------

function Invoke-OneTranscription([string]$target) {
    $target = $target.Trim().Trim('"').Trim("'")
    if ([string]::IsNullOrWhiteSpace($target)) {
        Write-Host "No input given." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host "Source : $target" -ForegroundColor Cyan
    Write-Host "Model  : $Model   batch_size: $BatchSize" -ForegroundColor DarkGray

    Write-Host "Probing duration..." -ForegroundColor DarkGray
    $dur = Get-MediaDurationSeconds $target
    if ($dur) {
        $eta = $dur / $SpeedFactor
        Write-Host ("Audio  : {0}   ETA: ~{1}  (at ~{2:N1}x realtime; YouTube fast-path may finish in seconds)" -f `
            (Format-Hms $dur), (Format-Hms $eta), $SpeedFactor) -ForegroundColor Cyan
    } else {
        Write-Host "Audio  : (couldn't probe duration -- ETA unknown)" -ForegroundColor DarkYellow
    }
    $gpu0 = Get-GpuStatus
    if ($gpu0) {
        Write-Host ("GPU    : {0}% util, {1}/{2} MB used" -f $gpu0.Util, $gpu0.UsedMB, $gpu0.TotalMB) -ForegroundColor DarkGray
    }
    Write-Host "Watchdog: heartbeat after ${StallSeconds}s of stdout silence" -ForegroundColor DarkGray
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host ""

    # Build python args
    $pyArgs = @($pyScript, "--model", $Model, "--batch-size", $BatchSize)
    if ($Language)    { $pyArgs += @("--language", $Language) }
    if ($ComputeType) { $pyArgs += @("--compute-type", $ComputeType) }
    if ($NoFastpath)  { $pyArgs += "--no-fastpath" }
    $pyArgs += $target

    # Spawn process with stdout/stderr streaming
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $venvPython
    $psi.Arguments = ($pyArgs | ForEach-Object { ConvertTo-EscapedArg $_ }) -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi

    # Shared state for the output handlers (event runspaces are isolated; use synchronized hashtable)
    $state = [hashtable]::Synchronized(@{
        LastOutput = Get-Date
        Lines      = 0
    })

    $sink = {
        if ($null -ne $EventArgs.Data) {
            [Console]::WriteLine($EventArgs.Data)
            $Event.MessageData.LastOutput = Get-Date
            $Event.MessageData.Lines++
        }
    }

    $oReg = Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived `
            -Action $sink -MessageData $state
    $eReg = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived `
            -Action $sink -MessageData $state

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$proc.Start()
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()

    try {
        while (-not $proc.HasExited) {
            Start-Sleep -Milliseconds 500
            $silentFor = ((Get-Date) - $state.LastOutput).TotalSeconds
            if ($silentFor -ge $StallSeconds) {
                $gpu = Get-GpuStatus
                $elapsed = Format-Hms $sw.Elapsed.TotalSeconds
                $msg = "[watchdog] elapsed $elapsed | silent {0:N0}s | pid $($proc.Id)" -f $silentFor
                if ($gpu) {
                    $msg += " | GPU {0}% / {1} MB" -f $gpu.Util, $gpu.UsedMB
                    if ($gpu.Util -eq 0 -and $gpu.UsedMB -lt 500) {
                        $msg += "  [WARN GPU idle, may be truly stalled]"
                    }
                }
                Write-Host $msg -ForegroundColor Yellow
                $state.LastOutput = Get-Date  # reset so we fire again only after another StallSeconds
            }
        }
    } finally {
        # Drain any final pending events before unregistering
        Start-Sleep -Milliseconds 200
        Unregister-Event -SourceIdentifier $oReg.Name -ErrorAction SilentlyContinue
        Unregister-Event -SourceIdentifier $eReg.Name -ErrorAction SilentlyContinue
        Remove-Job -Job $oReg -Force -ErrorAction SilentlyContinue
        Remove-Job -Job $eReg -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit()
    }

    $exit = $proc.ExitCode
    $sw.Stop()

    Write-Host ""
    if ($exit -eq 0) {
        $totalSec = $sw.Elapsed.TotalSeconds
        $msg = "Finished in {0}" -f (Format-Hms $totalSec)
        if ($dur -and $totalSec -gt 0) {
            $msg += " ({0:N1}x realtime)" -f ($dur / $totalSec)
        }
        Write-Host $msg -ForegroundColor Green
    } else {
        Write-Host "transcribe.py exited with code $exit" -ForegroundColor Red
    }
}

# ---------- entry ----------

if ($Source) {
    Invoke-OneTranscription $Source
    if (-not $Loop) { return }
}

while ($true) {
    Write-Host ""
    Write-Host "Enter a YouTube URL or local video path (blank or 'q' to quit):" -ForegroundColor Cyan
    $entry = Read-Host ">"
    if ([string]::IsNullOrWhiteSpace($entry) -or $entry -in @('q', 'quit', 'exit')) {
        Write-Host "Bye." -ForegroundColor DarkGray
        break
    }
    Invoke-OneTranscription $entry
    if (-not $Loop) { break }
}
