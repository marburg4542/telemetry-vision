[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$healthUrl = "http://127.0.0.1:8765/api/health"

# CUDA is roughly three times faster for OCR and seven for YOLO on this machine,
# so it is preferred, but a CPU-only install still has to start.
$candidates = @(
    @{ Name = "CUDA"; Exe = ".venv-cuda\Scripts\telemetry-ocr.exe"; Config = "configs\roi_calibrated_cuda.yaml" },
    @{ Name = "CPU";  Exe = ".venv\Scripts\telemetry-ocr.exe";      Config = "configs\roi_calibrated.yaml" }
)
$logDir = Join-Path $projectDir "output\logs"
$logFile = Join-Path $logDir "ocr-web-autostart.log"

# The scheduled task runs hidden, so a bare throw leaves no trace of why the
# PWA has nothing to talk to. Every exit path is recorded here instead.
function Write-StartupLog {
    param([string]$Message)
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logFile -Value $line -Encoding utf8
}

try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
    if ($health.status -eq "ready") {
        Write-StartupLog "Server already listening; nothing to start."
        exit 0
    }
} catch {
    # Nothing is listening yet; start the local OCR web server below.
}

$selected = $null
foreach ($candidate in $candidates) {
    $exe = Join-Path $projectDir $candidate.Exe
    $config = Join-Path $projectDir $candidate.Config
    if ((Test-Path -LiteralPath $exe) -and (Test-Path -LiteralPath $config)) {
        $selected = @{ Name = $candidate.Name; Exe = $exe; Config = $config }
        break
    }
    Write-StartupLog "$($candidate.Name) build unavailable (need $exe and $config)"
}

if ($null -eq $selected) {
    Write-StartupLog "MISSING every known build. Create one with:"
    Write-StartupLog "  py -3.12 -m venv .venv-cuda"
    Write-StartupLog "  .venv-cuda\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    Write-StartupLog "  .venv-cuda\Scripts\python.exe -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126"
    Write-StartupLog "  .venv-cuda\Scripts\python.exe -m pip install -e `".[ocr,web,ai]`" nvidia-cudnn-cu12==9.9.0.52"
    Write-StartupLog "Startup aborted."
    exit 1
}

$ocrExe = $selected.Exe
$ocrConfig = $selected.Config

# Recorded because the battery is what stopped this task from running at all
# before the two settings switches were added to the installer. If it ever
# happens again -- an older task registration, a group policy, a power plan --
# this line is the difference between a diagnosable fault and a PWA that talks
# to nothing for no visible reason.
try {
    $battery = Get-CimInstance Win32_Battery -ErrorAction Stop | Select-Object -First 1
    if ($null -eq $battery) {
        Write-StartupLog "Power: no battery detected (desktop or docked)."
    } else {
        # BatteryStatus 2 is "AC connected"; everything else is running on cells.
        $onAc = $battery.BatteryStatus -eq 2
        $source = if ($onAc) { "AC" } else { "battery" }
        Write-StartupLog "Power: running on $source ($($battery.EstimatedChargeRemaining)% charge)."
    }
} catch {
    Write-StartupLog "Power: could not be read ($($_.Exception.Message))."
}
Set-Location -LiteralPath $projectDir

$arguments = @("web", "--config", $ocrConfig, "--host", "127.0.0.1", "--port", "8765")

# A marker file rather than a flag baked into the scheduled task, so turning
# image sending on or off does not mean editing and reinstalling the task. Its
# presence is the whole switch; the contents are ignored.
$quickTunnelFlag = Join-Path $projectDir "configs\quick-tunnel.enabled"
if (Test-Path -LiteralPath $quickTunnelFlag) {
    $arguments += "--quick-tunnel"
    Write-StartupLog "Quick tunnel enabled: capture images will be fetchable by LINE."
}

Write-StartupLog "Starting the $($selected.Name) build: $ocrExe with $ocrConfig"
& $ocrExe @arguments

Write-StartupLog "Server exited with code $LASTEXITCODE"
exit $LASTEXITCODE
