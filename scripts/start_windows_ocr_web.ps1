[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$ocrExe = Join-Path $projectDir ".venv\Scripts\telemetry-ocr.exe"
$ocrConfig = Join-Path $projectDir "configs\roi_calibrated.yaml"
$healthUrl = "http://127.0.0.1:8765/api/health"
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

$missing = @()
if (-not (Test-Path -LiteralPath $ocrExe)) {
    $missing += "virtual environment executable: $ocrExe (create it with: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -e `".[ocr,web,ai]`")"
}
if (-not (Test-Path -LiteralPath $ocrConfig)) {
    $missing += "calibrated ROI config: $ocrConfig"
}
if ($missing.Count -gt 0) {
    foreach ($item in $missing) {
        Write-StartupLog "MISSING $item"
    }
    Write-StartupLog "Startup aborted. Fix the items above, then run this script again."
    exit 1
}

Set-Location -LiteralPath $projectDir
Write-StartupLog "Starting $ocrExe with $ocrConfig"
& $ocrExe web `
    --config $ocrConfig `
    --host 127.0.0.1 `
    --port 8765

Write-StartupLog "Server exited with code $LASTEXITCODE"
exit $LASTEXITCODE
