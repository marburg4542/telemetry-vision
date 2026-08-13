[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$ocrExe = Join-Path $projectDir ".venv\Scripts\telemetry-ocr.exe"
$ocrConfig = Join-Path $projectDir "configs\roi_calibrated.yaml"
$healthUrl = "http://127.0.0.1:8765/api/health"

try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
    if ($health.status -eq "ready") {
        exit 0
    }
} catch {
    # Nothing is listening yet; start the local OCR web server below.
}

if (-not (Test-Path -LiteralPath $ocrExe)) {
    throw "Telemetry OCR executable not found: $ocrExe"
}
if (-not (Test-Path -LiteralPath $ocrConfig)) {
    throw "Calibrated ROI config not found: $ocrConfig"
}

Set-Location -LiteralPath $projectDir
& $ocrExe web `
    --config $ocrConfig `
    --host 127.0.0.1 `
    --port 8765

exit $LASTEXITCODE
