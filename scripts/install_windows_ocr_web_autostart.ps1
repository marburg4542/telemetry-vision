[CmdletBinding()]
param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$taskName = "Telemetry OCR Web Server"
$startScript = Join-Path $PSScriptRoot "start_windows_ocr_web.ps1"

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        if ($existing.State -eq "Running") {
            Stop-ScheduledTask -TaskName $taskName
        }
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Removed Windows auto-start task: $taskName"
    } else {
        Write-Host "Auto-start task is not installed."
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $startScript)) {
    throw "OCR web startup script not found: $startScript"
}

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$powerShellExe = Join-Path $PSHOME "powershell.exe"
$quotedScript = '"' + $startScript + '"'
$action = New-ScheduledTaskAction `
    -Execute $powerShellExe `
    -Argument "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File $quotedScript" `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited
# The two battery switches are not optional here, and they are not defaults.
# Task Scheduler ships with DisallowStartIfOnBatteries and StopIfGoingOnBatteries
# both on, which is sensible for a background sync job and wrong for this: the
# footage is shot in the field, the laptop that reads it is on battery there, and
# without these the server never starts at logon and is killed the moment the
# machine is unplugged. It fails in the worst possible way, too -- the task never
# runs, so the startup script never logs, and the PWA simply has nothing behind
# it with no line anywhere saying why.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Start the standalone Telemetry OCR PWA server automatically when this user logs in."

Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Installed and started Windows auto-start task: $taskName"
Write-Host "OCR PWA URL: http://127.0.0.1:8765"
