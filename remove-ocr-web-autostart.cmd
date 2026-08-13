@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_windows_ocr_web_autostart.ps1" -Remove
if errorlevel 1 (
  echo.
  echo Failed to remove Telemetry OCR Web auto-start.
  pause
  exit /b 1
)
echo.
echo Telemetry OCR Web auto-start was removed.
pause
