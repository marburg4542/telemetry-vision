@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_windows_ocr_web_autostart.ps1"
if errorlevel 1 (
  echo.
  echo Failed to install Telemetry OCR Web auto-start.
  pause
  exit /b 1
)
echo.
echo Telemetry OCR Web will start automatically after Windows Login.
echo Open http://127.0.0.1:8765 to install or use the PWA.
pause
