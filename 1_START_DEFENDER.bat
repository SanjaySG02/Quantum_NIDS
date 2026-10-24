@echo off
title QUANTUM NIDS - LIVE ACTIVE DEFENDER
cd /d "%~dp0"
cls
echo ======================================================================
echo      HYBRID QUANTUM NIDS - LIVE AUTONOMOUS ACTIVE DEFENDER
echo ======================================================================
echo.
echo Select Mode:
echo   [1] Live Presentation Mode (Workstation Lock Countdown on Attack)
echo   [2] Safe Dry-Run Mode (Threat Alert Popup, Screen Lock Disabled)
echo.
set "mode=1"
set /p mode="Choose Mode [1 or 2] (Press Enter for Live Mode 1): "
echo.
if "%mode%"=="2" (
    echo [*] Launching in Safe Dry-Run Mode...
    python live_defender.py --no-lock
) else (
    echo [*] Launching in Live Presentation Mode (Lockdown Enabled)...
    python live_defender.py
)
echo.
pause
