@echo off
title QUANTUM NIDS - LIVE ACTIVE DEFENDER
cd /d "%~dp0"
cls
echo ======================================================================
echo      HYBRID QUANTUM NIDS - LIVE AUTONOMOUS ACTIVE DEFENDER
echo ======================================================================
echo.
echo Select Mode:
echo   [1] Live Presentation Mode - Workstation Lock Countdown on Attack
echo   [2] Safe Dry-Run Mode - Threat Alert Popup, Screen Lock Disabled
echo.
set "mode=1"
set /p "mode=Choose Mode [1 or 2] (Press Enter for Live Mode 1): "
echo.

set "DEFENDER_SCRIPT=live_defender.py"
if not exist "%DEFENDER_SCRIPT%" (
    if exist "simulators\live_defender.py" set "DEFENDER_SCRIPT=simulators\live_defender.py"
)

if "%mode%"=="2" (
    echo [*] Launching in Safe Dry-Run Mode...
    python "%DEFENDER_SCRIPT%" --no-lock
) else (
    echo [*] Launching in Live Presentation Mode - Lockdown Enabled...
    python "%DEFENDER_SCRIPT%"
)

echo.
pause
